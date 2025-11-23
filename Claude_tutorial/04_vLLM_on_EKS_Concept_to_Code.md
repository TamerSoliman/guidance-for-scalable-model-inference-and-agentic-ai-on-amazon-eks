# vLLM on EKS: Concept-to-Code Reference Guide

## Target Audience
Generative AI researchers with limited background in cloud infrastructure, MLOps, or Kubernetes.

## Purpose
This guide maps high-level concepts of containerized LLM inference to specific lines of code and configuration in the repository.

---

## Part 1: The Big Picture - What Are We Building?

### Mental Model for AI Researchers

Think of deploying vLLM on EKS like setting up a research lab:

1. **The Building** (EKS Cluster): Physical space with power, networking, security
2. **Lab Equipment Procurement** (Karpenter): Automatically orders GPU workstations when needed
3. **Compute Workstations** (EC2 GPU Instances): The physical machines with GPUs
4. **Software Environment** (Docker Container): Isolated environment with vLLM, CUDA, Python
5. **Model Serving** (vLLM Process): The actual inference engine running your model
6. **Front Door** (Kubernetes Service): How researchers access your model API

### Why Not Just `python run_vllm.py`?

**Research Setup:**
```python
# On your local machine
vllm serve Qwen/Qwen3-14B --tensor-parallel-size 4
```

**Production Requirements:**
- ✅ Auto-restart if crash
- ✅ Handle 1000+ concurrent requests
- ✅ Scale up/down based on demand
- ✅ Monitor performance and costs
- ✅ Secure network access
- ✅ Persistent model cache
- ✅ Zero-downtime updates

→ **This is why we use Kubernetes**

---

## Part 2: Anatomy of a vLLM Deployment

### Component Mapping Table

| What You Want | Kubernetes Resource | File Location | Purpose |
|---------------|---------------------|---------------|---------|
| "Run my model on GPU" | Deployment | `model-hosting/standalone-vllm-reasoning.yaml` (lines 17-109) | Defines the containerized workload |
| "Provision GPU instances automatically" | Karpenter NodePool | `base_eks_setup/karpenter_nodepool/gpu-nodepool.yaml` (lines 1-48) | Auto-scales GPU nodes |
| "Store model weights permanently" | PersistentVolumeClaim | `model-hosting/standalone-vllm-reasoning.yaml` (lines 3-15) | Requests 900GB EBS volume |
| "Give me a stable endpoint" | Service | `model-hosting/standalone-vllm-reasoning.yaml` (lines 111-126) | Creates ClusterIP for load balancing |
| "Configure vLLM parameters" | Container args | `model-hosting/standalone-vllm-reasoning.yaml` (lines 58-60) | vLLM serve command with flags |

---

## Part 3: Deep Dive - How Model Loading Works

### Conceptual Flow

```
User applies YAML → Kubernetes creates Pod → Pod requests 4 GPUs
→ Karpenter provisions g5.12xlarge → Node joins cluster
→ Pod scheduled to node → Container starts → vLLM downloads model from HF
→ Model loaded to GPU → Service becomes ready → Requests can be served
```

### Code Walkthrough: Model Download & Caching

#### Step 1: Container Starts with Hugging Face Token

**File:** `model-hosting/standalone-vllm-reasoning.yaml`

**Lines 63-67:**
```yaml
env:
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token
      key: token
```

**What This Does:**
- Injects your Hugging Face token into the container as an environment variable
- The `transformers` library automatically uses this token for authenticated downloads

**Why It Matters:**
- Without this, you can't download gated models (like Llama)
- Token is stored in Kubernetes Secret (encrypted at rest)

#### Step 2: vLLM Command Specifies Model

**Lines 58-60:**
```yaml
args: [
  "vllm serve Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes
   --trust-remote-code --max-num-batched-tokens 32768  --max-num-seqs 8
   --max-model-len 32768 --dtype bfloat16 --tensor-parallel-size 4
   --gpu-memory-utilization 0.90"
]
```

**Breaking Down the Command:**

| Parameter | Code Value | What It Means | Why It Matters |
|-----------|-----------|---------------|----------------|
| **Model ID** | `Qwen/Qwen3-14B` | Model repo on Hugging Face | vLLM downloads from `huggingface.co/Qwen/Qwen3-14B` |
| **Tensor Parallel** | `--tensor-parallel-size 4` | Split model across 4 GPUs | Model is ~28GB, each GPU gets ~7GB |
| **Precision** | `--dtype bfloat16` | Use 16-bit precision | Reduces memory from 56GB (FP32) to 28GB |
| **Max Tokens** | `--max-num-batched-tokens 32768` | Total tokens across all concurrent requests | Higher = more throughput, more GPU memory |
| **Concurrency** | `--max-num-seqs 8` | Max 8 concurrent requests | Balance between throughput and latency |
| **GPU Util** | `--gpu-memory-utilization 0.90` | Use 90% of GPU memory | Leaves 10% headroom for safety |

**What Happens Internally:**

```python
# Pseudo-code of what vLLM does

# 1. Download model weights
model_path = download_from_huggingface("Qwen/Qwen3-14B", token=os.environ['HUGGING_FACE_HUB_TOKEN'])

# 2. Load model with tensor parallelism
model = LLM(
    model_path,
    tensor_parallel_size=4,  # Split across 4 GPUs
    dtype="bfloat16",        # 16-bit precision
    max_model_len=32768      # 32K context window
)

# 3. Start OpenAI-compatible API server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Step 3: Caching to Persistent Volume

**Lines 89-90:**
```yaml
volumeMounts:
- mountPath: /root/.cache/huggingface
  name: cache-volume
```

**What This Does:**
- Maps the Hugging Face cache directory to a persistent EBS volume
- Model weights downloaded once, reused across Pod restarts

**Directory Structure on Volume:**
```
/root/.cache/huggingface/
├── hub/
│   └── models--Qwen--Qwen3-14B/
│       ├── snapshots/
│       │   └── abc123.../  (git commit hash)
│       │       ├── config.json
│       │       ├── model-00001-of-00004.safetensors
│       │       ├── model-00002-of-00004.safetensors
│       │       ├── model-00003-of-00004.safetensors
│       │       └── model-00004-of-00004.safetensors
│       └── refs/
│           └── main  (points to abc123 snapshot)
```

**Time Savings:**
- First startup: ~5 minutes (download 28GB)
- Subsequent startups: ~2 minutes (load from cache)

---

## Part 4: GPU Resource Allocation

### Understanding the GPU Request

**File:** `model-hosting/standalone-vllm-reasoning.yaml`

**Lines 80-87:**
```yaml
resources:
  limits:
    memory: 64Gi
    nvidia.com/gpu: "4"
  requests:
    cpu: "22"
    memory: 64Gi
    nvidia.com/gpu: "4"
```

### What Happens When You Request 4 GPUs?

#### Step 1: Pod Creation
```bash
kubectl apply -f standalone-vllm-reasoning.yaml
```

#### Step 2: Kubernetes Scheduler
```
Scheduler: "Pod needs 4 GPUs. Let me find a node with 4 available GPUs..."
Scheduler: "No nodes found. Pod status = Pending"
```

#### Step 3: Karpenter Autoscaler
```
Karpenter: "Detected pending Pod needing 4 GPUs"
Karpenter: "Reading gpu-inference NodePool for requirements..."
Karpenter: "Requirements: GPU instances, g5/g6 family, 4 GPUs, On-Demand"
Karpenter: "Best match: g5.12xlarge (4x A10G GPUs, $5.67/hour)"
Karpenter: "Provisioning EC2 instance..."
```

**NodePool Configuration (from `base_eks_setup/karpenter_nodepool/gpu-nodepool.yaml`):**

Lines 26-48:
```yaml
requirements:
  - key: karpenter.k8s.aws/instance-family
    operator: In
    values: ["g5", "g6"]
  - key: karpenter.k8s.aws/instance-gpu-count
    operator: In
    values: ["4"]
```

**Instance Selection Logic:**

| Requirement | Filter Effect | Matching Instances |
|-------------|---------------|-------------------|
| `instance-family: g5` | Only G5 series | g5.xlarge, g5.2xlarge, g5.12xlarge, g5.48xlarge |
| `instance-gpu-count: 4` | Only 4-GPU instances | g5.12xlarge, g5.48xlarge |
| `capacity-type: on-demand` | Exclude Spot | g5.12xlarge (cheaper), g5.48xlarge |

**Karpenter selects:** `g5.12xlarge` (4x A10G GPUs, 48 vCPUs, 192GB RAM)

#### Step 4: Node Joins Cluster
```
EC2 Instance boots → Kubelet starts → Registers with Kubernetes control plane
→ NVIDIA GPU device plugin detects 4 GPUs → Advertises to scheduler
→ Node is Ready with 4 allocatable GPUs
```

#### Step 5: Pod Scheduled
```
Scheduler: "New node available with 4 GPUs!"
Scheduler: "Binding Pod to g5.12xlarge-xyz123"
```

#### Step 6: Container Starts
```bash
# On the g5.12xlarge node
docker run vllm/vllm-openai:latest \
  -e HUGGING_FACE_HUB_TOKEN=hf_xxx \
  -v /mnt/ebs-900gb:/root/.cache/huggingface \
  --gpus all \
  vllm serve Qwen/Qwen3-14B --tensor-parallel-size 4
```

---

## Part 5: Environment Variables - Configuration Injection

### How Configuration Reaches the Container

**Concept:** Environment variables are like function arguments for your container.

**File:** `model-hosting/standalone-vllm-reasoning.yaml`

**Lines 63-77:**
```yaml
env:
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token
      key: token
- name: OMP_NUM_THREADS
  value: "8"
- name: VLLM_LOGGING_LEVEL
  value: "DEBUG"
```

### Variable Resolution Flow

```
YAML (env section)
  → Kubernetes pulls value from Secret
  → Kubelet injects into container environment
  → vLLM process reads os.environ['HUGGING_FACE_HUB_TOKEN']
```

### Critical Environment Variables Explained

| Variable | Source | Purpose | Impact if Missing |
|----------|--------|---------|-------------------|
| `HUGGING_FACE_HUB_TOKEN` | Kubernetes Secret | Download model weights | Fails to download model |
| `OMP_NUM_THREADS` | Hardcoded value | CPU parallelism for tokenization | Slower preprocessing |
| `VLLM_LOGGING_LEVEL` | Hardcoded value | Log verbosity | Less debugging info |
| `CUDA_VISIBLE_DEVICES` | (commented) | Limit GPUs visible to vLLM | vLLM uses all GPUs |

### Example: Tracing Token Usage in Code

**Kubernetes YAML:**
```yaml
env:
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token
      key: token
```

**vLLM Internals (pseudo-code):**
```python
# Inside vLLM's model loader
import os
from huggingface_hub import snapshot_download

token = os.environ.get('HUGGING_FACE_HUB_TOKEN')

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-14B",
    use_auth_token=token  # ← Token injected here
)
```

---

## Part 6: Health Checks - Ensuring Reliability

### Why Health Checks Matter

**Problem:** Container might be running but model server is crashed/hung.

**Solution:** Kubernetes periodically checks HTTP endpoints to verify health.

**File:** `model-hosting/standalone-vllm-reasoning.yaml`

**Lines 93-109:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 240
  periodSeconds: 10
  failureThreshold: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 240
  periodSeconds: 10
```

### Liveness vs. Readiness Probes

| Probe Type | Question It Answers | Action on Failure |
|------------|---------------------|-------------------|
| **Liveness** | "Is the container alive?" | Kill and restart container |
| **Readiness** | "Is the container ready to serve traffic?" | Remove from Service endpoints |

### Timeline of Health Checks

```
t=0s:   Container starts
t=240s: First liveness check (initialDelaySeconds)
        GET http://localhost:8000/health
        Response: 200 OK {"status": "ready"}
        → Container is ALIVE

t=250s: Second liveness check (periodSeconds=10)
        Response: 200 OK
        → Still alive

t=260s: Third check
        Response: 500 Internal Server Error (model crashed)
        → Failure count = 1

...

t=550s: 30th consecutive failure (failureThreshold=30)
        → Kubernetes KILLS container and starts a new one
```

### Why 240s Initial Delay?

**Model Loading Time Breakdown:**
- Download weights from HF: ~60s (if not cached)
- Load weights to GPU: ~90s
- Initialize KV cache: ~30s
- Warmup inference: ~30s
- **Total: ~210s**

Setting `initialDelaySeconds: 240` gives a 30s buffer.

### The /health Endpoint

**vLLM provides this endpoint automatically:**

```python
# Inside vLLM's API server
@app.get("/health")
async def health():
    if model_loaded and engine_ready:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready"}
        )
```

---

## Part 7: Networking - How Requests Reach vLLM

### Network Flow Diagram

```
External Client
     ↓
LiteLLM Gateway (Service: litellm:4000)
     ↓
vLLM Service (Service: vllm-qwen-server:8000)
     ↓ [load balances across Pods]
vLLM Pod 1 (Container port 8000)
```

### Service Definition

**File:** `model-hosting/standalone-vllm-reasoning.yaml`

**Lines 111-126:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen-server
spec:
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: vllm-qwen-server
  type: ClusterIP
```

### What This Creates

**Kubernetes allocates:**
- **ClusterIP:** `10.100.45.123` (internal IP, stable)
- **DNS Name:** `vllm-qwen-server.default.svc.cluster.local`
- **Short DNS:** `vllm-qwen-server` (within same namespace)

**Service acts as load balancer:**

```
Request to 10.100.45.123:8000
  → Kubernetes selects Pod with label "app: vllm-qwen-server"
  → Forwards to Pod IP (e.g., 10.2.34.56:8000)
  → vLLM container receives request
```

### Selector Matching

**Service selector:**
```yaml
selector:
  app: vllm-qwen-server
```

**Pod labels (from Deployment):**
```yaml
template:
  metadata:
    labels:
      app: vllm-qwen-server
```

→ **Match!** Service routes traffic to these Pods.

### Testing the Service

```bash
# From any Pod in the cluster:
curl http://vllm-qwen-server:8000/health
# Response: {"status": "ready"}

curl http://vllm-qwen-server:8000/v1/models
# Response: {"object": "list", "data": [{"id": "Qwen/Qwen3-14B", ...}]}
```

---

## Part 8: Putting It All Together - Complete Request Lifecycle

### Scenario: User Asks "What is quantum entanglement?"

**Step 1: Client sends request to LiteLLM**
```bash
curl -X POST https://api.your-domain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-123456" \
  -d '{
    "model": "vllm-server-qwen3",
    "messages": [{"role": "user", "content": "What is quantum entanglement?"}]
  }'
```

**Step 2: LiteLLM routes to vLLM Service**

LiteLLM config (`model-gateway/litellm-deployment.yaml`, lines 97-100):
```yaml
- model_name: vllm-server-qwen3
  litellm_params:
    model: hosted_vllm/Qwen/Qwen3-14B
    api_base: http://vllm-qwen-server:8000/v1
```

LiteLLM sends:
```
POST http://vllm-qwen-server:8000/v1/chat/completions
```

**Step 3: Kubernetes Service load balances**
```
Service DNS (vllm-qwen-server)
  → Resolved to ClusterIP 10.100.45.123
  → Service selects Pod: vllm-qwen-server-abc123
  → Forwards to Pod IP: 10.2.34.56:8000
```

**Step 4: vLLM container processes request**

```python
# Inside vLLM process

# 1. Receive request
request = await receive_http_request()

# 2. Tokenize input
tokens = tokenizer.encode("What is quantum entanglement?")
# Output: [3923, 374, 12961, 1218, 526, 2111, 30]

# 3. Check KV cache for prefix match (PagedAttention)
# No match (new query)

# 4. Run forward pass across 4 GPUs (tensor parallelism)
# GPU 0: Computes layers 0-7
# GPU 1: Computes layers 8-15
# GPU 2: Computes layers 16-23
# GPU 3: Computes layers 24-31

# 5. Generate tokens autoregressively
output_tokens = []
for i in range(max_tokens):
    next_token = model.generate_next_token(tokens + output_tokens)
    output_tokens.append(next_token)
    if next_token == tokenizer.eos_token:
        break

# 6. Decode to text
response_text = tokenizer.decode(output_tokens)

# 7. Return response
return JSONResponse({
    "choices": [{
        "message": {"role": "assistant", "content": response_text}
    }]
})
```

**Step 5: Response flows back**
```
vLLM Pod → Service → LiteLLM → Client
```

**Timing:**
- Network latency: ~5ms
- Tokenization: ~10ms
- Inference: ~500ms (100 tokens × 5ms/token)
- Total: ~515ms

---

## Part 9: Scaling and Cost Optimization

### Scaling Up (More Replicas)

**Current:** 1 Pod with 4 GPUs

**To scale to 2 Pods:**

Edit `standalone-vllm-reasoning.yaml`, line 24:
```yaml
replicas: 2
```

**What happens:**
- Karpenter provisions **second g5.12xlarge** instance
- Second Pod starts
- Service load-balances across 2 Pods
- **Cost:** $5.67/hr × 2 = $11.34/hr

### Autoscaling Based on Load

**Add HorizontalPodAutoscaler:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: vllm-qwen-server
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
```

**Behavior:**
- If average queue depth > 10 requests: scale up
- If queue depth < 10: scale down
- Autoscales between 1-10 Pods

### Cost Optimization with Karpenter

**NodePool consolidation (from `gpu-nodepool.yaml`, lines 11-13):**
```yaml
disruption:
  consolidationPolicy: WhenEmptyOrUnderutilized
  consolidateAfter: 30m
```

**What this does:**
- Monitors GPU node utilization
- If node is idle for 30 minutes → terminates the instance
- Saves cost by not paying for idle GPUs

**Example:**
- 8 AM - 5 PM: High load (10 Pods, 10 nodes)
- 5 PM - 8 AM: Low load (1 Pod, 1 node)
- **Savings:** 9 nodes × 13 hours × $5.67 = $664/day

---

## Part 10: Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Pod Stuck in Pending

**Symptom:**
```bash
kubectl get pods
NAME                           STATUS    AGE
vllm-qwen-server-abc123        Pending   5m
```

**Check events:**
```bash
kubectl describe pod vllm-qwen-server-abc123
# Events:
#   Warning  FailedScheduling  pod has unbound immediate PersistentVolumeClaims
```

**Root cause:** PVC not bound (storage class doesn't exist)

**Solution:**
```bash
# Apply GP3 storage class
kubectl apply -f base_eks_setup/gp3.yaml
```

#### Issue 2: CrashLoopBackOff

**Symptom:**
```bash
kubectl get pods
NAME                           STATUS             RESTARTS   AGE
vllm-qwen-server-abc123        CrashLoopBackOff   5          10m
```

**Check logs:**
```bash
kubectl logs vllm-qwen-server-abc123
# Error: CUDA out of memory. Tried to allocate 30.50 GiB
```

**Root cause:** Model too large for GPU memory

**Solution:** Reduce `max-model-len` or `max-num-batched-tokens`

```yaml
args: [
  "vllm serve Qwen/Qwen3-14B
   --max-model-len 16384  # ← Reduced from 32768
   --max-num-batched-tokens 16384"
]
```

#### Issue 3: Slow Inference

**Symptom:** 500ms+ per token latency

**Check GPU utilization:**
```bash
kubectl exec -it vllm-qwen-server-abc123 -- nvidia-smi
# GPU Util: 45%
```

**Root cause:** Low batch size

**Solution:** Increase `max-num-seqs` to batch more requests

```yaml
args: [
  "vllm serve Qwen/Qwen3-14B
   --max-num-seqs 16  # ← Increased from 8
]
```

---

## Summary: Key Files and Their Roles

| File | Purpose | Key Lines |
|------|---------|-----------|
| `model-hosting/standalone-vllm-reasoning.yaml` | Main vLLM deployment | 58-60 (vLLM command) |
| `base_eks_setup/karpenter_nodepool/gpu-nodepool.yaml` | GPU autoscaling | 26-48 (instance requirements) |
| `base_eks_setup/gp3.yaml` | Storage class definition | - |
| `model-gateway/litellm-deployment.yaml` | API gateway config | 97-100 (model routing) |

---

## Next Steps

1. **Deploy:** `kubectl apply -f model-hosting/standalone-vllm-reasoning.yaml`
2. **Monitor:** `kubectl logs -f deployment/vllm-qwen-server`
3. **Test:** `curl http://vllm-qwen-server:8000/health`
4. **Scale:** Edit replicas or add HPA
5. **Optimize:** Tune vLLM parameters based on metrics

This guide has walked through every critical concept from Kubernetes resources to vLLM internals, always tying back to specific lines of code in the repository.
