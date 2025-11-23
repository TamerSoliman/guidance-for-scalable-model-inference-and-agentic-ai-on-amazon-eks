# EKS-Based LLM Inference and Agentic AI - Deep Dive Tutorial

## Overview

This tutorial provides a comprehensive, code-centric deep dive into the EKS-based inference and agentic AI architecture in this repository. It is specifically designed for **generative AI researchers with limited background in cloud infrastructure, MLOps, DevOps, or Kubernetes**.

## What You'll Learn

1. **Infrastructure Fundamentals**: How Kubernetes, EKS, and Karpenter work together to provision GPU/CPU compute automatically
2. **Model Serving Architecture**: How vLLM and Ray Serve deploy production LLM inference at scale
3. **Agentic Workflows**: How Strands SDK orchestrates multi-agent systems with RAG, tool calling, and MCP
4. **Complete Data Flows**: Trace a user query from the internet through every system layer to the LLM response
5. **Tool Integration**: Understand the Model Context Protocol (MCP) and how LLMs invoke external functions

## Tutorial Structure

### Phase 1: Discovery and Architecture Mapping

**📄 [00_Discovery_Plan.md](00_Discovery_Plan.md)**
- Catalog of the top 50 most critical files in the repository
- Organized by architectural layer (infrastructure, model hosting, agentic apps, observability)
- Each file annotated with its role and key concepts
- Comprehensive glossary for AI researchers

**Key Takeaway:** Understand the repository organization and identify which files implement which concepts.

---

### Phase 2: Annotated Configuration Files

**📄 [01_ANNOTATED_GPU_NodePool.yaml](01_ANNOTATED_GPU_NodePool.yaml)**
- **Concept**: Dynamic GPU provisioning with Karpenter
- **What It Explains**:
  - How Karpenter automatically provisions GPU instances (g5.12xlarge)
  - Node selection criteria (instance family, GPU count, architecture)
  - EBS volume configuration for model weights
  - Disruption policies for cost optimization
- **Line-by-Line Annotations**: Every YAML field explained with "What, How, Why"

**Analogies for Researchers:**
- NodePool = "Procurement policy" for compute
- Karpenter = Automated lab equipment ordering system

---

**📄 [02_ANNOTATED_vLLM_Deployment.yaml](02_ANNOTATED_vLLM_Deployment.yaml)**
- **Concept**: Containerized high-throughput LLM inference
- **What It Explains**:
  - vLLM command parameters (tensor-parallel-size, dtype, max-model-len)
  - GPU resource allocation (nvidia.com/gpu: 4)
  - Model caching with PersistentVolumes
  - Health checks and readiness probes
  - Container startup process
- **Performance Characteristics**: Token latency, throughput, cold start times

**Key Sections:**
- Model loading timeline (download → GPU loading → warmup)
- Shared memory configuration for tensor parallelism
- Environment variable injection (Hugging Face tokens)
- Network flow (Service → Pod → Container)

---

**📄 [03_ANNOTATED_Supervisor_Agent.py](03_ANNOTATED_Supervisor_Agent.py)**
- **Concept**: Multi-agent orchestration with Strands SDK
- **What It Explains**:
  - Agent tool definitions (@tool decorator)
  - RAGAs relevance evaluation
  - MCP (Model Context Protocol) client integration
  - Async cleanup patterns for production
  - Decision logic (RAG vs. web search)
- **Framework Deep Dive**:
  - Strands SDK architecture
  - OpenTelemetry tracing integration
  - Langfuse observability

**Key Functions:**
- `search_knowledge_base()` - Vector similarity search
- `check_chunks_relevance()` - RAG quality evaluation
- `calculate_relevance_score()` - Content validation

---

### Phase 3: Concept-to-Code Reference Guides

**📄 [04_vLLM_on_EKS_Concept_to_Code.md](04_vLLM_on_EKS_Concept_to_Code.md)**
- **Purpose**: Map high-level concepts to specific code/config lines
- **Sections**:
  1. **Component Mapping**: YAML resources → What they do
  2. **Model Loading Flow**: Download → Cache → GPU memory
  3. **GPU Resource Allocation**: How 4 GPUs are requested and provisioned
  4. **Environment Variables**: Configuration injection patterns
  5. **Health Checks**: Liveness vs. Readiness probes explained
  6. **Networking**: Service → Pod → Container routing
  7. **Request Lifecycle**: Complete end-to-end flow
  8. **Scaling and Cost Optimization**: HPA, Karpenter consolidation
  9. **Troubleshooting Guide**: Common issues and solutions

**Example Walkthrough:**
- How `--tensor-parallel-size 4` maps to 4 GPU allocation
- Why `initialDelaySeconds: 240` in health checks (model loading time)
- Service ClusterIP DNS resolution

---

**📄 [05_EKS_Inference_Data_Flow_Guide.md](05_EKS_Inference_Data_Flow_Guide.md)**
- **Purpose**: Trace a user query through every network hop and function call
- **Scenario**: "What was the purpose of the study on encainide and flecainide?"
- **Complete Flow**:
  1. **Internet → ALB**: SSL termination, health checks, load balancing
  2. **ALB → EKS Pod**: Ingress routing, timeout configuration
  3. **FastAPI Server**: Request deserialization
  4. **Supervisor Agent**: Tool calling decision tree
  5. **Vector Search**: Embedding generation (Ray Serve) + OpenSearch k-NN
  6. **RAG Evaluation**: RAGAs with AWS Bedrock
  7. **vLLM Inference**: Tokenization → GPU inference → Response
  8. **Response Flow**: Pod → Service → ALB → Internet

**Performance Breakdown:**
- Latency budget (2.5s total)
- Cost per request ($0.004)
- Bottleneck identification (vLLM = 60% of time)

**Alternative Flow:** Web search triggered (when RAG relevance < 0.3)

---

**📄 [06_Agent_Tool_Calling_API_Reference.md](06_Agent_Tool_Calling_API_Reference.md)**
- **Purpose**: Complete reference for tool-calling architecture
- **Sections**:
  1. **Tool Calling Fundamentals**: How LLMs invoke functions
  2. **Strands SDK @tool Decorator**: Function → Tool definition conversion
  3. **Local Tools Inventory**:
     - `check_knowledge_status()`
     - `search_knowledge_base(query, top_k)`
     - `check_chunks_relevance(results, question)`
     - `file_read(path)`, `file_write(content, filename)`
  4. **MCP (Model Context Protocol)**:
     - What is MCP and why use it?
     - MCP server architecture
     - `web_search()`, `news_search()`, `health_check()`
  5. **Complete Tool Call Example**: Multi-step agent workflow
  6. **Extending the System**: How to add new tools
  7. **Best Practices**: Descriptions, type hints, error handling

**Tool Signatures:**
- Parameter types, defaults, descriptions
- Return value formats (JSON structures)
- LLM usage examples for each tool

---

## How to Use This Tutorial

### For Complete Beginners
1. Start with **00_Discovery_Plan.md** - understand the repository structure
2. Read the **glossary** at the end of the Discovery Plan
3. Read **05_EKS_Inference_Data_Flow_Guide.md** - see the big picture
4. Deep dive into **01_ANNOTATED_GPU_NodePool.yaml** - understand infrastructure
5. Deep dive into **02_ANNOTATED_vLLM_Deployment.yaml** - understand model serving

### For Infrastructure-Focused Learners
1. **01_ANNOTATED_GPU_NodePool.yaml** - Karpenter autoscaling
2. **02_ANNOTATED_vLLM_Deployment.yaml** - Kubernetes deployments
3. **04_vLLM_on_EKS_Concept_to_Code.md** - Detailed explanations
4. **05_EKS_Inference_Data_Flow_Guide.md** - Network flows

### For ML/AI-Focused Learners
1. **03_ANNOTATED_Supervisor_Agent.py** - Agent orchestration
2. **06_Agent_Tool_Calling_API_Reference.md** - Tool integration
3. **05_EKS_Inference_Data_Flow_Guide.md** (Phase 2) - RAG workflow
4. **04_vLLM_on_EKS_Concept_to_Code.md** (Part 4) - GPU inference

### For Hands-On Implementation
Follow this order:
1. **00_Discovery_Plan.md** - Identify key files
2. **04_vLLM_on_EKS_Concept_to_Code.md** - Implementation guide
3. **Annotated YAMLs** - Understand configurations
4. **05_Data_Flow_Guide** - Test your deployment

---

## Key Concepts Explained

### Infrastructure Layer
- **Kubernetes**: Container orchestration system
- **EKS**: Managed Kubernetes on AWS
- **Karpenter**: Intelligent autoscaler (provisions EC2 instances based on workload needs)
- **NodePool**: Template for instance provisioning
- **Pod**: Smallest deployable unit (1+ containers)
- **Service**: Stable network endpoint for Pods
- **Ingress**: HTTP routing rules (external → internal)

### Model Serving Layer
- **vLLM**: High-throughput LLM inference engine
- **Ray Serve**: Distributed model serving framework
- **Tensor Parallelism**: Splitting model across multiple GPUs
- **PagedAttention**: vLLM's memory-efficient KV cache management
- **PersistentVolume**: Durable storage for model weights

### Agentic Layer
- **Strands SDK**: Agent framework with built-in tracing
- **Tool Calling**: LLM invoking external functions
- **MCP (Model Context Protocol)**: Standard for tool integration
- **RAG**: Retrieval Augmented Generation
- **RAGAs**: Framework for evaluating RAG quality
- **Supervisor Agent**: Orchestrator that coordinates other agents

### Data Layer
- **OpenSearch**: Managed vector database with k-NN search
- **k-NN**: k-Nearest Neighbors (similarity search algorithm)
- **Embedding**: Vector representation of text
- **Cosine Similarity**: Measure of vector similarity

---

## Architecture Highlights

### Multi-Tier Design
```
Internet
  ↓
AWS ALB (Load Balancer)
  ↓
EKS Cluster
  ├─ Agentic App (Supervisor Agent)
  ├─ LiteLLM Gateway (API routing)
  ├─ vLLM Inference (GPU)
  ├─ Ray Serve (CPU embeddings)
  ├─ OpenSearch (Vector DB)
  └─ Tavily MCP Server (Web search)
```

### Key Benefits
1. **Auto-Scaling**: Karpenter provisions GPU nodes on demand
2. **Cost Optimization**: Nodes deleted after 30min idle
3. **High Availability**: Multi-Pod deployments with Service load balancing
4. **Observability**: OpenTelemetry + Langfuse tracing
5. **Extensibility**: MCP allows adding new tools without code changes

---

## Performance Characteristics

### vLLM Inference (Qwen3-14B on g5.12xlarge)
- **Cold Start**: ~4 minutes (model loading)
- **Warm Latency**: ~50ms per token
- **Throughput**: ~100 tokens/second (with batching)
- **Concurrent Requests**: Up to 8 (configurable)
- **Cost**: ~$5.67/hour

### Agentic RAG Workflow
- **End-to-End Latency**: ~2.5 seconds
- **Embedding**: 50ms (Ray Serve on Graviton)
- **Vector Search**: 20ms (OpenSearch k-NN)
- **RAG Evaluation**: 800ms (AWS Bedrock)
- **LLM Generation**: 1500ms (vLLM)
- **Cost per Request**: ~$0.004

---

## Common Pitfalls (and Solutions)

### 1. Pod Stuck in Pending
**Cause:** No GPU nodes available
**Solution:** Check Karpenter NodePool limits, verify instance quota

### 2. vLLM OOM (Out of Memory)
**Cause:** Model too large for GPU
**Solution:** Reduce `max-model-len` or `max-num-batched-tokens`

### 3. Slow Inference
**Cause:** Low GPU utilization
**Solution:** Increase `max-num-seqs` for better batching

### 4. Health Check Failures
**Cause:** `initialDelaySeconds` too short
**Solution:** Increase to >240s for large models

---

## Next Steps

### To Deploy This System
1. **Prerequisites**:
   - EKS cluster (see main README)
   - Hugging Face token
   - Tavily API key
   - AWS credentials

2. **Install Infrastructure**:
   ```bash
   make setup-base           # Karpenter, GPU operator
   make setup-models         # vLLM, Ray Serve
   make setup-gateway        # LiteLLM
   make setup-observability  # Langfuse
   ```

3. **Deploy Agentic App**:
   ```bash
   make setup-rag-strands    # Multi-agent RAG system
   ```

4. **Test**:
   ```bash
   curl -X POST http://<alb-endpoint>/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Bell palsy?", "top_k": 3}'
   ```

### To Extend the System
1. **Add New Tools**: See `06_Agent_Tool_Calling_API_Reference.md` (Part 6)
2. **Deploy New Models**: Modify `standalone-vllm-reasoning.yaml`
3. **Scale Replicas**: Edit `replicas: 1` → `replicas: 3`
4. **Add HPA**: Implement HorizontalPodAutoscaler (see guide)

---

## Questions and Further Learning

### Recommended Resources

**Kubernetes Basics:**
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)

**Model Serving:**
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ray Serve Guide](https://docs.ray.io/en/latest/serve/)

**Agentic AI:**
- [Strands SDK](https://github.com/strands-ai/strands-sdk)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [RAGAs Framework](https://docs.ragas.io/)

---

## Tutorial Feedback

This tutorial was created as part of a deep-dive analysis project. If you find gaps or areas that need more explanation for AI researchers, please open an issue or contribute improvements.

**Target Audience Reminder:** This material assumes strong AI/ML background but limited infrastructure knowledge. Concepts like "Kubernetes Pod" are explained in detail, while concepts like "vector embeddings" are assumed knowledge.

---

## Summary of Deliverables

| File | Purpose | Length | Difficulty |
|------|---------|--------|------------|
| 00_Discovery_Plan.md | Architecture overview and file catalog | Long | Beginner |
| 01_ANNOTATED_GPU_NodePool.yaml | Karpenter infrastructure | Long | Intermediate |
| 02_ANNOTATED_vLLM_Deployment.yaml | Model serving deployment | Long | Intermediate |
| 03_ANNOTATED_Supervisor_Agent.py | Agent orchestration code | Long | Advanced |
| 04_vLLM_on_EKS_Concept_to_Code.md | Concept mapping guide | Very Long | Intermediate |
| 05_EKS_Inference_Data_Flow_Guide.md | Complete data flow tracing | Very Long | Advanced |
| 06_Agent_Tool_Calling_API_Reference.md | Tool integration reference | Very Long | Advanced |

**Estimated Study Time:** 8-12 hours for complete understanding

---

## License and Attribution

This tutorial is part of the **AWS Solutions Library Guidance for Scalable Model Inference and Agentic AI on Amazon EKS** repository.

All code examples and configurations are provided under the same license as the main repository.
