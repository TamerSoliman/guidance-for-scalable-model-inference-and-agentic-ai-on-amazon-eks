# EKS Inference and Agentic AI - Complete Data Flow Guide

## Purpose
Trace a user query from the external internet through all system layers to the final LLM-generated response. This guide is for AI researchers who need to understand production ML system architecture.

---

## The Complete Architecture Stack

```
┌────────────────────────────────────────────────────────────────┐
│                      EXTERNAL INTERNET                          │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│           AWS Application Load Balancer (ALB)                   │
│  - SSL/TLS Termination                                          │
│  - DDoS Protection                                              │
│  - Health Checks                                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                    EKS CLUSTER (Kubernetes)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Ingress Controller (AWS Load Balancer Controller)       │  │
│  │  - Route /query → Agentic App Service                    │  │
│  │  - Route /v1/* → LiteLLM Service                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Agentic Application Pod (Strands SDK)                   │  │
│  │  - Supervisor Agent (orchestrator)                       │  │
│  │  - Knowledge Agent (RAG)                                 │  │
│  │  - MCP Agent (tool execution)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ↓                  ↓                  ↓               │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │ LiteLLM     │  │ OpenSearch       │  │ Tavily MCP      │   │
│  │ Gateway     │  │ (Vector DB)      │  │ Server          │   │
│  │ Service     │  │ - k-NN Search    │  │ - Web Search    │   │
│  └─────────────┘  └──────────────────┘  └─────────────────┘   │
│           ↓                                      ↓               │
│  ┌─────────────────────┐               ┌────────────────────┐  │
│  │ vLLM Reasoning Pod  │               │ External APIs      │  │
│  │ (Qwen3-14B)         │               │ (Tavily.com)       │  │
│  │ - 4x NVIDIA GPUs    │               └────────────────────┘  │
│  │ - Tensor Parallel   │                                        │
│  └─────────────────────┘                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## Scenario: User Asks "What was the purpose of the study on encainide and flecainide?"

We'll trace this query through **every network hop, every function call, and every decision point**.

---

## Phase 1: Request Entry (Internet → EKS)

### Step 1.1: Client Sends HTTP Request

**Location:** User's machine

**Request:**
```http
POST https://k8s-default-strandsd-xyz123.us-east-1.elb.amazonaws.com/query HTTP/1.1
Host: k8s-default-strandsd-xyz123.us-east-1.elb.amazonaws.com
Content-Type: application/json

{
  "question": "What was the purpose of the study on encainide and flecainide?",
  "top_k": 3
}
```

**Network Path:**
- Client → AWS Edge Location (CloudFront CDN, if enabled)
- Edge → AWS Region (us-east-1)
- Region → Availability Zone

**Latency:** ~50ms (cross-country) or ~5ms (same region)

---

### Step 1.2: AWS Application Load Balancer (ALB)

**AWS Resource:** Created by Kubernetes Ingress

**File Reference:** `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/main-app-deployment.yaml` (lines 177-194)

```yaml
kind: Ingress
metadata:
  name: strandsdk-rag-ingress-alb
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
```

**What ALB Does:**

1. **SSL Termination** (if HTTPS):
   ```
   HTTPS Request (encrypted)
   → ALB decrypts with SSL certificate
   → Forwards as HTTP to backend (internal network is trusted)
   ```

2. **Health Check**:
   ```
   Every 30 seconds: GET http://pod-ip:8000/health
   If 3 consecutive failures: Remove Pod from target group
   ```

3. **Load Balancing**:
   ```
   If multiple Pods exist:
   → Round-robin across healthy Pods
   → Sticky sessions disabled (stateless API)
   ```

4. **Timeout Configuration**:
   ```yaml
   alb.ingress.kubernetes.io/load-balancer-attributes: idle_timeout.timeout_seconds=900
   ```
   **Why 900s (15 minutes)?** Agentic workflows can take minutes (RAG + web search + LLM generation).

**Output:** Forwards to Pod IP (e.g., 10.2.45.67:8000)

---

## Phase 2: Agentic Application Processing

### Step 2.1: FastAPI Server Receives Request

**File:** `agentic-apps/strandsdk_agentic_rag_opensearch/src/server.py`

**Code:**
```python
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Endpoint: POST /query
    Input: {"question": str, "top_k": int}
    Output: {"response": str, "sources": [...]}
    """

    # Create a fresh supervisor agent (no conversation history)
    agent = create_fresh_supervisor_agent()

    # Invoke agent with user query
    response = agent(request.question)

    return {"response": response, "sources": [...]}
```

**What Happens:**
1. FastAPI deserializes JSON to `QueryRequest` object
2. Creates a new supervisor agent instance
3. Passes query to agent
4. Agent enters decision loop...

---

### Step 2.2: Supervisor Agent Decision Loop

**File:** `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/supervisor_agent.py`

**Agent System Prompt (lines 554-587):**
```
You are a RAG system with web search capabilities.

WORKFLOW:
1. ALWAYS start with check_knowledge_status()
2. search_knowledge_base(query="terms")
3. Check relevance_score: if < 0.3 use web_search
4. If relevance_score >= 0.3: use RAG results
```

**Decision Tree:**

```
Agent receives query
    ↓
Tool Call 1: check_knowledge_status()
    → Returns: {"status": "ready", "document_count": 150}
    ↓
Tool Call 2: search_knowledge_base(query="encainide flecainide study purpose")
    → Executes vector search in OpenSearch
    → Returns: {
        "results": [...],
        "relevance_score": 0.85,  ← HIGH relevance
        "formatted_for_evaluation": "Score: 0.95\nContent: ..."
      }
    ↓
LLM Decision: relevance_score (0.85) >= 0.3
    → Use RAG results
    → Skip web search
    ↓
Tool Call 3: check_chunks_relevance(results=..., question=...)
    → Calls AWS Bedrock for evaluation
    → Returns: {"chunk_relevance_score": "yes", "chunk_relevance_value": 0.88}
    ↓
LLM generates final answer using retrieved chunks
```

---

### Step 2.3: RAG Workflow - Vector Search

**Tool Function:** `search_knowledge_base()`

**File:** `supervisor_agent.py` (lines 411-494)

**Code Execution:**

```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    # Initialize retriever (connects to OpenSearch)
    retriever = EmbeddingRetriever()

    # Execute search
    results = retriever.search(query, top_k=top_k)

    # Calculate relevance score
    relevance_score = calculate_relevance_score(results, query)

    return JSON.dumps({
        "results": results,
        "relevance_score": relevance_score
    })
```

**What `retriever.search()` Does:**

**File:** `agentic-apps/strandsdk_agentic_rag_opensearch/src/tools/embedding_retriever.py`

```python
def search(self, query: str, top_k: int = 5) -> List[Dict]:
    # Step 1: Generate embedding for query
    query_embedding = self._generate_embedding(query)
    # → Calls Ray Serve llamacpp endpoint
    # → POST http://ray-service-llamacpp-serve-svc:8000/v1/embeddings
    # → Returns: [0.023, -0.145, 0.678, ...] (384-dim vector)

    # Step 2: Search OpenSearch k-NN index
    search_results = self.vector_store.similarity_search(
        query_embedding,
        k=top_k
    )
    # → Sends vector to OpenSearch
    # → OpenSearch runs k-NN algorithm
    # → Returns top 3 most similar documents

    return search_results
```

---

### Step 2.4: Embedding Generation (Ray Serve + llamacpp)

**Network Call:** Agent → Ray Service

**Request:**
```http
POST http://ray-service-llamacpp-serve-svc:8000/v1/embeddings
Content-Type: application/json

{
  "input": "encainide flecainide study purpose",
  "model": "snowflake-arctic-embed-s"
}
```

**Ray Service Processing:**

**File:** `model-hosting/ray-server/llamacpp.py` (lines 63-138)

**Code:**
```python
@app.post("/v1/embeddings")
async def create_embeddings(self, request: Request):
    # Extract input text
    input_text = body.get("input", "")

    # Generate embedding using llama-cpp-python
    embedding = self.llm.create_embedding(input_text)

    # Extract the embedding vector
    embedding_vector = embedding["data"][0]["embedding"]

    # Return OpenAI-compatible response
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": embedding_vector  # [0.023, -0.145, ...]
        }],
        "model": "snowflake-arctic-embed-s"
    })
```

**Where This Runs:**

- **Pod:** Ray worker pod on ARM64 Graviton instance
- **CPU:** 30 cores allocated
- **Memory:** 55GB
- **Cost:** ~$0.10/hour (m8g.8xlarge Graviton)

**Response:**
```json
{
  "object": "list",
  "data": [{
    "embedding": [0.023, -0.145, 0.678, ..., -0.234]  // 384 dimensions
  }]
}
```

**Latency:** ~50ms

---

### Step 2.5: Vector Search in OpenSearch

**Network Call:** Agent → OpenSearch

**OpenSearch Cluster:** Managed AWS OpenSearch Service

**Deployment File:** `agentic-apps/strandsdk_agentic_rag_opensearch/opensearch-cluster-simple.yaml`

**Request:**
```http
POST https://search-medical-docs-xyz.us-east-1.es.amazonaws.com/knowledge-embeddings/_search
Authorization: AWS4-HMAC-SHA256 ...
Content-Type: application/json

{
  "size": 3,
  "query": {
    "knn": {
      "embedding_vector": {
        "vector": [0.023, -0.145, 0.678, ..., -0.234],
        "k": 3
      }
    }
  }
}
```

**What OpenSearch Does:**

1. **k-NN Index Lookup:**
   ```
   Index: knowledge-embeddings
   Documents: 150 medical research papers
   Vector dimension: 384
   Algorithm: HNSW (Hierarchical Navigable Small World)
   ```

2. **Similarity Calculation:**
   ```python
   for doc in index:
       score = cosine_similarity(query_vector, doc.embedding_vector)
       # score = dot(query, doc) / (||query|| * ||doc||)

   top_3 = sorted(scores)[:3]
   ```

3. **Return Top Results:**
   ```json
   {
     "hits": {
       "hits": [
         {
           "_score": 0.95,
           "_source": {
             "content": "The purpose of the CAST study was to evaluate...",
             "metadata": {"source": "CAST_Study_1989.pdf"}
           }
         },
         {
           "_score": 0.87,
           "_source": {
             "content": "Encainide and flecainide are Class IC antiarrhythmics...",
             "metadata": {"source": "Antiarrhythmics_Review.pdf"}
           }
         },
         {
           "_score": 0.82,
           "_source": {
             "content": "The study enrolled 1,498 patients with...",
             "metadata": {"source": "CAST_Methods.pdf"}
           }
         }
       ]
     }
   }
   ```

**Latency:** ~20ms

**Cost:** ~$0.15/hour (single t3.medium.search node)

---

### Step 2.6: Relevance Evaluation (RAGAs + AWS Bedrock)

**Tool Function:** `check_chunks_relevance()`

**File:** `supervisor_agent.py` (lines 266-408)

**Why This Step?**
Vector similarity doesn't guarantee semantic relevance. Example:
- Query: "weather in Seattle"
- High similarity match: "weather in Portland" (wrong city!)

**Process:**

```python
@tool
def check_chunks_relevance(results: str, question: str):
    # Step 1: Parse retrieved chunks
    docs = extract_chunks_from_results(results)
    # → ["The purpose of CAST study...", "Encainide and flecainide...", ...]

    # Step 2: Generate answer using chunks (via Bedrock)
    answer = generate_answer_from_context(question, docs)

    # Step 3: Evaluate if chunks were useful (RAGAs)
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=docs
    )

    scorer = LLMContextPrecisionWithoutReference(llm=bedrock_claude)
    score = scorer.score(sample)
    # → Bedrock Claude evaluates: "Were these chunks helpful for answering?"

    return {
        "chunk_relevance_score": "yes" if score > 0.5 else "no",
        "chunk_relevance_value": score
    }
```

**Network Call to AWS Bedrock:**

```http
POST https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-3-7-sonnet-20250219-v1:0/invoke
Authorization: AWS4-HMAC-SHA256 ...
Content-Type: application/json

{
  "messages": [{
    "role": "user",
    "content": "Evaluate if these chunks are relevant to the question..."
  }],
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "content": [{
    "text": "The retrieved chunks directly address the query. Score: 0.88"
  }]
}
```

**Latency:** ~800ms (Bedrock inference)

**Cost:** ~$0.003 per request

---

### Step 2.7: Final Answer Generation (vLLM)

**Now that we have relevant chunks, generate the final answer.**

**Agent makes final LLM call with context:**

**Network Call:** Agent → LiteLLM → vLLM

**Request to LiteLLM:**

```http
POST http://litellm:4000/v1/chat/completions
Authorization: Bearer sk-123456
Content-Type: application/json

{
  "model": "vllm-server-qwen3",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Answer using the provided context."
    },
    {
      "role": "user",
      "content": "Context:\n1. The purpose of the CAST study was to evaluate...\n2. Encainide and flecainide are Class IC...\n\nQuestion: What was the purpose of the study on encainide and flecainide?"
    }
  ],
  "max_tokens": 500
}
```

**LiteLLM Routes to vLLM:**

**File:** `model-gateway/litellm-deployment.yaml` (lines 97-100)

```yaml
model_list:
- model_name: vllm-server-qwen3
  litellm_params:
    model: hosted_vllm/Qwen/Qwen3-14B
    api_base: http://vllm-qwen-server:8000/v1
```

**LiteLLM forwards:**
```http
POST http://vllm-qwen-server:8000/v1/chat/completions
```

---

### Step 2.8: vLLM Inference (GPU)

**Pod:** vllm-qwen-server-xyz123 (on g5.12xlarge node)

**File:** `model-hosting/standalone-vllm-reasoning.yaml`

**Processing Steps:**

```python
# Inside vLLM container

# 1. Tokenize input
tokens = tokenizer.encode(prompt)
# Output: [12, 453, 2987, ..., 7821]  (327 tokens)

# 2. Load into GPU memory
input_tensor = torch.tensor(tokens).cuda()

# 3. Run forward pass (tensor parallel across 4 GPUs)
# GPU 0: Processes layers 0-7
# GPU 1: Processes layers 8-15
# GPU 2: Processes layers 16-23
# GPU 3: Processes layers 24-31

# 4. Generate tokens autoregressively
output_tokens = []
for step in range(max_tokens):
    # Predict next token
    logits = model(input_tensor)
    next_token = torch.argmax(logits[-1])

    output_tokens.append(next_token)

    # Check for end of sequence
    if next_token == tokenizer.eos_token:
        break

    # Add to input for next iteration
    input_tensor = torch.cat([input_tensor, next_token])

# 5. Decode tokens to text
response_text = tokenizer.decode(output_tokens)
# Output: "The purpose of the CAST (Cardiac Arrhythmia Suppression Trial)..."
```

**Performance:**
- **Throughput:** ~100 tokens/second (with tensor parallelism)
- **Generated tokens:** ~150 tokens
- **Inference time:** ~1500ms
- **GPU utilization:** 85%

**Response:**
```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The purpose of the CAST study was to evaluate whether antiarrhythmic drugs (encainide and flecainide) could reduce mortality in patients with ventricular arrhythmias following myocardial infarction..."
    }
  }],
  "usage": {
    "prompt_tokens": 327,
    "completion_tokens": 150,
    "total_tokens": 477
  }
}
```

---

## Phase 3: Response Journey Back to User

### Step 3.1: vLLM → LiteLLM

**Path:** vLLM Pod (port 8000) → LiteLLM Pod (port 4000)

**LiteLLM Processing:**

1. **Caching (Redis):**
   ```python
   cache_key = hash(model + messages)
   redis.set(cache_key, response, ex=600)  # Cache for 10 minutes
   ```

2. **Logging to Langfuse:**
   ```python
   langfuse.log_completion(
       model="vllm-server-qwen3",
       prompt_tokens=327,
       completion_tokens=150,
       latency_ms=1500,
       cost_usd=0.0008  # Estimated
   )
   ```

3. **Forward response to agent**

---

### Step 3.2: Agent → FastAPI → User

**Supervisor Agent:**
```python
# Agent receives LLM response
final_answer = llm_response["choices"][0]["message"]["content"]

# Format with sources
formatted_response = {
    "response": final_answer,
    "sources": [
        {"title": "CAST_Study_1989.pdf", "score": 0.95},
        {"title": "Antiarrhythmics_Review.pdf", "score": 0.87}
    ],
    "evaluation": {
        "rag_relevance": 0.85,
        "chunk_relevance": 0.88,
        "method": "RAG (web search skipped)"
    }
}

return formatted_response
```

**FastAPI Server:**
```python
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = agent(request.question)
    return JSONResponse(content=result)
```

**HTTP Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "response": "The purpose of the CAST study was to evaluate whether antiarrhythmic drugs...",
  "sources": [
    {"title": "CAST_Study_1989.pdf", "score": 0.95},
    {"title": "Antiarrhythmics_Review.pdf", "score": 0.87}
  ],
  "evaluation": {
    "rag_relevance": 0.85,
    "chunk_relevance": 0.88,
    "method": "RAG"
  }
}
```

---

### Step 3.3: ALB → Internet → User

**Path:**
```
FastAPI (Pod IP 10.2.45.67:8000)
  → ALB Target Group
  → ALB Public IP
  → Internet
  → User's Browser
```

**Total Latency Breakdown:**

| Step | Component | Time | Notes |
|------|-----------|------|-------|
| 1 | Client → ALB | 50ms | Network latency |
| 2 | ALB → Pod | 5ms | Internal network |
| 3 | check_knowledge_status() | 10ms | Quick DB query |
| 4 | Embedding generation | 50ms | Ray Serve (CPU) |
| 5 | OpenSearch vector search | 20ms | k-NN lookup |
| 6 | check_chunks_relevance() | 800ms | Bedrock evaluation |
| 7 | vLLM inference | 1500ms | GPU generation |
| 8 | Response back to user | 50ms | Network latency |
| **Total** | **End-to-end** | **~2.5s** | Full pipeline |

---

## Alternative Flow: Web Search Triggered

**Scenario:** What if relevance_score < 0.3?

**Query:** "What's the weather in Seattle today?"

**Modified Flow:**

```
Step 1-3: Same (check_knowledge_status, search_knowledge_base)

Step 4: Relevance check
  → relevance_score = 0.12 (LOW - knowledge base has no weather data)

Step 5: Agent decision
  → LLM decides: "Use web_search tool"

Step 6: MCP Tool Call
  → Agent calls: web_search(query="weather Seattle today", max_results=5)

Step 7: Tavily MCP Server
  Request:
    POST http://tavily-mcp-server:8001/tools/web_search
  Response:
    {
      "results": [
        {"title": "Seattle Weather", "snippet": "Currently 55°F, rainy..."}
      ]
    }

Step 8: Tavily API Call (external)
  MCP Server → https://api.tavily.com/search
  Response: Live web search results

Step 9: Agent generates answer
  Context: Web search results (not RAG)
  LLM call to vLLM with web data

Step 10: Response to user
  "The current weather in Seattle is 55°F and rainy..."
```

**Latency:** ~3.5s (extra 1s for web search API call)

---

## Observability: How We Track This Flow

### Langfuse Tracing

**Every step creates a span:**

```
Trace: user-query-xyz123
├─ Span: supervisor-agent (2500ms)
│  ├─ Span: check_knowledge_status (10ms)
│  ├─ Span: search_knowledge_base (70ms)
│  │  ├─ Span: embedding-generation (50ms)
│  │  └─ Span: opensearch-knn-search (20ms)
│  ├─ Span: check_chunks_relevance (800ms)
│  │  └─ Span: bedrock-claude-call (750ms)
│  └─ Span: vllm-generation (1500ms)
```

**View in Langfuse Dashboard:**
- Total cost: $0.004
- Total tokens: 477
- Bottleneck: vLLM generation (60% of time)

---

## Cost Breakdown (per request)

| Component | Cost | Notes |
|-----------|------|-------|
| Ray Serve (embedding) | $0.0001 | 50ms on Graviton ($0.10/hr) |
| OpenSearch (vector search) | $0.0001 | 20ms on t3.medium ($0.15/hr) |
| Bedrock Claude (evaluation) | $0.003 | ~1000 tokens |
| vLLM (generation) | $0.0008 | 1500ms on g5.12xlarge ($5.67/hr) |
| Network & overhead | $0.0001 | ALB, EKS control plane |
| **Total per request** | **$0.0041** | ~$4 per 1000 requests |

**Optimization:**
- Cache responses (10 min TTL) → Reduce duplicate requests by 70%
- Effective cost: **$1.20 per 1000 unique queries**

---

## Summary: Key Insights for AI Researchers

1. **Multi-tier Architecture:** Not just "prompt → LLM → response"
   - Vector search pre-filters knowledge
   - Relevance evaluation guards against hallucination
   - Tool calling enables dynamic behavior

2. **Latency Budget:**
   - 60% spent in LLM generation (unavoidable)
   - 30% in evaluation/search (can be optimized)
   - 10% in network (negligible)

3. **Cost Drivers:**
   - GPU inference: 20% of cost
   - Evaluation LLM (Bedrock): 75% of cost
   - Infrastructure: 5% of cost

4. **Reliability:**
   - Load balancer health checks
   - Automatic Pod restarts
   - Multi-AZ deployment (not shown, but recommended)

5. **Observability:**
   - Every step is traced
   - Costs are tracked
   - Bottlenecks are identified

This architecture provides **production-grade RAG** with **intelligent fallback to web search**, **quality evaluation**, and **full observability** - far beyond a simple "RAG pipeline."
