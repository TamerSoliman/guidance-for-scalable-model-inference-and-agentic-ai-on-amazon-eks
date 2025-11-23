# Agent Tool-Calling and MCP API Reference

## Purpose
This document provides a comprehensive reference for the tool-calling architecture used in the multi-agent system. It covers both local tools (Python functions) and external tools (via MCP servers).

## Target Audience
Generative AI researchers who need to understand how LLMs invoke external functions and how to extend the system with new tools.

---

## Part 1: Understanding Tool Calling

### What is Tool Calling?

**Traditional LLM Interaction:**
```
User: "What's the weather in Seattle?"
LLM: "I don't have access to real-time data..." ❌
```

**Tool-Enabled LLM:**
```
User: "What's the weather in Seattle?"
LLM: *calls weather_api(location="Seattle")*
System: *executes function, returns {"temp": 55, "condition": "rainy"}*
LLM: "The weather in Seattle is 55°F and rainy." ✅
```

### How It Works (Technical Flow)

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         ↓
┌─────────────────────────────────────────┐
│  LLM receives query + tool definitions  │
│  "You have access to: weather_api()"    │
└────────┬────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  LLM decides: "I need weather_api"      │
│  Returns: {                              │
│    "tool": "weather_api",                │
│    "arguments": {"location": "Seattle"}  │
│  }                                       │
└────────┬────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  System executes: weather_api("Seattle")│
│  Returns: {"temp": 55, "rainy": true}   │
└────────┬────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  LLM receives result, generates answer  │
│  "The weather in Seattle is 55°F..."    │
└─────────────────────────────────────────┘
```

---

## Part 2: Strands SDK Tool Decorator

### Defining a Tool

**File:** `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/supervisor_agent.py`

**Example:**
```python
from strands import Agent, tool

@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        query (str): The search query - REQUIRED
        top_k (int): Number of top results to return (default: 3)

    Returns:
        str: JSON string with search results and relevance metadata
    """
    retriever = EmbeddingRetriever()
    results = retriever.search(query, top_k=top_k)

    return json.dumps({
        "results": results,
        "relevance_score": calculate_relevance_score(results, query)
    })
```

### What the @tool Decorator Does

**Behind the scenes:**

```python
# Strands SDK converts function to tool definition
tool_definition = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - REQUIRED"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}
```

**This JSON is sent to the LLM in the system prompt:**

```
You have access to the following tools:

1. search_knowledge_base(query: str, top_k: int = 3) -> str
   Search the knowledge base for relevant information.
   - query (str): The search query - REQUIRED
   - top_k (int): Number of top results to return (default: 3)
```

---

## Part 3: Local Tools Inventory

### Tool 1: check_knowledge_status()

**Purpose:** Check if the knowledge base is ready

**File:** `supervisor_agent.py` (lines 497-524)

**Function Signature:**
```python
@tool
def check_knowledge_status() -> str
```

**Parameters:** None

**Returns:**
```json
{
  "status": "ready" | "empty",
  "document_count": 150,
  "last_updated": "2025-11-23"
}
```

**LLM Usage Example:**
```
LLM: I should check if the knowledge base has documents.
LLM: *calls check_knowledge_status()*
System: {"status": "ready", "document_count": 150}
LLM: The knowledge base is ready with 150 documents.
```

**Implementation:**
```python
def check_knowledge_status() -> str:
    retriever = EmbeddingRetriever()
    count = retriever.get_document_count()

    status_data = {
        "status": "ready" if count > 0 else "empty",
        "document_count": count,
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }

    return json.dumps(status_data)
```

---

### Tool 2: search_knowledge_base(query, top_k)

**Purpose:** Perform vector similarity search in OpenSearch

**File:** `supervisor_agent.py` (lines 411-494)

**Function Signature:**
```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str
```

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Search query text |
| top_k | integer | No | 3 | Number of results to return |

**Returns:**
```json
{
  "results": [
    {
      "source": "CAST_Study_1989.pdf",
      "content": "The purpose of the study was...",
      "score": 0.95
    }
  ],
  "relevance_score": 0.85,
  "total_results": 3,
  "query": "encainide flecainide study",
  "formatted_for_evaluation": "Score: 0.95\nContent: ..."
}
```

**LLM Usage Example:**
```
User: "What is Bell's palsy?"
LLM: I need to search the knowledge base.
LLM: *calls search_knowledge_base(query="Bell's palsy", top_k=3)*
System: {"results": [...], "relevance_score": 0.92}
LLM: Based on the knowledge base, Bell's palsy is...
```

**Key Features:**
- Content validation (keyword overlap)
- Deduplication
- Relevance scoring
- Formatted output for RAGAs evaluation

---

### Tool 3: check_chunks_relevance(results, question)

**Purpose:** Evaluate if retrieved chunks are relevant using RAGAs

**File:** `supervisor_agent.py` (lines 266-408)

**Function Signature:**
```python
@tool
def check_chunks_relevance(results: str, question: str) -> dict
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| results | string | Yes | Formatted search results with "Score:" and "Content:" patterns |
| question | string | Yes | Original user question |

**Returns:**
```json
{
  "chunk_relevance_score": "yes" | "no",
  "chunk_relevance_value": 0.88,
  "evaluation_method": "ragas" | "fallback_heuristic"
}
```

**LLM Usage Example:**
```
LLM: I have search results. Let me check if they're relevant.
LLM: *calls check_chunks_relevance(
       results=formatted_results,
       question="What is Bell's palsy?"
     )*
System: {"chunk_relevance_score": "yes", "chunk_relevance_value": 0.88}
LLM: The chunks are highly relevant (score: 0.88). I'll use them.
```

**Decision Threshold:**
- `chunk_relevance_value > 0.5` → "yes" (use RAG)
- `chunk_relevance_value <= 0.5` → "no" (trigger web search)

---

### Tool 4: file_read(path)

**Purpose:** Read file contents from the filesystem

**Source:** Strands built-in tool (`strands_tools.file_read`)

**Function Signature:**
```python
@tool
def file_read(path: str) -> str
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| path | string | Yes | Absolute or relative file path |

**Returns:** String with file contents

**LLM Usage Example:**
```
User: "Read the summary from results.txt"
LLM: *calls file_read(path="output/results.txt")*
System: "Summary: The CAST study showed..."
LLM: The summary states...
```

---

### Tool 5: file_write(content, filename)

**Purpose:** Write content to a file in the output directory

**File:** `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/mcp_agent.py` (lines 23-65)

**Function Signature:**
```python
@tool
def file_write(content: str, path: str = None, filename: str = None) -> str
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| content | string | Yes | Content to write to file |
| filename | string | No | Filename (saved to output directory) |
| path | string | No | Full path (including directory) |

**Returns:**
```
"✅ File written successfully to output/summary.md (1234 bytes)"
```

**LLM Usage Example:**
```
User: "Save the summary to summary.md"
LLM: *calls file_write(
       content="# Summary\nThe study showed...",
       filename="summary.md"
     )*
System: "✅ File written successfully to output/summary.md (256 bytes)"
LLM: I've saved the summary to output/summary.md.
```

**Implementation Details:**
```python
def file_write(content: str, path: str = None, filename: str = None) -> str:
    if filename:
        # Automatically use output directory
        path = f"{config.OUTPUT_DIR}/{filename}"

    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    return f"✅ File written successfully to {path}"
```

---

## Part 4: MCP (Model Context Protocol) Tools

### What is MCP?

**MCP (Model Context Protocol)** is a standardized protocol for connecting LLMs to external tools and data sources.

**Architecture:**

```
┌─────────────────────┐         ┌─────────────────────┐
│  Supervisor Agent   │         │  Tavily MCP Server  │
│  (MCP Client)       │ <─HTTP─>│  (External Tools)   │
└─────────────────────┘         └─────────────────────┘
         |                               |
         | Tool calls                    | API calls
         v                               v
   LLM decides to                   Tavily.com API
   call web_search()                (web search)
```

### Why MCP?

**Benefits:**
1. **Standardization:** Common protocol for all tools
2. **Language-agnostic:** Server can be in any language
3. **Hot-reload:** Add/remove tools without restarting agent
4. **Security:** Tools run in separate process/container

---

### MCP Server Setup

**File:** `agentic-apps/strandsdk_agentic_rag_opensearch/src/mcp_servers/tavily_search_server.py`

**Server Initialization:**
```python
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Tavily Search Server")

@mcp.tool(description="Search the web using Tavily API")
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True
) -> dict:
    """
    Search the web for current information.

    Args:
        query: Search query
        max_results: Number of results (1-10)
        search_depth: "basic" or "advanced"
        include_answer: Include AI-generated answer

    Returns:
        {"results": [...], "answer": "..."}
    """
    # Call Tavily API
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer
    )

    return {
        "results": response.get("results", []),
        "answer": response.get("answer", "")
    }

# Start server
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
```

**Deployment:**

**Kubernetes:** `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/tavily-mcp-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tavily-mcp-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mcp-server
        image: <ecr-repo>/tavily-mcp-server:latest
        ports:
        - containerPort: 8001
        env:
        - name: TAVILY_API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: tavily-api-key
```

**Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: tavily-mcp-server
spec:
  selector:
    app: tavily-mcp-server
  ports:
  - port: 8001
    targetPort: 8001
```

---

### MCP Client Integration

**File:** `supervisor_agent.py` (lines 95-112)

**Client Setup:**
```python
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

def get_tavily_mcp_client():
    """Initialize MCP client for Tavily server"""

    # Connect to Kubernetes Service
    mcp_url = config.TAVILY_MCP_SERVICE_URL  # http://tavily-mcp-server:8001

    # Create client with streamable HTTP transport
    client = MCPClient(lambda: streamablehttp_client(mcp_url))

    return client
```

**Agent Creation with MCP Tools:**
```python
# Get MCP client
mcp_client = get_tavily_mcp_client()

# Use context manager (required by Strands SDK)
with mcp_client:
    # Get tools from MCP server
    mcp_tools = mcp_client.list_tools_sync()
    # Returns: [web_search, news_search, health_check]

    # Combine with local tools
    all_tools = [
        search_knowledge_base,
        check_chunks_relevance,
        file_read,
        file_write
    ] + mcp_tools

    # Create agent with all tools
    agent = Agent(
        model="vllm-server-qwen3",
        tools=all_tools,
        system_prompt="..."
    )
```

---

### MCP Tool 1: web_search()

**Purpose:** Real-time web search via Tavily API

**Function Signature:**
```python
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True
) -> dict
```

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| max_results | integer | No | 5 | Number of results (1-10) |
| search_depth | string | No | "basic" | "basic" or "advanced" |
| include_answer | boolean | No | true | Include AI-generated summary |

**Returns:**
```json
{
  "results": [
    {
      "title": "Seattle Weather - National Weather Service",
      "url": "https://weather.gov/seattle",
      "snippet": "Currently 55°F with light rain. Forecast: ...",
      "score": 0.98
    }
  ],
  "answer": "The current weather in Seattle is 55°F with light rain..."
}
```

**LLM Usage Example:**
```
User: "What's the weather in Seattle today?"
LLM: Knowledge base relevance is low. I'll search the web.
LLM: *calls web_search(query="Seattle weather today", max_results=3)*
System: {"results": [...], "answer": "Currently 55°F..."}
LLM: The current weather in Seattle is 55°F with light rain.
```

**Cost:** ~$0.002 per request (Tavily API)

---

### MCP Tool 2: news_search()

**Purpose:** Search recent news articles

**Function Signature:**
```python
def news_search(
    query: str,
    max_results: int = 5,
    days_back: int = 7
) -> dict
```

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | News search query |
| max_results | integer | No | 5 | Number of articles |
| days_back | integer | No | 7 | How far back to search (days) |

**Returns:**
```json
{
  "results": [
    {
      "title": "FDA Approves New Treatment",
      "url": "https://news.example.com/fda-approval",
      "published_date": "2025-11-22",
      "snippet": "The FDA has approved..."
    }
  ]
}
```

**LLM Usage Example:**
```
User: "Any recent news about FDA approvals?"
LLM: This requires recent news.
LLM: *calls news_search(query="FDA approvals", days_back=30)*
System: {"results": [...]}
LLM: Recent FDA approvals include...
```

---

### MCP Tool 3: health_check()

**Purpose:** Verify MCP server is responsive

**Function Signature:**
```python
def health_check() -> dict
```

**Parameters:** None

**Returns:**
```json
{
  "status": "healthy",
  "server": "Tavily MCP Server",
  "version": "1.0.0"
}
```

---

## Part 5: Tool Calling Flow (Complete Example)

### Scenario: "Find information about arrhythmias and save to a file"

**Step 1: Agent receives query**
```python
agent = SupervisorAgent()
response = agent("Find information about arrhythmias and save to a file")
```

**Step 2: LLM analyzes available tools**
```
Available tools:
1. check_knowledge_status()
2. search_knowledge_base(query, top_k)
3. check_chunks_relevance(results, question)
4. file_read(path)
5. file_write(content, filename)
6. web_search(query, max_results, ...)
7. news_search(query, days_back)
```

**Step 3: LLM creates execution plan**
```
LLM Reasoning:
1. Check if knowledge base has arrhythmia info
2. Search knowledge base
3. Evaluate relevance
4. Generate summary
5. Save to file
```

**Step 4: Tool calls sequence**

**Call 1: Check Status**
```json
{
  "tool": "check_knowledge_status",
  "arguments": {}
}
```
**Result:**
```json
{
  "status": "ready",
  "document_count": 150
}
```

**Call 2: Search**
```json
{
  "tool": "search_knowledge_base",
  "arguments": {
    "query": "arrhythmias cardiac treatment",
    "top_k": 5
  }
}
```
**Result:**
```json
{
  "results": [...],
  "relevance_score": 0.92
}
```

**Call 3: Evaluate**
```json
{
  "tool": "check_chunks_relevance",
  "arguments": {
    "results": "Score: 0.95\nContent: ...",
    "question": "arrhythmias information"
  }
}
```
**Result:**
```json
{
  "chunk_relevance_score": "yes",
  "chunk_relevance_value": 0.89
}
```

**Call 4: Generate Summary**
```
LLM: Based on relevant chunks, generate comprehensive summary
Output: "# Arrhythmias\n\nArrhythmias are irregular heartbeats..."
```

**Call 5: Save File**
```json
{
  "tool": "file_write",
  "arguments": {
    "content": "# Arrhythmias\n\nArrhythmias are...",
    "filename": "arrhythmias_summary.md"
  }
}
```
**Result:**
```
"✅ File written successfully to output/arrhythmias_summary.md (2048 bytes)"
```

**Step 5: Final Response**
```
"I've researched arrhythmias and saved a comprehensive summary to
output/arrhythmias_summary.md. The information was found in the knowledge
base with high relevance (0.89). The summary covers definition, types,
symptoms, and treatments."
```

---

## Part 6: Extending the System with New Tools

### Adding a Local Tool

**Example: Add a calculation tool**

```python
@tool
def calculate_dosage(weight_kg: float, drug: str) -> dict:
    """
    Calculate drug dosage based on patient weight.

    Args:
        weight_kg: Patient weight in kilograms
        drug: Drug name (e.g., "aspirin", "ibuprofen")

    Returns:
        dict: Dosage recommendation
    """
    dosage_table = {
        "aspirin": 10,  # mg per kg
        "ibuprofen": 5
    }

    mg_per_kg = dosage_table.get(drug, 0)
    total_dose = weight_kg * mg_per_kg

    return {
        "drug": drug,
        "weight_kg": weight_kg,
        "dosage_mg": total_dose,
        "warning": "Consult physician for actual prescription"
    }

# Add to agent
agent = create_traced_agent(
    Agent,
    model=get_reasoning_model(),
    tools=[
        search_knowledge_base,
        calculate_dosage,  # ← New tool
        file_write
    ],
    system_prompt="..."
)
```

**LLM Usage:**
```
User: "Calculate aspirin dosage for a 70kg patient"
LLM: *calls calculate_dosage(weight_kg=70, drug="aspirin")*
System: {"dosage_mg": 700, "warning": "Consult physician..."}
LLM: The recommended dosage is 700mg. However, consult a physician...
```

---

### Adding an MCP Tool Server

**Example: Database query tool**

**File:** `src/mcp_servers/database_server.py`

```python
from fastmcp import FastMCP

mcp = FastMCP("Database Query Server")

@mcp.tool(description="Query patient database")
def query_patients(condition: str, limit: int = 10) -> list:
    """
    Query patients by medical condition.

    Args:
        condition: Medical condition to filter by
        limit: Maximum results

    Returns:
        list: Patient records (anonymized)
    """
    # Connect to database
    results = db.query(
        "SELECT patient_id, age, diagnosis FROM patients WHERE diagnosis = ?",
        (condition,)
    ).limit(limit)

    return [dict(row) for row in results]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8002)
```

**Kubernetes Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: database-mcp-server
spec:
  template:
    spec:
      containers:
      - name: mcp-server
        image: database-mcp-server:latest
        ports:
        - containerPort: 8002
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
```

**Update Agent:**
```python
# Add new MCP client
db_mcp_client = MCPClient(
    lambda: streamablehttp_client("http://database-mcp-server:8002")
)

with db_mcp_client:
    db_tools = db_mcp_client.list_tools_sync()
    # Now agent can call query_patients()
```

---

## Part 7: Tool Calling Best Practices

### 1. Clear Descriptions

**❌ Bad:**
```python
@tool
def search(q: str) -> str:
    """Search stuff"""
```

**✅ Good:**
```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the medical knowledge base for relevant information.

    Use this when the user asks medical questions about:
    - Diagnoses
    - Treatments
    - Drug information
    - Clinical studies

    Args:
        query (str): Search query - describe what information is needed
        top_k (int): Number of results (default: 3, max: 10)

    Returns:
        str: JSON with search results and relevance scores
    """
```

### 2. Type Hints

**Always include type hints:**
```python
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    # ✅ Clear parameter types and return type
```

### 3. Error Handling

**Return error information:**
```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    try:
        results = retriever.search(query, top_k)
        return json.dumps({"results": results})
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "query": query,
            "status": "failed"
        })
```

### 4. Input Validation

```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    # Validate query
    if not query or not isinstance(query, str):
        return json.dumps({"error": "Query must be a non-empty string"})

    # Validate top_k
    if top_k < 1 or top_k > 10:
        return json.dumps({"error": "top_k must be between 1 and 10"})

    # Proceed with search...
```

---

## Summary

### Tool Inventory

| Tool Name | Type | Purpose | Response Time |
|-----------|------|---------|---------------|
| check_knowledge_status | Local | Check KB readiness | ~10ms |
| search_knowledge_base | Local | Vector search | ~70ms |
| check_chunks_relevance | Local | RAG evaluation | ~800ms |
| file_read | Local | Read files | ~5ms |
| file_write | Local | Write files | ~10ms |
| web_search | MCP | Real-time web search | ~1000ms |
| news_search | MCP | Recent news | ~1000ms |
| health_check | MCP | Server health | ~5ms |

### Architecture Benefits

1. **Modularity:** Tools are independent, reusable functions
2. **Extensibility:** Easy to add new tools
3. **Observability:** Each tool call is traced
4. **Security:** MCP tools run in isolated containers
5. **Standard Interface:** All tools follow same calling convention

This reference provides everything needed to understand, use, and extend the tool-calling system in this multi-agent architecture.
