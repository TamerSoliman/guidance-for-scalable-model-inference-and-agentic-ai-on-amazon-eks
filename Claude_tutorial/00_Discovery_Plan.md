# EKS-Based LLM Inference and Agent Architecture - Discovery Plan

## Overview
This document catalogs the **top 50 most critical configuration files and scripts** that demonstrate the production deployment patterns for high-throughput LLM serving (using vLLM/Ray) and agentic workflows on Amazon EKS.

## Target Audience
Generative AI researchers with limited background in cloud infrastructure, MLOps, DevOps, or Kubernetes.

## Repository Architecture Summary

This repository implements a comprehensive EKS-based platform for:
1. **Scalable Model Inference**: Using vLLM (GPU) and Ray + llamacpp (CPU/Graviton)
2. **Agentic AI Applications**: Multi-agent systems using Strands SDK with MCP (Model Context Protocol)
3. **Infrastructure Automation**: Karpenter-based autoscaling for different compute types
4. **Observability**: Langfuse for LLM tracing, Prometheus/Grafana for infrastructure

---

## Category 1: Kubernetes Infrastructure & Compute Provisioning (12 files)

### 1.1 Karpenter Node Pools - Dynamic Compute Provisioning

**What is Karpenter?** Karpenter is a Kubernetes cluster autoscaler that automatically provisions the right compute resources (EC2 instances) based on your workload requirements.

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 1 | `base_eks_setup/karpenter_nodepool/gpu-nodepool.yaml` | **GPU Node Pool for vLLM Inference** | Defines how Karpenter provisions NVIDIA GPU instances (g5, g6) for accelerated LLM inference. Includes taints/tolerations to ensure only GPU workloads land here. |
| 2 | `base_eks_setup/karpenter_nodepool/graviton-nodepool.yaml` | **ARM64/Graviton Node Pool for Cost-Effective Inference** | Configures Karpenter to provision AWS Graviton (ARM64) instances for CPU-based embedding generation and cost-optimized inference. |
| 3 | `base_eks_setup/karpenter_nodepool/x86-nodepool.yaml` | **x86 CPU Node Pool for General Workloads** | Standard x86_64 compute for compatibility with workloads that don't support ARM or don't require GPUs. |
| 4 | `base_eks_setup/karpenter_nodepool/inf2-nodepool.yaml` | **AWS Inferentia Node Pool for Optimized Inference** | Provisions AWS Inferentia2 instances for ultra-low-cost, high-throughput inference (specialized AWS ML accelerators). |

### 1.2 Storage and Monitoring

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 5 | `base_eks_setup/gp3.yaml` | **GP3 Storage Class Definition** | Defines how persistent storage is provisioned for model weights and caches using AWS EBS GP3 volumes (optimized IOPS/throughput). |
| 6 | `base_eks_setup/prometheus-monitoring.yaml` | **Prometheus Monitoring Stack** | Infrastructure metrics collection for cluster health, pod resource usage, and autoscaling metrics. |
| 7 | `base_eks_setup/tracking_stack.yaml` | **Observability Tracking Stack** | Complete observability setup including Grafana dashboards and alerting for production monitoring. |

### 1.3 Setup Scripts

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 8 | `base_eks_setup/install_operators.sh` | **Automated Operator Installation** | Shell script that installs KubeRay Operator, NVIDIA GPU Operator, and applies all Karpenter node pools. |
| 9 | `Makefile` | **Complete Platform Deployment Orchestration** | Master automation file for installing all components: base infrastructure, models, gateway, observability, and agentic apps. |

---

## Category 2: Model Hosting - vLLM & Ray Serve (10 files)

### 2.1 Standalone vLLM Deployments

**What is vLLM?** vLLM is a high-throughput, memory-efficient inference engine for LLMs that uses techniques like PagedAttention to maximize GPU utilization.

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 10 | `model-hosting/standalone-vllm-reasoning.yaml` | **vLLM Deployment for Reasoning Models (Qwen3-14B)** | Complete Kubernetes Deployment showing:<br>- GPU resource allocation (`nvidia.com/gpu: 4`)<br>- Tensor parallelism configuration<br>- Model serving parameters (batch size, sequence length)<br>- Node affinity for GPU instances<br>- Persistent volume for model caching |
| 11 | `model-hosting/standalone-vllm-vision.yaml` | **vLLM Deployment for Vision-Language Models** | Configures vLLM for multimodal models (Qwen2.5-VL) that process both text and images. |

### 2.2 Ray Serve - Distributed Inference Platform

**What is Ray Serve?** Ray is a distributed computing framework. Ray Serve adds autoscaling model serving on top, allowing multiple model replicas across many nodes.

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 12 | `model-hosting/ray-services/ray-service-llamacpp-with-embedding.yaml` | **Ray Serve + llamacpp for Embedding Generation** | RayService CRD (Custom Resource Definition) showing:<br>- Ray cluster architecture (head + worker nodes)<br>- Autoscaling configuration (min/max replicas)<br>- llamacpp integration for CPU-based embeddings<br>- OpenAI-compatible API implementation<br>- Environment variable configuration for model loading |
| 13 | `model-hosting/ray-server/llamacpp.py` | **llamacpp Serving Application Code** | Python FastAPI app embedded in ConfigMap defining:<br>- Model initialization with llama-cpp-python<br>- `/v1/embeddings` endpoint implementation<br>- Request/response handling in OpenAI format<br>- Performance logging and health checks |
| 14 | `model-hosting/ray-server/vllm.py` | **vLLM Ray Serve Application Code** | Python application code for serving LLMs via Ray with vLLM backend (if exists - for completions endpoint). |

### 2.3 Ingress Configuration for External Access

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 15 | `model-hosting/ray-services/ingress/ingress-cpu.yaml` | **ALB Ingress for CPU-Based Ray Services** | AWS ALB (Application Load Balancer) configuration for exposing Ray Serve endpoints externally. |
| 16 | `model-hosting/ray-services/ingress/ingress-embedding.yaml` | **ALB Ingress for Embedding Service** | Separate ingress configuration for embedding endpoint routing and load balancing. |

### 2.4 Setup and Deployment

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 17 | `model-hosting/setup.sh` | **Model Hosting Deployment Script** | Automated script to deploy all model serving components (vLLM standalone + Ray services). |

---

## Category 3: Model Gateway - LiteLLM Proxy (4 files)

**What is LiteLLM?** LiteLLM is a unified API gateway that provides a single OpenAI-compatible interface to multiple LLM backends, with features like load balancing, caching, and observability.

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 18 | `model-gateway/litellm-deployment.yaml` | **LiteLLM Gateway Deployment** | Complete deployment showing:<br>- ConfigMap with model routing configuration<br>- Backend service definitions (vLLM, Ray Serve endpoints)<br>- Redis caching layer for response caching<br>- PostgreSQL for API key management<br>- Langfuse integration for LLM observability<br>- Custom guardrails (PII detection) |
| 19 | `model-gateway/litellm-ingress.yaml` | **ALB Ingress for LiteLLM Gateway** | External access configuration for the unified API gateway endpoint. |
| 20 | `model-gateway/setup.sh` | **Gateway Deployment Script** | Automated deployment script for LiteLLM and its dependencies. |

---

## Category 4: Agentic Applications - Multi-Agent RAG System (20 files)

### 4.1 Agent Core Logic (Strands SDK)

**What is Strands SDK?** Strands is a framework for building multi-agent LLM applications with built-in support for tool calling, observability (OpenTelemetry), and the Model Context Protocol (MCP).

**What is MCP?** Model Context Protocol is a standard for connecting LLMs to external data sources and tools through a client-server architecture.

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 21 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/supervisor_agent.py` | **Supervisor Agent - Main Orchestrator** | The brain of the multi-agent system:<br>- Coordinates between RAG, MCP, and web search<br>- Implements relevance scoring with RAGAs<br>- Tool definitions (@tool decorator)<br>- Decision logic for RAG vs. web search<br>- MCP client integration<br>- Built-in OpenTelemetry tracing |
| 22 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/knowledge_agent.py` | **Knowledge Agent - RAG Management** | Manages the knowledge base:<br>- Document embedding pipeline<br>- Change detection and incremental updates<br>- Integration with OpenSearch vector store |
| 23 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/mcp_agent.py` | **MCP Agent - Tool Execution** | Executes external tools via MCP:<br>- File operations (read/write)<br>- Custom tool wrapping<br>- Context passing between agents |
| 24 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/agents/rag_agent.py` | **RAG Agent - Retrieval Logic** | Specialized retrieval agent for semantic search and context assembly. |

### 4.2 Vector Store and Search Tools

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 25 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/tools/opensearch_vector_store.py` | **OpenSearch Vector Database Integration** | Vector similarity search implementation:<br>- k-NN index creation<br>- Embedding storage and retrieval<br>- Metadata filtering |
| 26 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/tools/embedding_retriever.py` | **Embedding Retrieval Tool** | High-level API for:<br>- Query embedding generation<br>- Similarity search execution<br>- Result ranking and formatting |

### 4.3 MCP Servers - External Tool Interfaces

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 27 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/mcp_servers/tavily_search_server.py` | **Tavily Web Search MCP Server** | MCP server implementation for real-time web search:<br>- Tavily API integration<br>- Tool definition for web_search()<br>- News search capabilities<br>- Streamable HTTP transport |
| 28 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/mcp_servers/mcp_filesystem_server.py` | **Filesystem MCP Server** | MCP server for file operations as external tools accessible to agents. |

### 4.4 Utilities and Configuration

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 29 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/config.py` | **Application Configuration** | Centralized config loading from environment variables (API keys, endpoints, model names). |
| 30 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/utils/model_providers.py` | **LLM Provider Abstraction** | Factory functions for creating LLM clients that work with LiteLLM gateway. |
| 31 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/utils/strands_langfuse_integration.py` | **Strands + Langfuse Tracing Integration** | Custom integration bridging Strands SDK's OpenTelemetry traces to Langfuse for observability. |
| 32 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/utils/opensearch_client.py` | **OpenSearch Client Utilities** | AWS-authenticated OpenSearch client initialization. |
| 33 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/utils/langfuse_config.py` | **Langfuse Configuration** | Setup for LLM call tracing and performance monitoring. |

### 4.5 Application Server

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 34 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/server.py` | **FastAPI Application Server** | REST API exposing agent capabilities:<br>- `/query` endpoint for user questions<br>- `/embed` endpoint for knowledge ingestion<br>- `/health` for readiness checks<br>- Request/response models |
| 35 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/main.py` | **Main Application Entry Point** | CLI interface and application initialization orchestration. |

### 4.6 Kubernetes Deployment for Agentic App

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 36 | `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/main-app-deployment.yaml` | **Main Agent Application Deployment** | Kubernetes Deployment for the agentic app:<br>- ConfigMap/Secret mounting for sensitive config<br>- EKS Pod Identity for AWS service access<br>- Health check configuration<br>- Resource limits and requests<br>- ALB Ingress with extended timeout (900s for long agent executions) |
| 37 | `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/tavily-mcp-deployment.yaml` | **Tavily MCP Server Deployment** | Separate pod running the Tavily MCP server as a microservice. |
| 38 | `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/configmap.yaml` | **Application ConfigMap** | Non-sensitive configuration (API endpoints, model names, index names). |
| 39 | `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/service-account.yaml` | **EKS Service Account with IAM Role** | Kubernetes ServiceAccount mapped to IAM role for OpenSearch access via EKS Pod Identity. |

### 4.7 Build and Deployment Scripts

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 40 | `agentic-apps/strandsdk_agentic_rag_opensearch/build-images.sh` | **Container Build and ECR Push Script** | Automates:<br>- Docker image builds for app and MCP servers<br>- ECR repository creation<br>- Image pushing<br>- K8s manifest updates with image URLs |

---

## Category 5: Vector Database Deployments (4 files)

### 5.1 OpenSearch for RAG

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 41 | `agentic-apps/strandsdk_agentic_rag_opensearch/opensearch-cluster-simple.yaml` | **OpenSearch Cluster CloudFormation Template** | Deploys managed OpenSearch domain for vector storage with k-NN plugin enabled. |
| 42 | `agentic-apps/strandsdk_agentic_rag_opensearch/deploy-opensearch.sh` | **OpenSearch Deployment Automation** | Script to deploy OpenSearch via CloudFormation and configure EKS Pod Identity. |
| 43 | `agentic-apps/strandsdk_agentic_rag_opensearch/setup_opensearch_index.py` | **Vector Index Initialization** | Python script to create the k-NN index with proper mappings for embeddings. |

### 5.2 Milvus Alternative

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 44 | `milvus/milvus-standalone.yaml` | **Milvus Vector Database Deployment** | Alternative vector database deployment (Milvus) for RAG applications. |

---

## Category 6: Observability (2 files)

**What is Langfuse?** Langfuse is an open-source LLM observability platform that tracks LLM calls, costs, latency, and traces multi-agent workflows.

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 45 | `model-observability/langfuse-value.yaml` | **Langfuse Helm Chart Values** | Configuration for deploying Langfuse (Postgres backend, Redis, web UI). |
| 46 | `model-observability/langfuse-web-ingress.yaml` | **Langfuse Web UI Ingress** | ALB configuration to access Langfuse dashboard externally. |

---

## Category 7: Testing and Examples (4 files)

| # | File Path | Architecture Component | Key Concepts |
|---|-----------|----------------------|--------------|
| 47 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/test_agents.py` | **Agent Integration Tests** | Test suite for multi-agent workflows including web search integration. |
| 48 | `agentic-apps/strandsdk_agentic_rag_opensearch/run_single_query_clean.py` | **Single Query CLI Test** | Standalone script for testing agent with a single query (clean output mode). |
| 49 | `agentic-apps/strandsdk_agentic_rag_opensearch/src/scripts/embed_knowledge.py` | **Knowledge Embedding Script** | Batch embedding script to populate vector store from knowledge directory. |
| 50 | `README.md` | **Complete Deployment Guide** | Comprehensive documentation covering architecture, cost analysis, security, and step-by-step deployment instructions. |

---

## Key Architecture Patterns Identified

### 1. **Infrastructure Layer (Kubernetes + Karpenter)**
- **Pattern**: Dynamic compute provisioning using Karpenter NodePools
- **Why**: Automatically scale GPU, ARM64, and x86 nodes based on workload demands
- **Key Files**: #1-4 (NodePool YAMLs)

### 2. **Model Serving Layer (vLLM + Ray)**
- **Pattern**: Dual serving strategy - standalone vLLM for GPUs, Ray Serve for CPU/Graviton
- **Why**: GPU for high-throughput reasoning, CPU for cost-effective embeddings
- **Key Files**: #10, #12, #13 (vLLM Deployment, Ray Service, llamacpp app)

### 3. **API Gateway Layer (LiteLLM)**
- **Pattern**: Unified OpenAI-compatible gateway with caching and routing
- **Why**: Single API for multiple backends, response caching, observability integration
- **Key Files**: #18 (LiteLLM Deployment)

### 4. **Agentic Layer (Strands SDK + MCP)**
- **Pattern**: Multi-agent orchestration with external tool integration via MCP
- **Why**: Modular agents with specialized responsibilities, extensible tool system
- **Key Files**: #21-24 (Agent implementations), #27-28 (MCP servers)

### 5. **Data Layer (OpenSearch Vector Store)**
- **Pattern**: Managed OpenSearch with k-NN for semantic search
- **Why**: Scalable vector storage with AWS native integration (Pod Identity)
- **Key Files**: #25, #41 (Vector store integration, CloudFormation)

### 6. **Observability Layer (Langfuse + OpenTelemetry)**
- **Pattern**: Distributed tracing from LLM gateway through agent workflows
- **Why**: Track costs, latency, and agent decision paths in production
- **Key Files**: #31 (Strands-Langfuse integration), #45 (Langfuse deployment)

---

## Next Steps

This discovery plan will guide the following deep-dive activities:

1. **Phase 2**: Annotate the top 15 Kubernetes/Python files with inline explanations
2. **Phase 3**: Create reference guides mapping concepts to code
3. **Deliverable**: Educational materials for AI researchers new to cloud infrastructure

---

## Glossary for AI Researchers

**Kubernetes (K8s)**: Container orchestration system that automatically deploys, scales, and manages containerized applications across a cluster of machines.

**Pod**: The smallest deployable unit in Kubernetes - typically one container or a group of tightly coupled containers.

**Deployment**: A Kubernetes resource that manages a replicated set of Pods (e.g., 3 copies of your model server).

**Service**: A stable network endpoint that load-balances traffic across Pods (even as Pods are created/destroyed).

**Ingress**: HTTP(S) routing rules that expose Services to external users (maps `http://your-domain.com/api` to your backend Service).

**ConfigMap**: Non-sensitive configuration data stored in Kubernetes (API endpoints, model names).

**Secret**: Sensitive data stored encrypted in Kubernetes (API keys, passwords).

**NodePool (Karpenter)**: A template that tells Karpenter what type of EC2 instances to provision (GPU, CPU, memory size).

**PersistentVolume**: Durable storage that survives Pod restarts (for model weights).

**Ray**: A distributed computing framework for Python that scales workloads across multiple machines.

**vLLM**: A fast LLM inference engine that optimizes GPU memory usage with techniques like PagedAttention.

**Tensor Parallelism**: Splitting a large model across multiple GPUs so layers are computed in parallel.

**Tool Calling**: LLM capability to invoke external functions (e.g., web search, file write) based on natural language requests.

**Vector Database**: A database optimized for similarity search over high-dimensional embeddings.

**k-NN Index**: k-Nearest Neighbors index - data structure for fast vector similarity search.

**RAG (Retrieval Augmented Generation)**: Pattern where an LLM's response is enhanced by retrieving relevant documents from a knowledge base.

**MCP (Model Context Protocol)**: A standard for connecting LLMs to external tools and data sources via a client-server protocol.
