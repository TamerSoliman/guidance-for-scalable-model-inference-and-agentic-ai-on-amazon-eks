"""
============================================================================
HEAVILY ANNOTATED: Supervisor Agent - Multi-Agent Orchestrator
============================================================================

FILE PURPOSE:
This is the "brain" of the multi-agent system. The Supervisor Agent
coordinates between different specialized agents and external tools to
answer complex user queries.

ARCHITECTURE PATTERN: Multi-Agent Orchestration
- Supervisor Agent: Makes high-level decisions, coordinates workflow
- Knowledge Agent: Manages RAG (retrieval augmented generation)
- MCP Agent: Executes external tools (file I/O, web search)

FRAMEWORKS USED:
1. Strands SDK: Agent framework with built-in tracing
2. MCP (Model Context Protocol): Standard for tool integration
3. RAGAs: Relevance evaluation for RAG results
4. OpenTelemetry: Distributed tracing
5. Langfuse: LLM observability

ANALOGY FOR AI RESEARCHERS:
Think of this like a PhD advisor (Supervisor) who:
- Delegates literature review to a research assistant (Knowledge Agent)
- Delegates data collection to a lab technician (MCP Agent)
- Evaluates quality of retrieved papers (RAGAs)
- Decides whether to use internal knowledge or external sources
============================================================================
"""

# ============================================================================
# SECTION 1: Imports and Setup
# ============================================================================

# ----------------------------------------------------------------------------
# 1.1 Global Async Cleanup - MUST BE FIRST
# ----------------------------------------------------------------------------
# WHAT: Sets up cleanup handlers for async operations
# WHY: Python's asyncio can leave zombie tasks when programs exit
# IMPACT: Prevents warning messages and resource leaks
from ..utils.global_async_cleanup import setup_global_async_cleanup

import asyncio
import re
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

# ----------------------------------------------------------------------------
# 1.2 Strands SDK Imports - Agent Framework
# ----------------------------------------------------------------------------
# WHAT IS STRANDS SDK?
# A framework for building LLM agents with:
# - Automatic tool calling (LLM decides when to invoke functions)
# - Built-in OpenTelemetry tracing
# - Multi-agent coordination
# - Memory management

from strands import Agent, tool
from strands_tools import file_read
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

# ----------------------------------------------------------------------------
# 1.3 LangChain Imports - For Bedrock Integration
# ----------------------------------------------------------------------------
# WHAT: LangChain provides LLM wrappers for different providers
# WHY: We use AWS Bedrock for RAG relevance evaluation (separate from main LLM)
from langchain_aws import ChatBedrockConverse

# ----------------------------------------------------------------------------
# 1.4 RAGAs Imports - Relevance Evaluation
# ----------------------------------------------------------------------------
# WHAT IS RAGAS?
# RAGAs = Retrieval Augmented Generation Assessment
# A framework for evaluating the quality of RAG systems
#
# KEY METRICS:
# - Context Precision: Are retrieved chunks relevant to the query?
# - Context Recall: Did we retrieve all necessary information?
# - Faithfulness: Is the answer grounded in the retrieved context?
#
# WHY USE IT HERE?
# To decide: "Should I use RAG results or search the web?"

try:
    # Try new RAGAs version
    from ragas.dataset_schema import SingleTurnSample
except ImportError:
    # Fallback for older versions
    try:
        from ragas import SingleTurnSample
    except ImportError:
        # If RAGAs not available, create a mock class
        # This ensures code runs even without RAGAs
        class SingleTurnSample:
            """Mock class for RAGAs dataset schema"""
            def __init__(self, user_input, response, retrieved_contexts):
                self.user_input = user_input
                self.response = response
                self.retrieved_contexts = retrieved_contexts

try:
    # Metric: Evaluates if retrieved context is relevant
    from ragas.metrics import LLMContextPrecisionWithoutReference
except ImportError:
    # Fallback: Simple heuristic-based evaluation
    class LLMContextPrecisionWithoutReference:
        """Mock metric when RAGAs unavailable"""
        def __init__(self, llm=None):
            self.llm = llm

        def score(self, sample):
            # Simple heuristic: If we have contexts, assume decent quality
            if hasattr(sample, 'retrieved_contexts') and sample.retrieved_contexts:
                return 0.7  # Default reasonable relevance score
            return 0.3

try:
    # Wrapper to adapt LangChain LLMs for RAGAs
    from ragas.llms import LangchainLLMWrapper
except ImportError:
    class LangchainLLMWrapper:
        """Mock wrapper when RAGAs unavailable"""
        def __init__(self, llm):
            self.llm = llm

# ----------------------------------------------------------------------------
# 1.5 Application-Specific Imports
# ----------------------------------------------------------------------------
from ..config import config
from ..utils.logging import log_title
from ..utils.model_providers import get_reasoning_model
from ..utils.strands_langfuse_integration import create_traced_agent, setup_tracing_environment
from ..utils.async_cleanup import suppress_async_warnings, setup_async_environment
from ..tools.embedding_retriever import EmbeddingRetriever
from .mcp_agent import file_write  # Reuse file_write from MCP agent

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 1.6 Initialize Tracing and Async Environment
# ----------------------------------------------------------------------------
# WHAT: Sets up OpenTelemetry tracing and async event loop management
# WHY: Allows us to trace agent workflows in Langfuse/Jaeger
setup_tracing_environment()
setup_async_environment()

# ============================================================================
# SECTION 2: Evaluation Model Configuration (AWS Bedrock)
# ============================================================================

# DESIGN DECISION: Use separate model for evaluation
# WHY:
# - Main LLM (Qwen3-14B via vLLM): Fast inference, high throughput
# - Evaluation LLM (Claude via Bedrock): Better reasoning for quality assessment
#
# ANALOGY: Like having a peer reviewer (Claude) check quality of
#          research done by a research assistant (Qwen3)

eval_modelId = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'

# Disable extended thinking for faster evaluation
thinking_params = {
    "thinking": {
        "type": "disabled"
    }
}

# Lazy initialization - don't connect to Bedrock until needed
llm_for_evaluation = None

def get_evaluation_llm():
    """
    Lazy initialization of AWS Bedrock LLM for RAGAs evaluation.

    WHY LAZY?
    - Avoids AWS credential errors during module import
    - Only connects when actually needed
    - Allows graceful fallback if Bedrock unavailable

    FALLBACK STRATEGY:
    If Bedrock fails, use a mock LLM that returns default scores
    """
    global llm_for_evaluation

    if llm_for_evaluation is None:
        try:
            from langchain_aws import ChatBedrockConverse
            llm_for_evaluation = ChatBedrockConverse(
                model=eval_modelId,
                additional_model_request_fields=thinking_params
            )
            llm_for_evaluation = LangchainLLMWrapper(llm_for_evaluation)
        except Exception as e:
            logger.warning(f"Failed to initialize evaluation LLM: {e}")

            # Create a mock LLM for evaluation
            class MockLLM:
                def invoke(self, prompt):
                    class MockResponse:
                        content = "Mock evaluation response"
                    return MockResponse()

            llm_for_evaluation = MockLLM()

    return llm_for_evaluation

# ============================================================================
# SECTION 3: MCP Client Setup (External Tool Integration)
# ============================================================================

# WHAT IS MCP (Model Context Protocol)?
# A standard protocol for connecting LLMs to external tools and data sources.
#
# ARCHITECTURE:
# ┌─────────────────┐         ┌──────────────────┐
# │  Supervisor     │         │  Tavily MCP      │
# │  Agent (Client) │ <-----> │  Server          │
# └─────────────────┘         └──────────────────┘
#       |                            |
#       | Tool calls                 | Web search API
#       v                            v
#   LLM decides                  External API
#   when to search               (Tavily.com)
#
# BENEFITS:
# - Standardized tool interface
# - Tools are language-agnostic (server can be in any language)
# - Hot-reload tools without restarting agent
# - Security: Tools run in separate process

tavily_mcp_client = None

def get_tavily_mcp_client():
    """
    Initialize MCP client for Tavily web search tool.

    WHAT IS TAVILY?
    A web search API optimized for LLM applications:
    - Returns clean, LLM-friendly results
    - Filters out ads and navigation
    - Provides relevance scores
    - Supports news search with time filters

    CONNECTION:
    - URL: Kubernetes Service DNS (tavily-mcp-server:8000)
    - Protocol: Streamable HTTP (keeps connection open)

    ERROR HANDLING:
    - If MCP server unavailable, agent falls back to RAG-only mode
    """
    global tavily_mcp_client

    if tavily_mcp_client is None:
        try:
            # Get MCP server URL from config (env var)
            mcp_url = config.TAVILY_MCP_SERVICE_URL

            # Create client with streamable HTTP transport
            # WHAT IS STREAMABLE HTTP?
            # Keeps HTTP connection open for bidirectional streaming
            # Allows server to push updates to client
            tavily_mcp_client = MCPClient(
                lambda: streamablehttp_client(mcp_url)
            )
            logger.info(f"Tavily MCP client initialized: {mcp_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily MCP client: {e}")
            tavily_mcp_client = None

    return tavily_mcp_client

# ============================================================================
# SECTION 4: Relevance Scoring Function
# ============================================================================

def calculate_relevance_score(results: List[Dict], query: str) -> float:
    """
    Calculate relevance score with content validation to prevent false positives.

    PROBLEM THIS SOLVES:
    Vector search can return high similarity scores for semantically similar
    but contextually wrong documents. Example:
    - Query: "What's the weather in Seattle today?"
    - Bad match: "The weather in Portland is sunny" (high vector similarity)
    - Good match: "Seattle weather: rainy, 55°F" (relevant content)

    VALIDATION STRATEGY:
    1. Vector similarity score (from OpenSearch)
    2. Keyword overlap ratio (query words in document)
    3. Domain-specific validation (e.g., weather queries need weather terms)

    PARAMETERS:
        results: List of search results with scores and content
        query: Original search query

    RETURNS:
        float: Validated relevance score (0.0 to 1.0)
        - 0.0-0.3: Low relevance (trigger web search)
        - 0.3-0.7: Medium relevance
        - 0.7-1.0: High relevance (use RAG results confidently)
    """
    if not results:
        return 0.0

    # Extract scores and validate content relevance
    scores = []
    query_lower = query.lower()
    query_keywords = set(query_lower.split())

    for result in results:
        # Get the similarity score from OpenSearch
        score = None
        if isinstance(result, dict):
            # Try multiple possible score field names
            score = result.get('score') or result.get('_score')
            if score is None and 'metadata' in result:
                score = result['metadata'].get('score')

        if score is not None:
            # Validate content relevance by checking keyword overlap
            content = result.get('content', '').lower()
            content_keywords = set(content.split())

            # Calculate keyword overlap ratio
            overlap = len(query_keywords.intersection(content_keywords))
            overlap_ratio = overlap / len(query_keywords) if query_keywords else 0

            # PENALTY SYSTEM: Reduce score for low keyword overlap
            if overlap_ratio < 0.1:  # Less than 10% keyword overlap
                score = score * 0.2  # Heavily penalize
            elif overlap_ratio < 0.3:  # Less than 30% keyword overlap
                score = score * 0.5  # Moderately penalize

            scores.append(float(score))

    if not scores:
        return 0.0

    # Calculate average score
    avg_score = sum(scores) / len(scores)

    # DOMAIN-SPECIFIC VALIDATION: Weather queries
    # WHY: Weather queries are time-sensitive and often mismatch with old data
    if any(keyword in query_lower for keyword in ['weather', 'temperature', 'forecast']):
        # Check if results contain weather-related terms
        weather_terms = [
            'weather', 'temperature', 'rain', 'sunny', 'cloudy',
            'forecast', 'celsius', 'fahrenheit'
        ]
        has_weather_content = False

        for result in results:
            content = result.get('content', '').lower()
            if any(term in content for term in weather_terms):
                has_weather_content = True
                break

        if not has_weather_content:
            # Heavily penalize non-weather content for weather queries
            avg_score = avg_score * 0.1

    return min(avg_score, 1.0)

# ============================================================================
# SECTION 5: Agent Tools (Functions the LLM Can Call)
# ============================================================================

# WHAT IS A TOOL?
# A Python function decorated with @tool that the LLM can invoke.
#
# HOW IT WORKS:
# 1. LLM reads tool description and parameters
# 2. When LLM decides to use a tool, it returns a structured function call
# 3. Strands SDK executes the function and feeds result back to LLM
# 4. LLM uses the result to formulate final answer
#
# EXAMPLE CONVERSATION:
# User: "What is Bell's palsy?"
# LLM: *calls search_knowledge_base(query="Bell's palsy")*
# System: *returns results*
# LLM: "Bell's palsy is a condition that causes sudden weakness..."

# ----------------------------------------------------------------------------
# TOOL 1: check_chunks_relevance - RAG Quality Evaluation
# ----------------------------------------------------------------------------

def _run_async_evaluation_safe(scorer, sample):
    """
    Helper function to run async RAGAs evaluation with proper cleanup.

    PROBLEM:
    RAGAs uses asyncio internally, which can conflict with Strands SDK's
    async operations, leaving zombie tasks.

    SOLUTION:
    Run evaluation in a separate thread with its own event loop,
    then properly clean up all async resources.

    IMPLEMENTATION:
    1. Create new thread
    2. Thread creates new asyncio event loop
    3. Run evaluation with timeout
    4. Cancel all pending tasks
    5. Close loop properly
    6. Return result via queue

    PARAMETERS:
        scorer: RAGAs LLMContextPrecisionWithoutReference instance
        sample: SingleTurnSample with query, response, contexts

    RETURNS:
        float: Evaluation score (0.0 to 1.0)

    RAISES:
        TimeoutError: If evaluation takes >25 seconds
        Exception: If evaluation fails
    """
    import threading
    import queue

    def run_evaluation():
        """Inner function that runs in separate thread"""
        async def evaluate():
            try:
                # Set timeout to prevent hanging
                score = await asyncio.wait_for(
                    scorer.single_turn_ascore(sample),
                    timeout=20.0
                )
                return score
            except asyncio.TimeoutError:
                raise TimeoutError("RAGAs evaluation timed out")
            except Exception as e:
                raise Exception(f"RAGAs evaluation failed: {str(e)}")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run evaluation
            score = loop.run_until_complete(evaluate())
            result_queue.put(('success', score))
        except Exception as e:
            result_queue.put(('error', str(e)))
        finally:
            # CRITICAL: Properly clean up async resources
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Wait for tasks to complete cancellation
                if pending:
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=2.0
                            )
                        )
                    except asyncio.TimeoutError:
                        pass  # Ignore timeout during cleanup

                # Close the loop
                loop.close()
            except Exception:
                pass  # Ignore cleanup errors

    # Use queue for thread communication
    result_queue = queue.Queue()

    # Run in separate thread
    thread = threading.Thread(target=run_evaluation, daemon=True)
    thread.start()
    thread.join(timeout=25)  # Wait up to 25 seconds

    if thread.is_alive():
        raise TimeoutError("Evaluation thread timed out")

    # Get result from queue
    try:
        result_type, result_value = result_queue.get_nowait()
        if result_type == 'error':
            raise Exception(result_value)
        return result_value
    except queue.Empty:
        raise Exception("No result received from evaluation")

@tool
def check_chunks_relevance(results: str, question: str):
    """
    Evaluates the relevance of retrieved chunks to the user question using RAGAs.

    WHEN TO USE:
    After search_knowledge_base() returns results, use this tool to assess
    if the retrieved documents are actually relevant to the query.

    WHY NEEDED:
    Vector search can return high similarity scores for semantically similar
    but contextually irrelevant documents. This provides a second layer of
    validation using an LLM to evaluate relevance.

    WORKFLOW:
    1. Parse retrieved chunks from string format
    2. Generate an answer using the chunks (via Bedrock)
    3. Evaluate if the chunks were useful in generating the answer
    4. Return binary decision: "yes" (relevant) or "no" (not relevant)

    PARAMETERS:
        results (str): Retrieval output with 'Score:' and 'Content:' patterns
        question (str): Original user question

    RETURNS:
        dict: {
            "chunk_relevance_score": "yes" or "no",
            "chunk_relevance_value": float (0.0 to 1.0),
            "evaluation_method": "ragas" or "fallback_heuristic"
        }

    DECISION THRESHOLD:
        score > 0.5: "yes" (use RAG results)
        score <= 0.5: "no" (trigger web search)
    """
    # Use async warning suppression
    with suppress_async_warnings():
        try:
            # STEP 1: Input validation
            if not results or not isinstance(results, str):
                raise ValueError("Invalid input: 'results' must be a non-empty string.")
            if not question or not isinstance(question, str):
                raise ValueError("Invalid input: 'question' must be a non-empty string.")

            # STEP 2: Extract content chunks using regex
            # FORMAT EXPECTED:
            # Score: 0.95
            # Content: Bell's palsy is a condition...
            #
            # Score: 0.87
            # Content: Treatment typically includes...

            patterns_to_try = [
                r"Score:.*?\\nContent:\\s*(.*?)(?=\\n\\nScore:|\\Z)",  # Double newlines
                r"Score:.*?\\nContent:\\s*(.*?)(?=Score:|\\Z)",          # Original
                r"Score:\\s*[\\d.]+\\s*\\nContent:\\s*(.*?)(?=\\n\\nScore:|\\Z)",  # Specific
            ]

            docs = []
            pattern_used = None

            for i, pattern in enumerate(patterns_to_try):
                try:
                    docs = [
                        chunk.strip()
                        for chunk in re.findall(pattern, results, re.DOTALL)
                        if chunk.strip()
                    ]
                    if docs:
                        pattern_used = f"Pattern {i+1}"
                        logger.debug(f"Extracted {len(docs)} chunks using pattern {i+1}")
                        break
                except Exception as e:
                    logger.warning(f"Pattern {i+1} failed: {e}")
                    continue

            # Flexible fallback pattern if standard patterns fail
            if not docs:
                logger.warning("Standard patterns failed, trying flexible extraction...")
                flexible_pattern = r"Content:\\s*([^\\n]+(?:\\n(?!Score:)[^\\n]*)*)"
                try:
                    docs = [
                        chunk.strip()
                        for chunk in re.findall(flexible_pattern, results, re.MULTILINE)
                        if chunk.strip()
                    ]
                    if docs:
                        pattern_used = "Flexible pattern"
                        logger.info(f"Flexible extraction found {len(docs)} chunks")
                except Exception as e:
                    logger.error(f"Flexible extraction also failed: {e}")

            if not docs:
                # Provide detailed debugging information
                debug_info = {
                    "results_length": len(results),
                    "contains_score": "Score:" in results,
                    "contains_content": "Content:" in results,
                    "results_preview": results[:200] if len(results) > 200 else results
                }
                logger.error(f"No valid content chunks found. Debug info: {debug_info}")
                raise ValueError(f"No valid content chunks found in 'results'. Debug: {debug_info}")

            # STEP 3: Limit chunks to avoid timeout
            # WHY: RAGAs evaluation can be slow with many chunks
            if len(docs) > 3:
                docs = docs[:3]
                logger.info(f"Limited evaluation to first 3 chunks out of {len(docs)} total")

            # STEP 4: Generate answer from context
            # WHY: RAGAs LLMContextPrecisionWithoutReference needs an answer
            #      to evaluate if the context was useful
            try:
                context_for_answer = "\\n\\n".join(docs[:2])
                answer_prompt = f"""Based on the following context, provide a brief answer to the question.

Question: {question}
Context: {context_for_answer}

Answer:"""

                # Use Bedrock Claude for answer generation
                answer_llm = ChatBedrockConverse(
                    model='us.anthropic.claude-3-7-sonnet-20250219-v1:0'
                )
                answer_response = answer_llm.invoke(answer_prompt)
                generated_answer = answer_response.content.strip()

                logger.debug(f"Generated answer: {generated_answer[:100]}...")

            except Exception as e:
                logger.warning(f"Failed to generate answer: {e}")
                generated_answer = "Unable to generate answer from context"

            # STEP 5: Prepare evaluation sample
            sample = SingleTurnSample(
                user_input=question,
                response=generated_answer,
                retrieved_contexts=docs
            )

            # STEP 6: Evaluate using RAGAs
            scorer = LLMContextPrecisionWithoutReference(
                llm=get_evaluation_llm()
            )

            print("------------------------")
            print("Context evaluation (RAGAs)")
            print("------------------------")
            print(f"Evaluating {len(docs)} chunks...")

            # Run evaluation in safe async context
            score = _run_async_evaluation_safe(scorer, sample)

            print(f"chunk_relevance_score: {score}")
            print("------------------------")

            return {
                "chunk_relevance_score": "yes" if score > 0.5 else "no",
                "chunk_relevance_value": score
            }

        except Exception as e:
            logger.error(f"Error in chunk relevance evaluation: {e}")

            # FALLBACK: Simple keyword-based heuristic
            try:
                question_words = set(question.lower().split())
                results_words = set(results.lower().split())
                overlap = len(question_words.intersection(results_words))
                fallback_score = min(overlap / len(question_words), 1.0) if question_words else 0.0

                logger.info(f"Using fallback relevance score: {fallback_score}")

                return {
                    "chunk_relevance_score": "yes" if fallback_score > 0.3 else "no",
                    "chunk_relevance_value": fallback_score,
                    "evaluation_method": "fallback_heuristic",
                    "note": f"RAGAs evaluation failed, using keyword overlap heuristic"
                }
            except Exception as fallback_error:
                logger.error(f"Fallback evaluation also failed: {fallback_error}")
                return {
                    "error": f"Both RAGAs and fallback evaluation failed: {str(e)}",
                    "chunk_relevance_score": "unknown",
                    "chunk_relevance_value": None
                }

# (Continuing in next file due to length...)
