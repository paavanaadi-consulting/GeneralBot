"""
Agentic RAG: Agent-Based Query Processing and Reasoning

This module implements Agentic RAG which uses agent-based approaches for
intelligent query reformulation, tool selection, and multi-step reasoning.

Key Features:
- Autonomous query analysis and reformulation
- Multi-turn conversation context
- Query expansion and decomposition
- Tool selection and execution
- Multi-step reasoning chains
- Self-correction and refinement

Note: This is a composable implementation using existing techniques.
For full agentic capabilities with external tools, consider using:
- LangChain Agents
- LlamaIndex Agents  
- Contextual AI platform
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from ..core.base import BaseRAGTechnique

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for agent classification."""
    SIMPLE_FACT = "simple_fact"
    MULTI_PART = "multi_part"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


class AgentAction(Enum):
    """Actions the agent can take."""
    DIRECT_RETRIEVAL = "direct_retrieval"
    QUERY_DECOMPOSITION = "query_decomposition"
    QUERY_EXPANSION = "query_expansion"
    MULTI_TURN_CONTEXT = "multi_turn_context"
    WEB_SEARCH = "web_search"
    SELF_CORRECTION = "self_correction"


@dataclass
class AgenticRAGConfig:
    """Configuration for Agentic RAG."""
    
    # Agent behavior
    enable_query_analysis: bool = True
    enable_multi_turn: bool = True
    enable_self_correction: bool = True
    max_reasoning_steps: int = 5
    
    # Query reformulation
    enable_decomposition: bool = True
    enable_expansion: bool = True
    enable_context_addition: bool = True
    
    # Tool selection
    available_tools: List[str] = None
    enable_web_search: bool = False
    
    # Quality thresholds
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.6
    
    # LLM settings
    agent_model: str = "gpt-4"
    agent_temperature: float = 0.0


class AgenticRAG(BaseRAGTechnique):
    """
    Implements Agentic RAG with intelligent query processing.
    
    This technique uses agent-based reasoning to intelligently process
    queries, selecting and executing appropriate strategies. It can:
    
    - Analyze query type and complexity
    - Reformulate queries (expand, decompose, contextualize)
    - Select appropriate retrieval strategies
    - Perform multi-step reasoning
    - Self-correct and refine responses
    
    The agent autonomously decides:
    1. How to interpret the query
    2. What reformulation strategy to use
    3. How many retrieval steps needed
    4. When to refine or correct
    
    Example:
        >>> config = AgenticRAGConfig(
        ...     enable_decomposition=True,
        ...     enable_self_correction=True,
        ...     max_reasoning_steps=3
        ... )
        >>> agent_rag = AgenticRAG(config=config)
        >>> 
        >>> # Index documents
        >>> agent_rag.index_documents(documents)
        >>> 
        >>> # Query with agentic processing
        >>> result = agent_rag.retrieve_and_generate(
        ...     query="Compare X and Y, then explain implications",
        ...     conversation_history=previous_messages
        ... )
        >>> 
        >>> # View agent's reasoning
        >>> print("Agent actions:", result['agent_actions'])
        >>> print("Reasoning steps:", result['reasoning_steps'])
    """
    
    def __init__(
        self,
        config: Optional[AgenticRAGConfig] = None,
        **kwargs
    ):
        """Initialize Agentic RAG.
        
        Args:
            config: Configuration for agentic behavior
            **kwargs: Additional arguments passed to BaseRAGTechnique
        """
        super().__init__(**kwargs)
        self.config = config or AgenticRAGConfig()
        
        if self.config.available_tools is None:
            self.config.available_tools = [
                "retrieval",
                "decomposition",
                "expansion",
                "reranking"
            ]
        
        # Initialize agent LLM (potentially different from main LLM)
        self._init_agent_llm()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(
            f"Initialized AgenticRAG with {len(self.config.available_tools)} tools"
        )
    
    def _init_agent_llm(self):
        """Initialize LLM for agent reasoning."""
        try:
            from langchain_openai import ChatOpenAI
            self.agent_llm = ChatOpenAI(
                model=self.config.agent_model,
                temperature=self.config.agent_temperature
            )
        except ImportError:
            logger.warning("Using same LLM for agent and generation")
            self.agent_llm = self.llm
    
    def analyze_query(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze query to determine type and processing strategy.
        
        Args:
            query: User query
            conversation_context: Previous conversation turns
            
        Returns:
            Dictionary with analysis results:
            - query_type: Type of query
            - complexity: Complexity score
            - recommended_action: Suggested processing
            - needs_context: Whether context is needed
        """
        if not self.config.enable_query_analysis:
            return {
                "query_type": QueryType.SIMPLE_FACT,
                "complexity": 0.5,
                "recommended_action": AgentAction.DIRECT_RETRIEVAL,
                "needs_context": False
            }
        
        # Build analysis prompt
        context_info = ""
        if conversation_context:
            context_info = f"\nPrevious conversation:\n"
            for turn in conversation_context[-3:]:
                context_info += f"User: {turn.get('user', '')}\n"
                context_info += f"Assistant: {turn.get('assistant', '')}\n"
        
        prompt = f"""Analyze the following query and determine the best processing strategy.

Query: {query}
{context_info}

Classify the query type:
- simple_fact: Direct factual question
- multi_part: Multiple questions or aspects
- comparative: Comparing multiple things
- analytical: Requires reasoning or analysis
- conversational: Refers to previous context

Determine:
1. Query type
2. Complexity (0-1)
3. Recommended action (direct_retrieval, query_decomposition, query_expansion, multi_turn_context)
4. Whether previous conversation context is needed

Respond in JSON format:
{{"query_type": "...", "complexity": 0.X, "recommended_action": "...", "needs_context": true/false, "reasoning": "..."}}
"""
        
        try:
            response = self.agent_llm.invoke(prompt)
            
            # Parse response (simplified - would use structured output in production)
            import json
            # Extract JSON from response
            content = response.content
            
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                analysis = json.loads(json_str)
            else:
                # Fallback
                analysis = {
                    "query_type": "simple_fact",
                    "complexity": 0.5,
                    "recommended_action": "direct_retrieval",
                    "needs_context": False
                }
            
            logger.info(
                f"Query analysis: type={analysis.get('query_type')}, "
                f"action={analysis.get('recommended_action')}"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                "query_type": "simple_fact",
                "complexity": 0.5,
                "recommended_action": "direct_retrieval",
                "needs_context": False,
                "error": str(e)
            }
    
    def reformulate_query(
        self,
        query: str,
        action: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Reformulate query based on agent's recommended action.
        
        Args:
            query: Original query
            action: Reformulation action to take
            conversation_context: Previous conversation
            
        Returns:
            Dictionary with reformulated queries
        """
        if action == "direct_retrieval":
            return {"queries": [query], "strategy": "direct"}
        
        elif action == "query_decomposition" and self.config.enable_decomposition:
            # Decompose into sub-queries
            prompt = f"""Break down the following complex query into simpler sub-queries that can be answered independently.

Query: {query}

Generate 2-4 focused sub-queries, one per line:"""
            
            response = self.agent_llm.invoke(prompt)
            sub_queries = [
                q.strip().lstrip('0123456789.-) ')
                for q in response.content.split('\n')
                if q.strip()
            ]
            
            return {"queries": sub_queries[:4], "strategy": "decomposition"}
        
        elif action == "query_expansion" and self.config.enable_expansion:
            # Expand with related queries
            prompt = f"""Generate 2-3 alternative phrasings or related queries for:

Query: {query}

Alternative queries:"""
            
            response = self.agent_llm.invoke(prompt)
            expanded_queries = [query] + [
                q.strip().lstrip('0123456789.-) ')
                for q in response.content.split('\n')
                if q.strip()
            ]
            
            return {"queries": expanded_queries[:4], "strategy": "expansion"}
        
        elif action == "multi_turn_context" and self.config.enable_multi_turn:
            # Add conversation context
            if conversation_context:
                context_summary = " ".join([
                    turn.get("user", "") for turn in conversation_context[-2:]
                ])
                contextualized_query = f"{context_summary} {query}"
            else:
                contextualized_query = query
            
            return {"queries": [contextualized_query], "strategy": "contextualized"}
        
        else:
            return {"queries": [query], "strategy": "fallback"}
    
    def execute_retrieval(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[Any]:
        """
        Execute retrieval for multiple queries.
        
        Args:
            queries: List of queries to retrieve for
            k: Number of documents per query
            
        Returns:
            Combined list of retrieved documents
        """
        if not self.retriever:
            raise ValueError("No retriever initialized")
        
        all_docs = []
        seen = set()
        
        for query in queries:
            docs = self.retriever.invoke(query)[:k]
            
            for doc in docs:
                # Deduplicate
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    all_docs.append(doc)
        
        return all_docs
    
    def self_correct(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Self-correct the answer if needed.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context used
            
        Returns:
            Dictionary with corrected answer and metadata
        """
        if not self.config.enable_self_correction:
            return {"answer": answer, "corrected": False}
        
        prompt = f"""Evaluate the following answer for accuracy and completeness.

Query: {query}

Context: {context[:1000]}...

Answer: {answer}

Is the answer:
1. Accurate (based on context)?
2. Complete (addresses all parts)?
3. Clear and well-structured?

If any issues, provide a corrected answer. Otherwise, confirm the answer is good.

Response format:
{{"needs_correction": true/false, "issues": ["..."], "corrected_answer": "..."}}
"""
        
        try:
            response = self.agent_llm.invoke(prompt)
            
            # Parse response
            import json
            content = response.content
            
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                correction = json.loads(json_str)
                
                if correction.get("needs_correction"):
                    return {
                        "answer": correction.get("corrected_answer", answer),
                        "corrected": True,
                        "issues": correction.get("issues", [])
                    }
            
            return {"answer": answer, "corrected": False}
            
        except Exception as e:
            logger.error(f"Error in self-correction: {e}")
            return {"answer": answer, "corrected": False, "error": str(e)}
    
    def retrieve_and_generate(
        self,
        query: str,
        k: int = 5,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        return_reasoning: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agentic RAG pipeline.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            conversation_history: Previous conversation turns
            return_reasoning: Include agent reasoning in response
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with answer and agent reasoning
        """
        if not self.retriever:
            raise ValueError("No retriever initialized. Call index_documents first.")
        
        agent_actions = []
        reasoning_steps = []
        
        # Step 1: Analyze query
        logger.info("Step 1: Analyzing query...")
        analysis = self.analyze_query(query, conversation_history)
        agent_actions.append({"step": "analyze", "result": analysis})
        reasoning_steps.append(f"Analyzed query as: {analysis.get('query_type')}")
        
        # Step 2: Reformulate query
        logger.info("Step 2: Reformulating query...")
        recommended_action = analysis.get("recommended_action", "direct_retrieval")
        reformulation = self.reformulate_query(
            query,
            recommended_action,
            conversation_history
        )
        agent_actions.append({"step": "reformulate", "result": reformulation})
        reasoning_steps.append(
            f"Used {reformulation['strategy']} strategy, "
            f"generated {len(reformulation['queries'])} queries"
        )
        
        # Step 3: Retrieve documents
        logger.info("Step 3: Retrieving documents...")
        docs = self.execute_retrieval(reformulation['queries'], k=k)
        agent_actions.append({"step": "retrieve", "num_docs": len(docs)})
        reasoning_steps.append(f"Retrieved {len(docs)} unique documents")
        
        # Step 4: Generate answer
        logger.info("Step 4: Generating answer...")
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Based on the context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.invoke(prompt).content
        agent_actions.append({"step": "generate", "answer_length": len(answer)})
        reasoning_steps.append("Generated initial answer")
        
        # Step 5: Self-correction (if enabled)
        if self.config.enable_self_correction:
            logger.info("Step 5: Self-correction...")
            correction = self.self_correct(query, answer, context)
            
            if correction['corrected']:
                answer = correction['answer']
                agent_actions.append({"step": "correct", "corrected": True})
                reasoning_steps.append("Applied self-correction")
            else:
                agent_actions.append({"step": "correct", "corrected": False})
                reasoning_steps.append("No correction needed")
        
        # Build result
        result = {
            "answer": answer,
            "documents": docs,
            "num_retrieved": len(docs)
        }
        
        if return_reasoning:
            result.update({
                "agent_actions": agent_actions,
                "reasoning_steps": reasoning_steps,
                "query_analysis": analysis,
                "reformulation": reformulation
            })
        
        # Update conversation history
        if conversation_history is not None:
            self.conversation_history = conversation_history + [{
                "user": query,
                "assistant": answer
            }]
        
        logger.info(f"Completed agentic pipeline in {len(agent_actions)} steps")
        
        return result
