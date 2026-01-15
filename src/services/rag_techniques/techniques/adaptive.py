"""
Adaptive Retrieval - Query-based retrieval strategy selection

This module implements adaptive retrieval that classifies queries and
applies different retrieval strategies based on query type.
"""

from typing import List, Dict, Any, Optional
from enum import Enum

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.base import BaseRAG
from ..core.config import RAGConfig


class QueryCategory(str, Enum):
    """Query category types."""
    FACTUAL = "Factual"
    ANALYTICAL = "Analytical"
    OPINION = "Opinion"
    CONTEXTUAL = "Contextual"


class QueryClassification(BaseModel):
    """Model for query classification output."""
    category: str = Field(description="Query category: Factual, Analytical, Opinion, or Contextual")
    confidence: float = Field(description="Confidence score 0-1", default=0.0)


class RelevanceScore(BaseModel):
    """Model for document relevance scoring."""
    score: float = Field(description="Relevance score 1-10")
    reasoning: str = Field(description="Brief explanation")


class AdaptiveRetrievalRAG(BaseRAG):
    """
    RAG with adaptive retrieval strategies.
    
    Classifies queries into categories and applies appropriate
    retrieval strategies for each type:
    - Factual: Enhanced query + reranking
    - Analytical: Multi-query decomposition
    - Opinion: Diverse perspective retrieval
    - Contextual: Context-aware retrieval
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        content: Optional[str] = None,
        config: Optional[RAGConfig] = None
    ):
        super().__init__(pdf_path, content, config)
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens
        )
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query into one of four categories.
        
        Args:
            query: The user query
            
        Returns:
            QueryClassification with category and confidence
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Classify the following query into ONE of these categories:
            
            1. Factual: Questions seeking specific facts, data, or definitions
               Examples: "What is X?", "When did Y happen?", "Who invented Z?"
            
            2. Analytical: Questions requiring analysis, comparison, or reasoning
               Examples: "Why does X happen?", "How does Y affect Z?", "Compare A and B"
            
            3. Opinion: Questions seeking perspectives, recommendations, or subjective views
               Examples: "What is the best way to...?", "Should we...?", "Is X better than Y?"
            
            4. Contextual: Questions requiring understanding of broader context
               Examples: "Explain the implications of...", "What is the significance of...?"
            
            Query: {query}
            
            Classify this query and provide your confidence level (0-1).
            """
        )
        
        chain = prompt | self.llm.with_structured_output(QueryClassification)
        return chain.invoke({"query": query})
    
    def factual_retrieval(self, query: str, k: int) -> List[Any]:
        """
        Retrieval strategy for factual queries.
        Enhanced query + document reranking.
        """
        # Enhance query
        enhance_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        chain = enhance_prompt | self.llm
        enhanced_query = chain.invoke({"query": query}).content
        
        # Retrieve more documents
        docs = self.vectorstore.similarity_search(enhanced_query, k=k*2)
        
        # Rerank using LLM
        rank_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="""
            Rate the relevance of this document to the query on a scale of 1-10.
            
            Query: {query}
            Document: {doc}
            
            Provide score and brief reasoning.
            """
        )
        rank_chain = rank_prompt | self.llm.with_structured_output(RelevanceScore)
        
        scored_docs = []
        for doc in docs:
            try:
                result = rank_chain.invoke({
                    "query": enhanced_query,
                    "doc": doc.page_content[:1000]
                })
                scored_docs.append((doc, result.score))
            except:
                scored_docs.append((doc, 5.0))
        
        # Sort and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]
    
    def analytical_retrieval(self, query: str, k: int) -> List[Any]:
        """
        Retrieval strategy for analytical queries.
        Decompose into sub-questions and retrieve for each.
        """
        # Decompose query
        decompose_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Break down this analytical query into 2-3 simpler sub-questions.
            
            Query: {query}
            
            Return a list of sub-questions.
            """
        )
        chain = decompose_prompt | self.llm
        result = chain.invoke({"query": query}).content
        
        # Extract sub-questions (simple parsing)
        sub_queries = [q.strip() for q in result.split('\n') if q.strip() and not q.strip().startswith('#')]
        sub_queries = [q.lstrip('0123456789.-) ') for q in sub_queries[:3]]
        
        # Retrieve for each sub-query
        all_docs = []
        seen = set()
        
        for sub_q in sub_queries:
            docs = self.vectorstore.similarity_search(sub_q, k=k)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)
        
        return all_docs[:k*2]  # Return more docs for analytical queries
    
    def opinion_retrieval(self, query: str, k: int) -> List[Any]:
        """
        Retrieval strategy for opinion queries.
        Retrieve diverse perspectives using MMR.
        """
        # Use MMR for diversity
        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*3,
            lambda_mult=0.5  # Balance relevance and diversity
        )
        return docs
    
    def contextual_retrieval(self, query: str, k: int) -> List[Any]:
        """
        Retrieval strategy for contextual queries.
        Retrieve broader context with larger chunks.
        """
        # Standard retrieval with more results
        docs = self.vectorstore.similarity_search(query, k=k*2)
        return docs
    
    def query(
        self,
        query_text: str,
        force_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query with adaptive retrieval strategy.
        
        Args:
            query_text: The search query
            force_strategy: Optional - force a specific strategy
                          ('factual', 'analytical', 'opinion', 'contextual')
            
        Returns:
            Dictionary with answer and adaptive retrieval details
        """
        k = self.config.n_retrieved
        
        # Classify query
        if force_strategy:
            classification = QueryClassification(
                category=force_strategy.capitalize(),
                confidence=1.0
            )
        else:
            classification = self.classify_query(query_text)
        
        category = classification.category.lower()
        
        # Apply appropriate strategy
        if category == "factual":
            docs = self.factual_retrieval(query_text, k)
            strategy_info = "Enhanced query with LLM reranking"
        elif category == "analytical":
            docs = self.analytical_retrieval(query_text, k)
            strategy_info = "Query decomposition with multi-retrieval"
        elif category == "opinion":
            docs = self.opinion_retrieval(query_text, k)
            strategy_info = "MMR for diverse perspectives"
        elif category == "contextual":
            docs = self.contextual_retrieval(query_text, k)
            strategy_info = "Broader context retrieval"
        else:
            # Fallback to standard retrieval
            docs = self.retriever.get_relevant_documents(query_text)
            strategy_info = "Standard retrieval (fallback)"
        
        context = [doc.page_content for doc in docs]
        
        # Generate answer
        answer_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Answer the following question based on the provided context.
            
            Question: {query}
            
            Context:
            {context}
            
            Answer:
            """
        )
        
        chain = answer_prompt | self.llm
        result = chain.invoke({
            "query": query_text,
            "context": "\n\n".join(context)
        })
        
        return {
            "query": query_text,
            "answer": result.content,
            "query_category": classification.category,
            "confidence": classification.confidence,
            "strategy": strategy_info,
            "context": context,
            "num_docs": len(docs)
        }
