"""
Reliable RAG: Document Grading and Validation

This module implements a reliability layer for RAG systems that validates
retrieved documents before using them for generation. It grades documents
for relevance and filters out low-quality or irrelevant content.

Key Features:
- Document relevance grading using LLM
- Binary classification (relevant/not relevant)
- Filtering of irrelevant documents
- Confidence scoring
- Quality thresholds
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..core.base import BaseRAGTechnique
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Brief explanation of the relevance decision"
    )


@dataclass
class ReliableRAGConfig:
    """Configuration for Reliable RAG."""
    
    # Grading parameters
    relevance_threshold: float = 0.7
    require_unanimous: bool = False  # All docs must be relevant
    min_relevant_docs: int = 1  # Minimum relevant docs needed
    
    # LLM settings for grading
    grading_model: str = "gpt-3.5-turbo"
    grading_temperature: float = 0.0
    
    # Fallback strategy
    use_web_search: bool = False  # If no relevant docs found
    fallback_to_llm: bool = True  # Use LLM without context if needed


class ReliableRAG(BaseRAGTechnique):
    """
    Implements Reliable RAG with document grading and validation.
    
    This technique adds a reliability layer that grades retrieved documents
    for relevance before using them for generation. Only relevant documents
    are passed to the generation stage.
    
    Example:
        >>> config = ReliableRAGConfig(min_relevant_docs=2)
        >>> reliable_rag = ReliableRAG(config=config)
        >>> 
        >>> # Index documents
        >>> reliable_rag.index_documents(documents)
        >>> 
        >>> # Query with reliability checking
        >>> result = reliable_rag.retrieve_and_generate(
        ...     query="What are design patterns?",
        ...     k=5
        ... )
        >>> 
        >>> # Check reliability metrics
        >>> print(f"Relevant docs: {result['num_relevant']}/{result['num_retrieved']}")
        >>> print(f"Reliability score: {result['reliability_score']:.2f}")
    """
    
    def __init__(
        self,
        config: Optional[ReliableRAGConfig] = None,
        **kwargs
    ):
        """Initialize Reliable RAG.
        
        Args:
            config: Configuration for reliable RAG
            **kwargs: Additional arguments passed to BaseRAGTechnique
        """
        super().__init__(**kwargs)
        self.config = config or ReliableRAGConfig()
        
        # Initialize grading LLM
        try:
            from langchain_openai import ChatOpenAI
            self.grading_llm = ChatOpenAI(
                model=self.config.grading_model,
                temperature=self.config.grading_temperature
            )
        except ImportError:
            logger.warning("ChatOpenAI not available, grading may not work")
            self.grading_llm = None
        
        logger.info(
            f"Initialized ReliableRAG with threshold={self.config.relevance_threshold}"
        )
    
    def grade_document(
        self,
        document: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Grade a single document for relevance to the query.
        
        Args:
            document: Document text to grade
            query: User query
            
        Returns:
            Dictionary with grading results:
            - is_relevant: bool
            - confidence: float
            - explanation: str
        """
        if not self.grading_llm:
            logger.warning("No grading LLM available, assuming relevant")
            return {
                "is_relevant": True,
                "confidence": 0.5,
                "explanation": "No grading model available"
            }
        
        # Create grading prompt
        grading_prompt = f"""You are a document relevance grader. Assess whether the following document is relevant to the user question.

Question: {query}

Document: {document[:1000]}...

Respond with 'yes' if the document contains information that could help answer the question, or 'no' if it does not.
Provide a brief explanation of your decision."""
        
        try:
            # Use structured output if available
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a document relevance grader."),
                ("human", grading_prompt)
            ])
            
            # Create structured output chain
            structured_llm = self.grading_llm.with_structured_output(GradeDocuments)
            chain = prompt | structured_llm
            
            # Get grading
            result = chain.invoke({"query": query, "document": document})
            
            is_relevant = result.binary_score.lower() == "yes"
            
            return {
                "is_relevant": is_relevant,
                "confidence": 0.9 if is_relevant else 0.1,
                "explanation": result.explanation or "No explanation provided"
            }
            
        except Exception as e:
            logger.error(f"Error grading document: {e}")
            # Fallback to simple heuristic
            return self._simple_relevance_check(document, query)
    
    def _simple_relevance_check(
        self,
        document: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Simple fallback relevance check using keyword matching.
        
        Args:
            document: Document text
            query: User query
            
        Returns:
            Grading result dictionary
        """
        # Extract keywords from query
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Calculate overlap
        overlap = len(query_words & doc_words)
        score = overlap / len(query_words) if query_words else 0
        
        is_relevant = score >= self.config.relevance_threshold
        
        return {
            "is_relevant": is_relevant,
            "confidence": score,
            "explanation": f"Keyword overlap: {overlap}/{len(query_words)}"
        }
    
    def grade_documents(
        self,
        documents: List[str],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Grade multiple documents for relevance.
        
        Args:
            documents: List of document texts
            query: User query
            
        Returns:
            List of grading results for each document
        """
        results = []
        
        for i, doc in enumerate(documents):
            logger.debug(f"Grading document {i+1}/{len(documents)}")
            result = self.grade_document(doc, query)
            results.append(result)
        
        return results
    
    def filter_relevant_documents(
        self,
        documents: List[str],
        gradings: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Filter documents based on relevance gradings.
        
        Args:
            documents: Original documents
            gradings: Grading results
            
        Returns:
            List of relevant documents
        """
        relevant_docs = [
            doc for doc, grade in zip(documents, gradings)
            if grade["is_relevant"]
        ]
        
        logger.info(
            f"Filtered {len(relevant_docs)}/{len(documents)} relevant documents"
        )
        
        return relevant_docs
    
    def retrieve_and_generate(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve documents, grade them, and generate response.
        
        Args:
            query: User query
            k: Number of documents to retrieve initially
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - answer: Generated response
            - relevant_documents: Filtered relevant documents
            - gradings: Grading results for each document
            - num_relevant: Number of relevant documents
            - num_retrieved: Total documents retrieved
            - reliability_score: Overall reliability metric
        """
        if not self.retriever:
            raise ValueError("No retriever initialized. Call index_documents first.")
        
        # Retrieve documents
        logger.info(f"Retrieving {k} documents for query: {query[:100]}...")
        retrieved_docs = self.retriever.invoke(query)[:k]
        
        # Extract document texts
        doc_texts = [doc.page_content for doc in retrieved_docs]
        
        # Grade documents
        logger.info("Grading retrieved documents...")
        gradings = self.grade_documents(doc_texts, query)
        
        # Filter relevant documents
        relevant_docs = self.filter_relevant_documents(doc_texts, gradings)
        
        # Calculate reliability metrics
        num_relevant = len(relevant_docs)
        num_retrieved = len(doc_texts)
        reliability_score = num_relevant / num_retrieved if num_retrieved > 0 else 0
        
        logger.info(
            f"Reliability: {num_relevant}/{num_retrieved} relevant "
            f"(score: {reliability_score:.2f})"
        )
        
        # Check if we have enough relevant documents
        if num_relevant < self.config.min_relevant_docs:
            logger.warning(
                f"Only {num_relevant} relevant documents found, "
                f"minimum is {self.config.min_relevant_docs}"
            )
            
            if self.config.fallback_to_llm:
                logger.info("Falling back to LLM without context")
                # Generate without context
                answer = self._generate_without_context(query)
                
                return {
                    "answer": answer,
                    "relevant_documents": relevant_docs,
                    "gradings": gradings,
                    "num_relevant": num_relevant,
                    "num_retrieved": num_retrieved,
                    "reliability_score": reliability_score,
                    "fallback_used": True,
                    "warning": "Insufficient relevant documents, used LLM fallback"
                }
        
        # Generate answer with relevant documents
        context = "\n\n".join(relevant_docs)
        
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.invoke(prompt).content
        
        return {
            "answer": answer,
            "relevant_documents": relevant_docs,
            "gradings": gradings,
            "num_relevant": num_relevant,
            "num_retrieved": num_retrieved,
            "reliability_score": reliability_score,
            "fallback_used": False
        }
    
    def _generate_without_context(self, query: str) -> str:
        """Generate answer without document context (fallback)."""
        prompt = f"""Please answer the following question to the best of your ability.
Note: No relevant context documents were found.

Question: {query}

Answer:"""
        
        return self.llm.invoke(prompt).content
