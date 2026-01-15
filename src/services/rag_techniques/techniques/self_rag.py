"""
Self-RAG: Self-Reflective Retrieval-Augmented Generation

This module implements Self-RAG which dynamically decides whether to retrieve,
evaluates relevance of retrieved documents, and assesses the quality of generated
responses through multiple reflection steps.
"""

from typing import List, Optional, Dict, Any, Literal
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class RetrievalDecision(BaseModel):
    """Decision on whether retrieval is needed."""
    needs_retrieval: bool = Field(description="Whether retrieval is necessary")
    reasoning: str = Field(description="Reasoning for the decision")


class RelevanceScore(BaseModel):
    """Relevance score for a document."""
    relevance: Literal["relevant", "irrelevant"] = Field(description="Relevance assessment")
    score: float = Field(description="Relevance score between 0 and 1")


class SupportScore(BaseModel):
    """Support assessment for generated response."""
    support: Literal["fully_supported", "partially_supported", "not_supported"] = Field(
        description="Support level"
    )
    score: float = Field(description="Support score between 0 and 1")


class UtilityScore(BaseModel):
    """Utility assessment for generated response."""
    utility: Literal["high", "medium", "low"] = Field(description="Utility level")
    score: float = Field(description="Utility score between 0 and 1")


class SelfRAG(BaseRAGTechnique):
    """
    Self-RAG with dynamic retrieval and multi-stage evaluation.
    
    Implements a reflective RAG system that:
    1. Decides if retrieval is necessary
    2. Evaluates relevance of retrieved documents
    3. Generates responses using relevant context
    4. Assesses how well responses are supported
    5. Evaluates utility of responses
    
    Attributes:
        embeddings: OpenAI embeddings model
        vectorstore: FAISS vector store
        relevance_threshold: Threshold for document relevance
        support_threshold: Threshold for response support
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        top_k: int = 3,
        relevance_threshold: float = 0.7,
        support_threshold: float = 0.7
    ):
        """
        Initialize Self-RAG.
        
        Args:
            config: Configuration manager instance
            top_k: Number of documents to retrieve
            relevance_threshold: Minimum relevance score for documents
            support_threshold: Minimum support score for responses
        """
        super().__init__(config)
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        self.vectorstore: Optional[FAISS] = None
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.retrieval_decision_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Determine if external information retrieval is needed to answer this query.
Consider if the query can be answered with general knowledge or requires specific facts.

Query: {query}

Does this query need retrieval? Provide your reasoning."""
        )
        
        self.relevance_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""Evaluate if the following document is relevant to the query.
Rate relevance as "relevant" or "irrelevant" and provide a score from 0 to 1.

Query: {query}

Document: {document}

Is this document relevant?"""
        )
        
        self.support_prompt = PromptTemplate(
            input_variables=["context", "response"],
            template="""Evaluate how well the response is supported by the context.
Rate as "fully_supported", "partially_supported", or "not_supported" and provide a score from 0 to 1.

Context: {context}

Response: {response}

How well is the response supported?"""
        )
        
        self.utility_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""Evaluate the utility of this response in answering the query.
Rate as "high", "medium", or "low" and provide a score from 0 to 1.

Query: {query}

Response: {response}

What is the utility of this response?"""
        )
    
    def decide_retrieval(self, query: str) -> RetrievalDecision:
        """
        Decide if retrieval is necessary for the query.
        
        Args:
            query: User query
            
        Returns:
            RetrievalDecision object
        """
        prompt = self.retrieval_decision_prompt.format(query=query)
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Simple heuristic for decision (can be enhanced with structured output)
        needs_retrieval = any(word in response_text.lower() for word in ['yes', 'need', 'require', 'necessary'])
        
        decision = RetrievalDecision(
            needs_retrieval=needs_retrieval,
            reasoning=response_text
        )
        
        self.logger.info(f"Retrieval decision: {needs_retrieval}")
        return decision
    
    def evaluate_relevance(
        self,
        query: str,
        document: Document
    ) -> RelevanceScore:
        """
        Evaluate relevance of a document to the query.
        
        Args:
            query: User query
            document: Document to evaluate
            
        Returns:
            RelevanceScore object
        """
        prompt = self.relevance_prompt.format(
            query=query,
            document=document.page_content[:500]  # Limit length
        )
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response (simplified)
        is_relevant = 'relevant' in response_text.lower() and 'irrelevant' not in response_text.lower()
        
        # Extract score if present (simplified)
        score = 0.8 if is_relevant else 0.3
        
        return RelevanceScore(
            relevance="relevant" if is_relevant else "irrelevant",
            score=score
        )
    
    def evaluate_support(
        self,
        context: str,
        response: str
    ) -> SupportScore:
        """
        Evaluate how well the response is supported by context.
        
        Args:
            context: Retrieved context
            response: Generated response
            
        Returns:
            SupportScore object
        """
        prompt = self.support_prompt.format(
            context=context[:1000],
            response=response
        )
        llm_response = self.llm.invoke(prompt)
        response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Parse response (simplified)
        if 'fully' in response_text.lower():
            support = "fully_supported"
            score = 0.9
        elif 'partially' in response_text.lower():
            support = "partially_supported"
            score = 0.6
        else:
            support = "not_supported"
            score = 0.3
        
        return SupportScore(support=support, score=score)
    
    def evaluate_utility(
        self,
        query: str,
        response: str
    ) -> UtilityScore:
        """
        Evaluate utility of the response.
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            UtilityScore object
        """
        prompt = self.utility_prompt.format(query=query, response=response)
        llm_response = self.llm.invoke(prompt)
        response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Parse response (simplified)
        if 'high' in response_text.lower():
            utility = "high"
            score = 0.9
        elif 'medium' in response_text.lower():
            utility = "medium"
            score = 0.6
        else:
            utility = "low"
            score = 0.3
        
        return UtilityScore(utility=utility, score=score)
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """Create FAISS vector store from documents."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        self.logger.info(f"Created vector store with {len(chunks)} chunks")
        return self.vectorstore
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_reflections: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using Self-RAG with reflection.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_reflections: Whether to return reflection scores
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer and reflection metadata
        """
        result = {
            "query": query,
            "reflections": {}
        }
        
        # Step 1: Decide if retrieval is needed
        retrieval_decision = self.decide_retrieval(query)
        result["reflections"]["retrieval_decision"] = retrieval_decision.dict()
        
        # Step 2: Retrieve and evaluate if needed
        if retrieval_decision.needs_retrieval and self.vectorstore:
            k = top_k or self.top_k
            retrieved_docs = self.vectorstore.similarity_search(query, k=k)
            
            # Evaluate relevance of each document
            relevant_docs = []
            relevance_scores = []
            
            for doc in retrieved_docs:
                relevance = self.evaluate_relevance(query, doc)
                relevance_scores.append(relevance.dict())
                
                if relevance.score >= self.relevance_threshold:
                    relevant_docs.append(doc)
            
            result["reflections"]["relevance_scores"] = relevance_scores
            result["reflections"]["num_relevant_docs"] = len(relevant_docs)
            
            # Generate response with relevant context
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                response = self._generate_with_context(query, context)
                
                # Evaluate support
                support = self.evaluate_support(context, response)
                result["reflections"]["support"] = support.dict()
            else:
                # Generate without retrieval
                response = self._generate_without_context(query)
                result["reflections"]["support"] = {"note": "No relevant documents found"}
        else:
            # Generate without retrieval
            response = self._generate_without_context(query)
            result["reflections"]["retrieval"] = "Not needed"
        
        # Step 3: Evaluate utility
        utility = self.evaluate_utility(query, response)
        result["reflections"]["utility"] = utility.dict()
        
        result["answer"] = response
        
        if not return_reflections:
            # Return simplified result
            return {
                "answer": response,
                "quality_score": utility.score
            }
        
        return result
    
    def _generate_with_context(self, query: str, context: str) -> str:
        """Generate response using retrieved context."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _generate_without_context(self, query: str) -> str:
        """Generate response without retrieval."""
        prompt = f"""Answer the following question based on your general knowledge.

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
