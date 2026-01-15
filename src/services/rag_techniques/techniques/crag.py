"""
CRAG: Corrective Retrieval-Augmented Generation

This module implements CRAG which evaluates retrieved documents and dynamically
corrects by performing web search when local knowledge is insufficient or irrelevant.
"""

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class CorrectiveRAG(BaseRAGTechnique):
    """
    Corrective RAG with dynamic knowledge acquisition.
    
    Evaluates retrieved documents and adapts strategy:
    - High relevance (>0.7): Use retrieved document as-is
    - Low relevance (<0.3): Perform web search with rewritten query
    - Ambiguous (0.3-0.7): Combine retrieved docs with web search
    
    Attributes:
        embeddings: OpenAI embeddings model
        vectorstore: FAISS vector store
        high_relevance_threshold: Threshold for high relevance
        low_relevance_threshold: Threshold for low relevance
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        top_k: int = 3,
        high_relevance_threshold: float = 0.7,
        low_relevance_threshold: float = 0.3,
        enable_web_search: bool = False
    ):
        """
        Initialize Corrective RAG.
        
        Args:
            config: Configuration manager instance
            top_k: Number of documents to retrieve
            high_relevance_threshold: Threshold for high relevance
            low_relevance_threshold: Threshold for low relevance
            enable_web_search: Whether to enable web search (requires API)
        """
        super().__init__(config)
        self.top_k = top_k
        self.high_relevance_threshold = high_relevance_threshold
        self.low_relevance_threshold = low_relevance_threshold
        self.enable_web_search = enable_web_search
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        self.vectorstore: Optional[FAISS] = None
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.relevance_eval_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""Evaluate the relevance of the following document to the query.
Provide a relevance score from 0 to 1, where:
- 0.0-0.3: Not relevant (requires web search)
- 0.3-0.7: Partially relevant (needs supplementation)
- 0.7-1.0: Highly relevant (sufficient information)

Query: {query}

Document: {document}

Provide only the relevance score as a number between 0 and 1."""
        )
        
        self.query_rewrite_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Rewrite the following query to be more effective for web search.
Make it more specific and search-engine friendly.

Original query: {query}

Rewritten query:"""
        )
        
        self.knowledge_refinement_prompt = PromptTemplate(
            input_variables=["query", "content"],
            template="""Extract the key information relevant to the query from the following content.
Provide a concise summary of the most important points.

Query: {query}

Content: {content}

Key information:"""
        )
    
    def evaluate_relevance(
        self,
        query: str,
        document: Document
    ) -> float:
        """
        Evaluate relevance of a document to the query.
        
        Args:
            query: User query
            document: Document to evaluate
            
        Returns:
            Relevance score between 0 and 1
        """
        prompt = self.relevance_eval_prompt.format(
            query=query,
            document=document.page_content[:1000]  # Limit length
        )
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract score from response
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response_text)
            if numbers:
                score = float(numbers[0])
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            else:
                # Fallback: simple keyword matching
                if any(word in response_text.lower() for word in ['high', 'relevant', 'good']):
                    score = 0.8
                elif any(word in response_text.lower() for word in ['partial', 'some', 'moderate']):
                    score = 0.5
                else:
                    score = 0.2
        except Exception as e:
            self.logger.warning(f"Error parsing relevance score: {e}")
            score = 0.5
        
        return score
    
    def rewrite_query_for_web(self, query: str) -> str:
        """
        Rewrite query for web search.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query optimized for search
        """
        prompt = self.query_rewrite_prompt.format(query=query)
        response = self.llm.invoke(prompt)
        rewritten = response.content if hasattr(response, 'content') else str(response)
        
        self.logger.info(f"Rewritten query: {rewritten}")
        return rewritten.strip()
    
    def refine_knowledge(
        self,
        query: str,
        content: str
    ) -> str:
        """
        Refine and extract key knowledge from content.
        
        Args:
            query: User query
            content: Content to refine
            
        Returns:
            Refined knowledge
        """
        prompt = self.knowledge_refinement_prompt.format(
            query=query,
            content=content[:2000]  # Limit length
        )
        
        response = self.llm.invoke(prompt)
        refined = response.content if hasattr(response, 'content') else str(response)
        
        return refined
    
    def web_search(self, query: str, num_results: int = 3) -> List[str]:
        """
        Perform web search (placeholder - requires search API).
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search result snippets
        """
        if not self.enable_web_search:
            return [f"Web search simulated for: {query}"]
        
        # Placeholder for actual web search implementation
        # In production, integrate with search APIs like Tavily, Google, etc.
        self.logger.warning("Web search not implemented - using placeholder")
        return [
            f"Placeholder web search result 1 for: {query}",
            f"Placeholder web search result 2 for: {query}"
        ]
    
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
        return_metadata: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using Corrective RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_metadata: Whether to return correction metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k
        
        # Step 1: Initial retrieval
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        
        # Step 2: Evaluate relevance
        relevance_scores = [
            self.evaluate_relevance(query, doc)
            for doc in retrieved_docs
        ]
        
        max_relevance = max(relevance_scores) if relevance_scores else 0.0
        best_doc_idx = relevance_scores.index(max_relevance) if relevance_scores else 0
        
        self.logger.info(f"Max relevance score: {max_relevance:.2f}")
        
        # Step 3: Corrective strategy based on relevance
        strategy = "none"
        knowledge_sources = []
        
        if max_relevance >= self.high_relevance_threshold:
            # High relevance: use retrieved document
            strategy = "direct_use"
            knowledge = retrieved_docs[best_doc_idx].page_content
            knowledge_sources.append("local_retrieval")
            self.logger.info("Using direct retrieval (high relevance)")
            
        elif max_relevance < self.low_relevance_threshold:
            # Low relevance: perform web search
            strategy = "web_search"
            rewritten_query = self.rewrite_query_for_web(query)
            web_results = self.web_search(rewritten_query)
            
            # Refine web results
            web_knowledge = "\n\n".join(web_results)
            knowledge = self.refine_knowledge(query, web_knowledge)
            knowledge_sources.append("web_search")
            self.logger.info("Using web search (low relevance)")
            
        else:
            # Ambiguous: combine retrieval with web search
            strategy = "hybrid"
            local_knowledge = retrieved_docs[best_doc_idx].page_content
            
            rewritten_query = self.rewrite_query_for_web(query)
            web_results = self.web_search(rewritten_query)
            web_knowledge = "\n\n".join(web_results)
            refined_web = self.refine_knowledge(query, web_knowledge)
            
            knowledge = f"{local_knowledge}\n\nAdditional information:\n{refined_web}"
            knowledge_sources.extend(["local_retrieval", "web_search"])
            self.logger.info("Using hybrid approach (ambiguous relevance)")
        
        # Step 4: Generate answer
        answer = self._generate_answer(query, knowledge)
        
        result = {
            "answer": answer,
            "strategy": strategy,
            "max_relevance_score": max_relevance
        }
        
        if return_metadata:
            result["metadata"] = {
                "retrieved_docs": len(retrieved_docs),
                "relevance_scores": relevance_scores,
                "knowledge_sources": knowledge_sources,
                "correction_applied": strategy != "direct_use"
            }
        
        return result
    
    def _generate_answer(self, query: str, knowledge: str) -> str:
        """Generate answer using corrected knowledge."""
        prompt = f"""Based on the following knowledge (which may be from retrieval, web search, or both), answer the question.

Knowledge:
{knowledge}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about correction behavior."""
        return {
            "high_relevance_threshold": self.high_relevance_threshold,
            "low_relevance_threshold": self.low_relevance_threshold,
            "web_search_enabled": self.enable_web_search,
            "strategies": {
                "direct_use": f"relevance >= {self.high_relevance_threshold}",
                "web_search": f"relevance < {self.low_relevance_threshold}",
                "hybrid": f"{self.low_relevance_threshold} <= relevance < {self.high_relevance_threshold}"
            }
        }
