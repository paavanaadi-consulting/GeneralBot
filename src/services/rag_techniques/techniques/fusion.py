"""
Fusion Retrieval Techniques

This module implements fusion retrieval that combines multiple
retrieval methods for better results.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from ..core.base import BaseRAG
from ..core.config import RAGConfig


class FusionRAG(BaseRAG):
    """
    RAG with fusion retrieval.
    
    Combines multiple retrieval strategies:
    1. Semantic search (embeddings)
    2. Keyword search (BM25)
    3. Reciprocal Rank Fusion (RRF) for combining results
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
        
        # Initialize BM25 for keyword search
        self._init_bm25()
    
    def _init_bm25(self):
        """Initialize BM25 index."""
        # Get all documents from vector store
        all_docs = self.vectorstore.similarity_search("", k=1000)  # Get many docs
        
        # Tokenize documents for BM25
        self.bm25_docs = all_docs
        tokenized_docs = [doc.page_content.split() for doc in all_docs]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def semantic_search(self, query: str, k: int) -> List[tuple]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_and_scores
    
    def keyword_search(self, query: str, k: int) -> List[tuple]:
        """
        Perform keyword search using BM25.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        query_tokens = query.split()
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        return [(self.bm25_docs[i], scores[i]) for i in top_k_indices]
    
    def reciprocal_rank_fusion(
        self,
        retrieval_results: List[List[tuple]],
        k: int = 60
    ) -> List[tuple]:
        """
        Combine multiple retrieval results using Reciprocal Rank Fusion.
        
        Args:
            retrieval_results: List of retrieval results, each as list of (doc, score)
            k: Constant for RRF formula (default: 60)
            
        Returns:
            Fused and reranked list of (document, score) tuples
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}  # Map doc content to doc object
        
        for results in retrieval_results:
            for rank, (doc, _) in enumerate(results, start=1):
                doc_key = doc.page_content  # Use content as key
                rrf_scores[doc_key] += 1.0 / (k + rank)
                doc_map[doc_key] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_map[doc_key], score) for doc_key, score in sorted_docs]
    
    def hybrid_search(
        self,
        query: str,
        k: int,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5
    ) -> List[tuple]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            List of (document, score) tuples
        """
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        semantic_weight /= total_weight
        keyword_weight /= total_weight
        
        # Get results from both methods
        semantic_results = self.semantic_search(query, k=k*2)
        keyword_results = self.keyword_search(query, k=k*2)
        
        # Normalize scores to 0-1 range
        def normalize_scores(results):
            if not results:
                return results
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            if score_range == 0:
                return [(doc, 0.5) for doc, _ in results]
            return [(doc, (score - min_score) / score_range) for doc, score in results]
        
        semantic_results = normalize_scores(semantic_results)
        keyword_results = normalize_scores(keyword_results)
        
        # Combine scores
        combined_scores = defaultdict(float)
        doc_map = {}
        
        for doc, score in semantic_results:
            doc_key = doc.page_content
            combined_scores[doc_key] += semantic_weight * score
            doc_map[doc_key] = doc
        
        for doc, score in keyword_results:
            doc_key = doc.page_content
            combined_scores[doc_key] += keyword_weight * score
            doc_map[doc_key] = doc
        
        # Sort and return top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[doc_key], score) for doc_key, score in sorted_docs[:k]]
    
    def query(
        self,
        query_text: str,
        method: str = "rrf",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using fusion retrieval.
        
        Args:
            query_text: Search query
            method: Fusion method - 'rrf' (reciprocal rank fusion) or 'hybrid'
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with answer and retrieval details
        """
        k = self.config.n_retrieved
        
        if method == "rrf":
            # Retrieve using both methods
            semantic_results = self.semantic_search(query_text, k=k*2)
            keyword_results = self.keyword_search(query_text, k=k*2)
            
            # Combine using RRF
            fused_results = self.reciprocal_rank_fusion(
                [semantic_results, keyword_results],
                k=kwargs.get("rrf_k", 60)
            )
            
            # Take top-k
            final_results = fused_results[:k]
            
            retrieval_info = {
                "method": "Reciprocal Rank Fusion",
                "num_semantic": len(semantic_results),
                "num_keyword": len(keyword_results),
                "num_fused": len(fused_results)
            }
            
        elif method == "hybrid":
            # Weighted hybrid search
            semantic_weight = kwargs.get("semantic_weight", 0.5)
            keyword_weight = kwargs.get("keyword_weight", 0.5)
            
            final_results = self.hybrid_search(
                query_text,
                k=k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            
            retrieval_info = {
                "method": "Hybrid Search",
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight
            }
            
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Extract documents and scores
        docs = [doc for doc, score in final_results]
        scores = [score for doc, score in final_results]
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
            "fusion_method": method,
            "retrieval_info": retrieval_info,
            "context": context,
            "scores": scores,
            "num_docs": len(docs)
        }
