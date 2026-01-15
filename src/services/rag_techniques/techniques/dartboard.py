"""
Dartboard RAG: Balanced Relevance and Diversity Retrieval

This module implements the Dartboard RAG technique which optimizes for both
relevance and diversity in retrieved documents, preventing redundancy while
maintaining high relevance.

Based on: "Better RAG using Relevant Information Gain" (arxiv.org/abs/2407.12101)

Key Features:
- Combines relevance and diversity scoring
- Prevents redundant document retrieval
- Weighted balance control
- Works with dense, sparse, and hybrid retrieval
- Compatible with cross-encoders
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.special import logsumexp

from ..core.base import BaseRAGTechnique
from ..core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DartboardConfig:
    """Configuration for Dartboard RAG."""
    
    # Scoring weights
    relevance_weight: float = 0.7  # Weight for relevance score
    diversity_weight: float = 0.3  # Weight for diversity score
    
    # Retrieval parameters
    initial_k: int = 20  # Initial candidates to retrieve
    final_k: int = 5  # Final documents to return
    
    # Diversity parameters
    min_diversity_score: float = 0.1  # Minimum diversity to consider
    temperature: float = 1.0  # Temperature for diversity calculation


class DartboardRAG(BaseRAGTechnique):
    """
    Implements Dartboard RAG with balanced relevance-diversity retrieval.
    
    This technique retrieves documents by optimizing a combination of
    relevance (how well documents match the query) and diversity (how
    different documents are from each other). This prevents retrieving
    multiple similar documents while maintaining high relevance.
    
    The algorithm:
    1. Retrieve initial candidate documents (top-N by relevance)
    2. Iteratively select documents by:
       - Scoring relevance to query
       - Penalizing similarity to already-selected documents
       - Combining scores with configurable weights
    3. Return top-k diverse yet relevant documents
    
    Example:
        >>> config = DartboardConfig(
        ...     relevance_weight=0.6,
        ...     diversity_weight=0.4,
        ...     final_k=5
        ... )
        >>> dartboard = DartboardRAG(config=config)
        >>> 
        >>> # Index documents
        >>> dartboard.index_documents(documents)
        >>> 
        >>> # Retrieve with diversity
        >>> result = dartboard.retrieve_with_diversity(
        ...     query="What is machine learning?",
        ...     k=5
        ... )
        >>> 
        >>> # Check diversity metrics
        >>> print(f"Diversity score: {result['diversity_score']:.3f}")
        >>> print(f"Relevance score: {result['relevance_score']:.3f}")
    """
    
    def __init__(
        self,
        config: Optional[DartboardConfig] = None,
        **kwargs
    ):
        """Initialize Dartboard RAG.
        
        Args:
            config: Configuration for Dartboard RAG
            **kwargs: Additional arguments passed to BaseRAGTechnique
        """
        super().__init__(**kwargs)
        self.config = config or DartboardConfig()
        
        # Validate weights
        total_weight = self.config.relevance_weight + self.config.diversity_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(
                f"Weights sum to {total_weight:.3f}, normalizing to 1.0"
            )
            self.config.relevance_weight /= total_weight
            self.config.diversity_weight /= total_weight
        
        logger.info(
            f"Initialized DartboardRAG with relevance_weight={self.config.relevance_weight:.2f}, "
            f"diversity_weight={self.config.diversity_weight:.2f}"
        )
    
    def compute_similarity_matrix(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for embeddings.
        
        Args:
            embeddings: Array of shape (n_docs, embedding_dim)
            
        Returns:
            Similarity matrix of shape (n_docs, n_docs)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)
        
        return similarity
    
    def compute_diversity_penalty(
        self,
        candidate_idx: int,
        selected_indices: List[int],
        similarity_matrix: np.ndarray
    ) -> float:
        """
        Compute diversity penalty for a candidate document.
        
        The penalty is based on similarity to already-selected documents.
        Higher penalty means the candidate is too similar to what we have.
        
        Args:
            candidate_idx: Index of candidate document
            selected_indices: Indices of already-selected documents
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            Diversity penalty (lower is more diverse)
        """
        if not selected_indices:
            return 0.0
        
        # Get similarities to selected documents
        similarities = similarity_matrix[candidate_idx, selected_indices]
        
        # Use max similarity as penalty (most similar doc determines penalty)
        # Could also use mean or logsumexp for different behaviors
        penalty = np.max(similarities)
        
        return penalty
    
    def compute_information_gain(
        self,
        candidate_idx: int,
        selected_indices: List[int],
        query_similarities: np.ndarray,
        doc_similarities: np.ndarray
    ) -> float:
        """
        Compute information gain of adding a candidate document.
        
        This is the core scoring function that balances relevance and diversity.
        
        Args:
            candidate_idx: Index of candidate document
            selected_indices: Indices of already-selected documents
            query_similarities: Similarities to query for all documents
            doc_similarities: Pairwise document similarity matrix
            
        Returns:
            Information gain score (higher is better)
        """
        # Relevance score (similarity to query)
        relevance = query_similarities[candidate_idx]
        
        # Diversity score (dissimilarity to selected documents)
        if selected_indices:
            diversity_penalty = self.compute_diversity_penalty(
                candidate_idx,
                selected_indices,
                doc_similarities
            )
            # Convert similarity penalty to diversity score (1 - similarity)
            diversity = 1.0 - diversity_penalty
        else:
            diversity = 1.0  # First document has maximum diversity
        
        # Combine scores with weights
        score = (
            self.config.relevance_weight * relevance +
            self.config.diversity_weight * diversity
        )
        
        return score
    
    def select_diverse_documents(
        self,
        query_similarities: np.ndarray,
        doc_embeddings: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Select k documents that optimize relevance and diversity.
        
        Uses greedy selection: iteratively pick the document with highest
        combined relevance-diversity score.
        
        Args:
            query_similarities: Similarity scores to query
            doc_embeddings: Document embeddings
            k: Number of documents to select
            
        Returns:
            List of selected document indices
        """
        n_docs = len(query_similarities)
        k = min(k, n_docs)
        
        # Compute pairwise document similarities
        doc_similarities = self.compute_similarity_matrix(doc_embeddings)
        
        selected_indices = []
        available_indices = list(range(n_docs))
        
        logger.debug(f"Selecting {k} diverse documents from {n_docs} candidates")
        
        # Greedy selection
        for iteration in range(k):
            best_score = float('-inf')
            best_idx = None
            
            # Evaluate each available document
            for idx in available_indices:
                score = self.compute_information_gain(
                    idx,
                    selected_indices,
                    query_similarities,
                    doc_similarities
                )
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            # Add best document to selection
            if best_idx is not None:
                selected_indices.append(best_idx)
                available_indices.remove(best_idx)
                
                logger.debug(
                    f"Iteration {iteration+1}: Selected doc {best_idx} "
                    f"(score: {best_score:.3f})"
                )
        
        return selected_indices
    
    def retrieve_with_diversity(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve documents with diversity optimization.
        
        Args:
            query: User query
            k: Number of final documents to return
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - documents: Selected diverse documents
            - scores: Combined relevance-diversity scores
            - relevance_scores: Pure relevance scores
            - diversity_score: Average diversity metric
            - selected_indices: Indices of selected documents
        """
        if not self.retriever:
            raise ValueError("No retriever initialized. Call index_documents first.")
        
        k = k or self.config.final_k
        initial_k = max(k * 4, self.config.initial_k)
        
        logger.info(
            f"Retrieving {initial_k} candidates, selecting {k} diverse documents"
        )
        
        # Retrieve initial candidates
        candidates = self.retriever.invoke(query)[:initial_k]
        
        if not candidates:
            logger.warning("No candidates retrieved")
            return {
                "documents": [],
                "scores": [],
                "relevance_scores": [],
                "diversity_score": 0.0,
                "selected_indices": []
            }
        
        # Get embeddings for candidates and query
        candidate_texts = [doc.page_content for doc in candidates]
        
        try:
            # Get document embeddings
            doc_embeddings = np.array(
                self.embeddings.embed_documents(candidate_texts)
            )
            
            # Get query embedding
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            # Compute query-document similarities
            query_norms = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1)
            
            query_similarities = np.dot(
                doc_embeddings,
                query_embedding
            ) / (doc_norms * query_norms + 1e-10)
            
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            # Fallback: return top-k by initial retrieval order
            return {
                "documents": candidates[:k],
                "scores": [1.0] * min(k, len(candidates)),
                "relevance_scores": [1.0] * min(k, len(candidates)),
                "diversity_score": 0.0,
                "selected_indices": list(range(min(k, len(candidates)))),
                "error": str(e)
            }
        
        # Select diverse documents
        selected_indices = self.select_diverse_documents(
            query_similarities,
            doc_embeddings,
            k
        )
        
        # Get selected documents and scores
        selected_docs = [candidates[i] for i in selected_indices]
        relevance_scores = [float(query_similarities[i]) for i in selected_indices]
        
        # Calculate average diversity
        if len(selected_indices) > 1:
            doc_similarities = self.compute_similarity_matrix(doc_embeddings)
            selected_similarities = []
            for i in range(len(selected_indices)):
                for j in range(i + 1, len(selected_indices)):
                    sim = doc_similarities[selected_indices[i], selected_indices[j]]
                    selected_similarities.append(sim)
            
            avg_similarity = np.mean(selected_similarities)
            diversity_score = 1.0 - avg_similarity
        else:
            diversity_score = 1.0
        
        logger.info(
            f"Selected {len(selected_docs)} documents with "
            f"diversity_score={diversity_score:.3f}"
        )
        
        return {
            "documents": selected_docs,
            "scores": relevance_scores,
            "relevance_scores": relevance_scores,
            "diversity_score": float(diversity_score),
            "selected_indices": selected_indices
        }
    
    def retrieve_and_generate(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve diverse documents and generate response.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with answer and retrieval metrics
        """
        # Retrieve with diversity
        retrieval_result = self.retrieve_with_diversity(query, k, **kwargs)
        
        if not retrieval_result["documents"]:
            return {
                "answer": "No relevant documents found.",
                **retrieval_result
            }
        
        # Generate answer
        context = "\n\n".join([
            doc.page_content for doc in retrieval_result["documents"]
        ])
        
        prompt = f"""Based on the following diverse and relevant documents, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.invoke(prompt).content
        
        return {
            "answer": answer,
            **retrieval_result
        }
