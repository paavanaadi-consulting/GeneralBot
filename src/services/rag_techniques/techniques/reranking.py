"""
Reranking Techniques

This module implements document reranking to improve the quality of
retrieved documents by reordering them based on relevance.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.base import BaseRAG
from ..core.config import RAGConfig


class RelevanceScore(BaseModel):
    """Model for relevance scoring."""
    score: float = Field(description="Relevance score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the score")


class RerankingRAG(BaseRAG):
    """
    RAG with document reranking.
    
    Implements multiple reranking strategies:
    1. LLM-based reranking - Use LLM to score document relevance
    2. Cross-encoder reranking - Use similarity-based scoring
    3. Diversity reranking - Balance relevance with diversity (MMR)
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        content: Optional[str] = None,
        config: Optional[RAGConfig] = None,
        rerank_top_k: int = 10
    ):
        """
        Initialize reranking RAG.
        
        Args:
            pdf_path: Path to PDF document
            content: Text content
            config: RAG configuration
            rerank_top_k: Number of documents to retrieve before reranking
        """
        super().__init__(pdf_path, content, config)
        self.rerank_top_k = rerank_top_k
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens
        )
    
    def llm_rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """
        Rerank documents using LLM to score relevance.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if top_k is None:
            top_k = self.config.n_retrieved
        
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            Rate the relevance of the following document to the query on a scale of 0 to 1.
            
            Query: {query}
            
            Document: {document}
            
            Provide:
            1. A relevance score (0.0 to 1.0, where 1.0 is most relevant)
            2. Brief reasoning for your score
            
            Consider:
            - Direct answer to the query
            - Topical relevance
            - Information completeness
            """
        )
        
        chain = prompt | self.llm.with_structured_output(RelevanceScore)
        
        # Score each document
        scored_docs = []
        for doc in documents:
            try:
                result = chain.invoke({
                    "query": query,
                    "document": doc.page_content[:1000]  # Limit length
                })
                scored_docs.append((doc, result.score))
            except Exception as e:
                print(f"Error scoring document: {e}")
                scored_docs.append((doc, 0.0))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def diversity_rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None,
        lambda_param: float = 0.5
    ) -> List[Tuple[Any, float]]:
        """
        Rerank using Maximal Marginal Relevance (MMR) for diversity.
        
        Balances relevance with diversity to avoid redundant results.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            lambda_param: Balance parameter (0=max diversity, 1=max relevance)
            
        Returns:
            List of (document, score) tuples
        """
        if top_k is None:
            top_k = self.config.n_retrieved
        
        # Get embeddings
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        
        # Embed query
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array(query_embedding)
        
        # Embed documents
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = embeddings.embed_documents(doc_texts)
        doc_embeddings = np.array(doc_embeddings)
        
        # Calculate similarity to query
        query_similarities = np.dot(doc_embeddings, query_embedding)
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break
            
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_similarities[idx]
                
                # Max similarity to already selected documents
                if selected_indices:
                    selected_embeddings = doc_embeddings[selected_indices]
                    similarities = np.dot(selected_embeddings, doc_embeddings[idx])
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))
            
            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected documents with scores
        return [(documents[idx], query_similarities[idx]) for idx in selected_indices]
    
    def semantic_rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """
        Rerank documents using semantic similarity scores.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples
        """
        if top_k is None:
            top_k = self.config.n_retrieved
        
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        
        # Embed query
        query_embedding = np.array(embeddings.embed_query(query))
        
        # Embed documents and calculate similarity
        scored_docs = []
        for doc in documents:
            doc_embedding = np.array(embeddings.embed_query(doc.page_content[:1000]))
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scored_docs.append((doc, float(similarity)))
        
        # Sort by similarity descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def query(
        self,
        query_text: str,
        rerank_method: str = "llm",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with document reranking.
        
        Args:
            query_text: The search query
            rerank_method: Reranking method - 'llm', 'diversity', 'semantic'
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary with answer, reranked documents, and scores
        """
        # Retrieve more documents than needed
        initial_docs = self.vectorstore.similarity_search(
            query_text,
            k=self.rerank_top_k
        )
        
        # Rerank documents
        if rerank_method == "llm":
            reranked_docs = self.llm_rerank(query_text, initial_docs)
        elif rerank_method == "diversity":
            lambda_param = kwargs.get("lambda_param", 0.5)
            reranked_docs = self.diversity_rerank(query_text, initial_docs, lambda_param=lambda_param)
        elif rerank_method == "semantic":
            reranked_docs = self.semantic_rerank(query_text, initial_docs)
        else:
            raise ValueError(f"Unknown reranking method: {rerank_method}")
        
        # Extract documents and scores
        docs = [doc for doc, score in reranked_docs]
        scores = [score for doc, score in reranked_docs]
        context = [doc.page_content for doc in docs]
        
        # Generate answer using top documents
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
            "method": rerank_method,
            "context": context,
            "relevance_scores": scores,
            "num_docs_initial": len(initial_docs),
            "num_docs_final": len(docs)
        }
