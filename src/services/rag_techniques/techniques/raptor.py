"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

This module implements RAPTOR, which creates a hierarchical tree structure
of document summaries at multiple levels, allowing for both broad and detailed
information retrieval.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.mixture import GaussianMixture
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class RAPTORRAG(BaseRAGTechnique):
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
    
    Creates a hierarchical tree structure where each level contains summaries
    of clusters from the previous level. This allows efficient navigation between
    high-level concepts and specific details.
    
    Attributes:
        embeddings: OpenAI embeddings model
        tree_levels: List of documents at each level of the tree
        vectorstore: Combined vector store with all levels
        max_levels: Maximum tree depth
        n_clusters: Number of clusters per level
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        max_levels: int = 3,
        n_clusters: int = 5,
        top_k: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize RAPTOR RAG.
        
        Args:
            config: Configuration manager instance
            max_levels: Maximum depth of the tree
            n_clusters: Number of clusters per level
            top_k: Number of documents to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        super().__init__(config)
        self.max_levels = max_levels
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        self.tree_levels: List[List[Document]] = []
        self.vectorstore: Optional[FAISS] = None
        
        # Summary prompt template
        self.summary_template = ChatPromptTemplate.from_template(
            """Summarize the following documents into a cohesive summary that captures the main themes and key information:

{documents}

Summary:"""
        )
    
    def _embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Embed documents and return embeddings as numpy array."""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)
    
    def _cluster_documents(
        self,
        documents: List[Document],
        n_clusters: int
    ) -> List[List[int]]:
        """
        Cluster documents using Gaussian Mixture Model.
        
        Args:
            documents: List of documents to cluster
            n_clusters: Number of clusters
            
        Returns:
            List of lists, where each inner list contains document indices for a cluster
        """
        if len(documents) <= n_clusters:
            # If we have fewer docs than clusters, each doc is its own cluster
            return [[i] for i in range(len(documents))]
        
        # Embed documents
        embeddings = self._embed_documents(documents)
        
        # Perform clustering
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(embeddings)
        
        # Group document indices by cluster
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        # Filter out empty clusters
        clusters = [c for c in clusters if c]
        
        self.logger.info(f"Clustered {len(documents)} documents into {len(clusters)} clusters")
        return clusters
    
    def _summarize_cluster(
        self,
        documents: List[Document],
        cluster_id: int,
        level: int
    ) -> Document:
        """
        Summarize a cluster of documents.
        
        Args:
            documents: Documents in the cluster
            cluster_id: Cluster identifier
            level: Tree level
            
        Returns:
            Summary document
        """
        # Combine document texts
        combined_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
        
        # Generate summary
        prompt = self.summary_template.format_messages(documents=combined_text)
        response = self.llm.invoke(prompt)
        summary_text = response.content if hasattr(response, 'content') else str(response)
        
        # Create summary document with metadata
        summary_doc = Document(
            page_content=summary_text,
            metadata={
                "level": level,
                "cluster_id": cluster_id,
                "num_source_docs": len(documents),
                "type": "summary"
            }
        )
        
        self.logger.info(f"Created summary for cluster {cluster_id} at level {level}")
        return summary_doc
    
    def build_tree(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[List[Document]]:
        """
        Build hierarchical tree structure.
        
        Args:
            documents: Base level documents
            **kwargs: Additional parameters
            
        Returns:
            List of document lists, one per tree level
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Split documents into chunks for level 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        level_0_docs = text_splitter.split_documents(documents)
        
        # Add level metadata
        for doc in level_0_docs:
            doc.metadata["level"] = 0
            doc.metadata["type"] = "chunk"
        
        self.tree_levels = [level_0_docs]
        self.logger.info(f"Level 0: {len(level_0_docs)} documents")
        
        # Build subsequent levels
        current_level_docs = level_0_docs
        for level in range(1, self.max_levels):
            # Stop if we've condensed to very few documents
            if len(current_level_docs) <= 2:
                self.logger.info(f"Stopping at level {level-1}: too few documents")
                break
            
            # Cluster current level
            clusters = self._cluster_documents(
                current_level_docs,
                min(self.n_clusters, len(current_level_docs))
            )
            
            # Generate summaries for each cluster
            next_level_docs = []
            for cluster_id, cluster_indices in enumerate(clusters):
                cluster_docs = [current_level_docs[i] for i in cluster_indices]
                summary_doc = self._summarize_cluster(cluster_docs, cluster_id, level)
                next_level_docs.append(summary_doc)
            
            self.tree_levels.append(next_level_docs)
            self.logger.info(f"Level {level}: {len(next_level_docs)} summaries")
            
            current_level_docs = next_level_docs
        
        return self.tree_levels
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        Create vector store containing all tree levels.
        
        Args:
            documents: Base documents
            **kwargs: Additional parameters
            
        Returns:
            FAISS vector store with all levels
        """
        # Build tree
        tree_levels = self.build_tree(documents)
        
        # Combine all levels into single list
        all_docs = []
        for level_docs in tree_levels:
            all_docs.extend(level_docs)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=all_docs,
            embedding=self.embeddings
        )
        
        self.logger.info(f"Created vector store with {len(all_docs)} documents from all levels")
        return self.vectorstore
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        prefer_level: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents from the tree.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            prefer_level: Preferred tree level (None for all levels)
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k
        
        # Retrieve documents
        results = self.vectorstore.similarity_search(query, k=k * 2)
        
        # Filter by level if specified
        if prefer_level is not None:
            results = [doc for doc in results if doc.metadata.get("level") == prefer_level]
            results = results[:k]
        else:
            results = results[:k]
        
        self.logger.info(f"Retrieved {len(results)} documents from tree")
        return results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        prefer_level: Optional[int] = None,
        return_context: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the RAPTOR system.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            prefer_level: Preferred tree level
            return_context: Whether to return retrieved context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k, prefer_level)
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate answer
        answer = self._generate_answer(query, context, retrieved_docs)
        
        result = {
            "answer": answer,
            "num_docs": len(retrieved_docs),
            "levels_used": list(set(doc.metadata.get("level", 0) for doc in retrieved_docs))
        }
        
        if return_context:
            result["context"] = context
            result["documents"] = [
                {
                    "content": doc.page_content,
                    "level": doc.metadata.get("level"),
                    "type": doc.metadata.get("type")
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        documents: List[Document]
    ) -> str:
        """Generate answer using LLM with multi-level context."""
        # Organize context by level
        levels_info = {}
        for doc in documents:
            level = doc.metadata.get("level", 0)
            if level not in levels_info:
                levels_info[level] = []
            levels_info[level].append(doc.page_content)
        
        # Format context with level information
        formatted_context = []
        for level in sorted(levels_info.keys()):
            level_content = "\n".join(levels_info[level])
            formatted_context.append(f"Level {level} ({'summaries' if level > 0 else 'details'}):\n{level_content}")
        
        context_str = "\n\n".join(formatted_context)
        
        prompt = f"""Based on the following hierarchical context from RAPTOR tree retrieval, answer the question.
The context includes information from different abstraction levels.

{context_str}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the tree structure."""
        if not self.tree_levels:
            return {"error": "Tree not built yet"}
        
        return {
            "num_levels": len(self.tree_levels),
            "docs_per_level": [len(level) for level in self.tree_levels],
            "total_docs": sum(len(level) for level in self.tree_levels),
            "max_levels": self.max_levels,
            "n_clusters": self.n_clusters
        }
