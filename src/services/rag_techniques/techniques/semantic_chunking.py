"""
Semantic Chunking for Document Processing

This module implements semantic chunking that splits text at natural breakpoints
while preserving semantic coherence within each chunk, rather than using fixed
character or word counts.
"""

from typing import List, Optional, Dict, Any, Literal
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class SemanticChunkingRAG(BaseRAGTechnique):
    """
    Semantic chunking that creates meaningful and context-aware text segments.
    
    Unlike traditional methods that split text based on fixed sizes, semantic chunking
    attempts to split text at natural breakpoints, preserving semantic coherence.
    
    Attributes:
        breakpoint_type: Type of breakpoint detection ('percentile', 'standard_deviation', 'interquartile')
        breakpoint_threshold: Threshold value for breakpoint detection
        embeddings: OpenAI embeddings model
        semantic_chunker: SemanticChunker instance
        vectorstore: FAISS vector store for chunk retrieval
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        breakpoint_type: Literal["percentile", "standard_deviation", "interquartile"] = "percentile",
        breakpoint_threshold: int = 90,
        top_k: int = 3
    ):
        """
        Initialize Semantic Chunking RAG.
        
        Args:
            config: Configuration manager instance
            breakpoint_type: Method for determining split points:
                - 'percentile': Splits at differences greater than the X percentile
                - 'standard_deviation': Splits at differences greater than X std devs
                - 'interquartile': Uses interquartile distance for split points
            breakpoint_threshold: Threshold value for the breakpoint type
            top_k: Number of chunks to retrieve
        """
        super().__init__(config)
        self.breakpoint_type = breakpoint_type
        self.breakpoint_threshold = breakpoint_threshold
        self.top_k = top_k
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_threshold
        )
        
        self.vectorstore: Optional[FAISS] = None
        
    def process_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[Document]:
        """
        Process documents using semantic chunking.
        
        Args:
            documents: List of LangChain Document objects to process
            **kwargs: Additional parameters
            
        Returns:
            List of semantically chunked documents
        """
        # Combine all documents into one text if needed
        if len(documents) == 1:
            text = documents[0].page_content
        else:
            text = "\n\n".join([doc.page_content for doc in documents])
        
        # Create semantic chunks
        chunks = self.semantic_chunker.create_documents([text])
        
        self.logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        Create FAISS vector store from semantically chunked documents.
        
        Args:
            documents: List of chunked documents
            **kwargs: Additional parameters
            
        Returns:
            FAISS vector store instance
        """
        # Process documents with semantic chunking
        chunks = self.process_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        self.logger.info("Created FAISS vector store from semantic chunks")
        return self.vectorstore
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant chunks using semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve (overrides default)
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k
        
        # Retrieve similar documents
        results = self.vectorstore.similarity_search(query, k=k)
        
        self.logger.info(f"Retrieved {len(results)} semantically relevant chunks")
        return results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the semantic chunking RAG system.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            return_context: Whether to return retrieved context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing answer and optionally context
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve(query, top_k=top_k)
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in relevant_chunks])
        
        # Generate answer using LLM
        answer = self._generate_answer(query, context)
        
        result = {
            "answer": answer,
            "num_chunks": len(relevant_chunks)
        }
        
        if return_context:
            result["context"] = context
            result["chunks"] = [doc.page_content for doc in relevant_chunks]
        
        return result
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_chunk_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the semantic chunks.
        
        Returns:
            Dictionary with chunk statistics
        """
        if self.vectorstore is None:
            return {"error": "Vector store not initialized"}
        
        # Get all documents from vectorstore
        docs = self.vectorstore.docstore._dict.values()
        chunk_sizes = [len(doc.page_content) for doc in docs]
        
        return {
            "num_chunks": len(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "breakpoint_type": self.breakpoint_type,
            "breakpoint_threshold": self.breakpoint_threshold
        }
