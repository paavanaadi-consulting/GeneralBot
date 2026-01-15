"""
Hierarchical Indices for Document Retrieval

This module implements a hierarchical indexing system with two levels:
document-level summaries and detailed chunks. This improves efficiency by first
identifying relevant sections through summaries, then drilling down to details.
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class HierarchicalIndicesRAG(BaseRAGTechnique):
    """
    Hierarchical indexing with document summaries and detailed chunks.
    
    Creates a two-tier search system:
    1. First searches document-level summaries to identify relevant sections
    2. Then searches detailed chunks within those relevant sections
    
    Attributes:
        summary_vectorstore: Vector store for document summaries
        chunk_vectorstore: Vector store for detailed chunks
        embeddings: OpenAI embeddings model
        summaries: List of generated summaries
        chunks_with_metadata: Chunks with document metadata
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        num_summaries: int = 3,
        num_chunks_per_summary: int = 3,
        batch_size: int = 5
    ):
        """
        Initialize Hierarchical Indices RAG.
        
        Args:
            config: Configuration manager instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            num_summaries: Number of summaries to retrieve
            num_chunks_per_summary: Number of chunks to retrieve per summary
            batch_size: Batch size for async operations
        """
        super().__init__(config)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_summaries = num_summaries
        self.num_chunks_per_summary = num_chunks_per_summary
        self.batch_size = batch_size
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.summary_vectorstore: Optional[FAISS] = None
        self.chunk_vectorstore: Optional[FAISS] = None
        self.summaries: List[Document] = []
        self.chunks_with_metadata: List[Document] = []
    
    async def _summarize_document_async(
        self,
        document: Document,
        doc_id: int
    ) -> Document:
        """
        Asynchronously summarize a document.
        
        Args:
            document: Document to summarize
            doc_id: Document identifier
            
        Returns:
            Document containing summary with metadata
        """
        try:
            # Create summarization chain
            summarize_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff"
            )
            
            # Generate summary
            summary = await summarize_chain.arun([document])
            
            # Create summary document with metadata
            summary_doc = Document(
                page_content=summary,
                metadata={
                    "doc_id": doc_id,
                    "type": "summary",
                    "source": document.metadata.get("source", "unknown")
                }
            )
            
            self.logger.info(f"Generated summary for document {doc_id}")
            return summary_doc
            
        except Exception as e:
            self.logger.error(f"Error summarizing document {doc_id}: {e}")
            # Return a fallback summary
            return Document(
                page_content=f"Summary of document {doc_id}",
                metadata={"doc_id": doc_id, "type": "summary", "error": str(e)}
            )
    
    async def _process_documents_async(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        Process documents asynchronously: create summaries and chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Tuple of (summaries, chunks_with_metadata)
        """
        summaries = []
        all_chunks = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            # Generate summaries for batch
            summary_tasks = [
                self._summarize_document_async(doc, i + j)
                for j, doc in enumerate(batch)
            ]
            
            batch_summaries = await asyncio.gather(*summary_tasks)
            summaries.extend(batch_summaries)
            
            # Create chunks for batch
            for j, doc in enumerate(batch):
                doc_id = i + j
                chunks = self.text_splitter.split_documents([doc])
                
                # Add metadata to chunks
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "doc_id": doc_id,
                        "chunk_id": chunk_idx,
                        "type": "chunk"
                    })
                
                all_chunks.extend(chunks)
            
            # Small delay between batches to avoid rate limits
            if i + self.batch_size < len(documents):
                await asyncio.sleep(1)
        
        return summaries, all_chunks
    
    def process_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> Tuple[List[Document], List[Document]]:
        """
        Process documents: create summaries and chunks.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (summaries, chunks)
        """
        # Run async processing
        summaries, chunks = asyncio.run(
            self._process_documents_async(documents)
        )
        
        self.summaries = summaries
        self.chunks_with_metadata = chunks
        
        self.logger.info(
            f"Processed {len(documents)} documents into "
            f"{len(summaries)} summaries and {len(chunks)} chunks"
        )
        
        return summaries, chunks
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> Tuple[FAISS, FAISS]:
        """
        Create hierarchical vector stores: summaries and chunks.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (summary_vectorstore, chunk_vectorstore)
        """
        # Process documents
        summaries, chunks = self.process_documents(documents)
        
        # Create summary vector store
        self.summary_vectorstore = FAISS.from_documents(
            documents=summaries,
            embedding=self.embeddings
        )
        
        # Create chunk vector store
        self.chunk_vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        self.logger.info("Created hierarchical vector stores")
        return self.summary_vectorstore, self.chunk_vectorstore
    
    def retrieve_hierarchical(
        self,
        query: str,
        num_summaries: Optional[int] = None,
        num_chunks_per_summary: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform hierarchical retrieval: summaries first, then chunks.
        
        Args:
            query: Query string
            num_summaries: Number of summaries to retrieve
            num_chunks_per_summary: Number of chunks per summary
            **kwargs: Additional parameters
            
        Returns:
            List of relevant Document chunks
        """
        if self.summary_vectorstore is None or self.chunk_vectorstore is None:
            raise ValueError("Vector stores not initialized. Call create_vectorstore first.")
        
        num_sums = num_summaries or self.num_summaries
        num_chunks = num_chunks_per_summary or self.num_chunks_per_summary
        
        # Step 1: Retrieve relevant summaries
        relevant_summaries = self.summary_vectorstore.similarity_search(
            query,
            k=num_sums
        )
        
        # Step 2: For each relevant summary, retrieve detailed chunks
        all_chunks = []
        for summary in relevant_summaries:
            doc_id = summary.metadata.get("doc_id")
            
            # Filter chunks by document ID
            if doc_id is not None:
                # Search in chunk vectorstore with document filter
                chunks = self.chunk_vectorstore.similarity_search(
                    query,
                    k=num_chunks,
                    filter=lambda metadata: metadata.get("doc_id") == doc_id
                )
                all_chunks.extend(chunks)
        
        # If no chunks found with filter, fallback to general search
        if not all_chunks:
            all_chunks = self.chunk_vectorstore.similarity_search(
                query,
                k=num_sums * num_chunks
            )
        
        self.logger.info(
            f"Retrieved {len(relevant_summaries)} summaries and "
            f"{len(all_chunks)} total chunks"
        )
        
        return all_chunks
    
    def query(
        self,
        query: str,
        num_summaries: Optional[int] = None,
        num_chunks_per_summary: Optional[int] = None,
        return_context: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the hierarchical RAG system.
        
        Args:
            query: Query string
            num_summaries: Number of summaries to retrieve
            num_chunks_per_summary: Number of chunks per summary
            return_context: Whether to return retrieved context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing answer and optionally context
        """
        # Perform hierarchical retrieval
        relevant_chunks = self.retrieve_hierarchical(
            query,
            num_summaries=num_summaries,
            num_chunks_per_summary=num_chunks_per_summary
        )
        
        # Prepare context
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # Generate answer
        answer = self._generate_answer(query, context)
        
        result = {
            "answer": answer,
            "num_chunks": len(relevant_chunks)
        }
        
        if return_context:
            result["context"] = context
            result["chunks"] = [chunk.page_content for chunk in relevant_chunks]
            result["metadata"] = [chunk.metadata for chunk in relevant_chunks]
        
        return result
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""Based on the following context retrieved from document summaries and detailed chunks, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hierarchical structure.
        
        Returns:
            Dictionary with hierarchy statistics
        """
        return {
            "num_summaries": len(self.summaries),
            "num_chunks": len(self.chunks_with_metadata),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_config": {
                "num_summaries": self.num_summaries,
                "num_chunks_per_summary": self.num_chunks_per_summary
            }
        }
