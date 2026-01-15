"""
Contextual Chunk Headers (CCH)

This module implements contextual chunk headers that prepend higher-level context
(document titles, summaries, section headers) to chunks before embedding, improving
retrieval accuracy and reducing irrelevant results.
"""

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class ContextualChunkHeadersRAG(BaseRAGTechnique):
    """
    RAG with contextual chunk headers.
    
    Adds higher-level context to chunks by prepending headers containing:
    - Document title
    - Document summary (optional)
    - Section/subsection titles (optional)
    
    This gives embeddings more accurate representation and improves retrieval.
    
    Attributes:
        embeddings: OpenAI embeddings model
        vectorstore: FAISS vector store
        include_summary: Whether to include document summary in headers
        include_sections: Whether to include section titles
        chunk_headers: Generated headers for each chunk
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        include_summary: bool = True,
        include_sections: bool = False,
        top_k: int = 3
    ):
        """
        Initialize Contextual Chunk Headers RAG.
        
        Args:
            config: Configuration manager instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            include_summary: Include document summary in headers
            include_sections: Include section titles in headers
            top_k: Number of chunks to retrieve
        """
        super().__init__(config)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_summary = include_summary
        self.include_sections = include_sections
        self.top_k = top_k
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        self.vectorstore: Optional[FAISS] = None
        self.chunk_headers: Dict[str, str] = {}
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.title_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate a concise, descriptive title for the following document.
The title should capture the main topic and be suitable as a document identifier.

Document (first 1000 characters):
{text}

Title:"""
        )
        
        self.summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate a concise 2-3 sentence summary of the main points in this document.

Document:
{text}

Summary:"""
        )
    
    def generate_document_title(self, document: Document) -> str:
        """
        Generate a descriptive title for a document.
        
        Args:
            document: Document to generate title for
            
        Returns:
            Generated title string
        """
        # Use existing title if available
        if "title" in document.metadata:
            return document.metadata["title"]
        
        # Generate title using LLM
        text_preview = document.page_content[:1000]
        prompt = self.title_prompt.format(text=text_preview)
        response = self.llm.invoke(prompt)
        title = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up title
        title = title.strip().strip('"').strip("'")
        
        self.logger.info(f"Generated title: {title}")
        return title
    
    def generate_document_summary(self, document: Document) -> str:
        """
        Generate a concise summary for a document.
        
        Args:
            document: Document to summarize
            
        Returns:
            Generated summary string
        """
        # Limit text for summary to avoid token limits
        text = document.page_content[:2000]
        prompt = self.summary_prompt.format(text=text)
        response = self.llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        self.logger.info("Generated document summary")
        return summary.strip()
    
    def create_chunk_header(
        self,
        title: str,
        summary: Optional[str] = None,
        section: Optional[str] = None
    ) -> str:
        """
        Create a contextual header for a chunk.
        
        Args:
            title: Document title
            summary: Document summary (optional)
            section: Section title (optional)
            
        Returns:
            Formatted header string
        """
        header_parts = [f"Document: {title}"]
        
        if summary and self.include_summary:
            header_parts.append(f"Summary: {summary}")
        
        if section and self.include_sections:
            header_parts.append(f"Section: {section}")
        
        header = " | ".join(header_parts)
        return f"[{header}]\n\n"
    
    def process_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[Document]:
        """
        Process documents with contextual chunk headers.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            List of documents with headers prepended
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks_with_headers = []
        
        for doc_idx, document in enumerate(documents):
            # Generate document-level context
            title = self.generate_document_title(document)
            summary = None
            if self.include_summary:
                summary = self.generate_document_summary(document)
            
            # Split document into chunks
            chunks = text_splitter.split_documents([document])
            
            # Add contextual headers to each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Get section title if available
                section = chunk.metadata.get("section", None)
                
                # Create header
                header = self.create_chunk_header(title, summary, section)
                
                # Create new document with header prepended
                chunk_with_header = Document(
                    page_content=header + chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "doc_id": doc_idx,
                        "chunk_id": chunk_idx,
                        "title": title,
                        "has_header": True,
                        "header_length": len(header)
                    }
                )
                
                # Store header for later use
                chunk_id = f"{doc_idx}_{chunk_idx}"
                self.chunk_headers[chunk_id] = header
                
                chunks_with_headers.append(chunk_with_header)
        
        self.logger.info(
            f"Created {len(chunks_with_headers)} chunks with contextual headers "
            f"from {len(documents)} documents"
        )
        
        return chunks_with_headers
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        Create FAISS vector store with contextual headers.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            FAISS vector store instance
        """
        # Process documents with headers
        chunks = self.process_documents(documents)
        
        # Create vector store (embeddings include headers)
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        self.logger.info("Created vector store with contextual chunk headers")
        return self.vectorstore
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant chunks.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents with headers
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k
        
        # Retrieve similar documents
        results = self.vectorstore.similarity_search(query, k=k)
        
        self.logger.info(f"Retrieved {len(results)} chunks with contextual headers")
        return results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_headers_in_context: bool = True,
        return_metadata: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using contextual chunk headers.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            include_headers_in_context: Keep headers when passing to LLM
            return_metadata: Whether to return header metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve chunks
        chunks = self.retrieve(query, top_k=top_k)
        
        # Prepare context
        if include_headers_in_context:
            # Keep headers - they provide context for the LLM
            context = "\n\n".join([chunk.page_content for chunk in chunks])
        else:
            # Remove headers for cleaner context
            context_parts = []
            for chunk in chunks:
                header_len = chunk.metadata.get("header_length", 0)
                content = chunk.page_content[header_len:] if header_len else chunk.page_content
                context_parts.append(content)
            context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self._generate_answer(query, context)
        
        result = {
            "answer": answer,
            "num_chunks": len(chunks)
        }
        
        if return_metadata:
            result["chunks"] = [
                {
                    "title": chunk.metadata.get("title", "Unknown"),
                    "chunk_id": f"{chunk.metadata.get('doc_id')}_{chunk.metadata.get('chunk_id')}",
                    "header_length": chunk.metadata.get("header_length", 0)
                }
                for chunk in chunks
            ]
        
        return result
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using context with headers."""
        prompt = f"""Answer the question using the following context.
Note: Context includes document headers that provide additional information about each chunk.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def compare_with_without_headers(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Compare retrieval with and without contextual headers.
        
        Args:
            query: Query string
            documents: Documents to test with
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary comparing both approaches
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Without headers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks_no_headers = text_splitter.split_documents(documents)
        vectorstore_no_headers = FAISS.from_documents(
            chunks_no_headers,
            self.embeddings
        )
        results_no_headers = vectorstore_no_headers.similarity_search(query, k=top_k)
        
        # With headers
        self.create_vectorstore(documents)
        results_with_headers = self.retrieve(query, top_k=top_k)
        
        return {
            "query": query,
            "without_headers": {
                "num_results": len(results_no_headers),
                "results": [doc.page_content[:200] for doc in results_no_headers]
            },
            "with_headers": {
                "num_results": len(results_with_headers),
                "results": [doc.page_content[:200] for doc in results_with_headers]
            }
        }
    
    def get_header_stats(self) -> Dict[str, Any]:
        """
        Get statistics about chunk headers.
        
        Returns:
            Dictionary with header statistics
        """
        if not self.chunk_headers:
            return {"error": "No chunk headers generated yet"}
        
        header_lengths = [len(header) for header in self.chunk_headers.values()]
        
        return {
            "total_chunks": len(self.chunk_headers),
            "avg_header_length": sum(header_lengths) / len(header_lengths),
            "min_header_length": min(header_lengths),
            "max_header_length": max(header_lengths),
            "include_summary": self.include_summary,
            "include_sections": self.include_sections
        }
