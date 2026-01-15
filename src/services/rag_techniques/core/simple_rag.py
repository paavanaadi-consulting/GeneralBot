"""
Simple RAG implementation
"""

import time
from typing import Any, Dict, Optional
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from rag_techniques.core.base import BaseRAG
from rag_techniques.core.config import RAGConfig, EmbeddingProvider
from rag_techniques.utils.helpers import replace_t_with_space


class SimpleRAG(BaseRAG):
    """
    Simple RAG implementation using FAISS vector store and OpenAI
    
    This is the foundational RAG technique that:
    1. Loads and processes PDF documents
    2. Splits text into chunks
    3. Creates embeddings and vector store
    4. Retrieves relevant chunks for queries
    5. Generates answers using LLM
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize Simple RAG
        
        Args:
            pdf_path: Path to PDF document
            config: RAG configuration
        """
        super().__init__(config)
        self.pdf_path = pdf_path
        
        if pdf_path:
            self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system with document"""
        print("\n--- Initializing Simple RAG ---")
        
        # Load and process documents
        start_time = time.time()
        documents = self.load_documents(self.pdf_path)
        self.vector_store = self.create_vector_store(documents)
        self.time_records['initialization'] = time.time() - start_time
        
        print(f"Initialization Time: {self.time_records['initialization']:.2f} seconds")
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.n_retrieved}
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def load_documents(self, source: str) -> Any:
        """
        Load documents from PDF
        
        Args:
            source: Path to PDF file
            
        Returns:
            List of document chunks
        """
        loader = PyPDFLoader(source)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Clean texts
        cleaned_texts = replace_t_with_space(texts)
        
        return cleaned_texts
    
    def create_vector_store(self, documents: Any) -> Any:
        """
        Create FAISS vector store from documents
        
        Args:
            documents: Document chunks
            
        Returns:
            FAISS vector store
        """
        # Get embeddings based on provider
        if self.config.embedding_provider == EmbeddingProvider.OPENAI:
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model or "text-embedding-3-small"
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.config.embedding_provider}"
            )
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        return vectorstore
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer, source documents, and metadata
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Provide pdf_path during init.")
        
        start_time = time.time()
        
        # Get response
        result = self.qa_chain({"query": question})
        
        retrieval_time = time.time() - start_time
        self.time_records['last_query_time'] = retrieval_time
        
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", []),
            "retrieval_time": retrieval_time,
            "query": question
        }
    
    def retrieve_context(self, question: str) -> list:
        """
        Retrieve relevant context for a question
        
        Args:
            question: User question
            
        Returns:
            List of relevant document chunks
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        docs = self.retriever.get_relevant_documents(question)
        return [doc.page_content for doc in docs]
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Re-initialize if necessary
        if any(key in kwargs for key in ['chunk_size', 'chunk_overlap', 'n_retrieved']):
            if self.pdf_path:
                self._initialize()
