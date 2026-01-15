"""
Hypothetical Document Embedding (HyDE)

This module implements HyDE, which transforms queries into hypothetical documents
that contain the answer. This bridges the gap between query and document distributions
in vector space, improving retrieval relevance.
"""

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class HyDERAG(BaseRAGTechnique):
    """
    Hypothetical Document Embedding for improved retrieval.
    
    HyDE generates a hypothetical document that would answer the query,
    then uses this expanded query for retrieval. This helps bridge the
    semantic gap between short queries and longer documents.
    
    Attributes:
        embeddings: OpenAI embeddings model
        vectorstore: FAISS vector store
        hypothetical_doc_template: Template for generating hypothetical documents
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        top_k: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize HyDE RAG.
        
        Args:
            config: Configuration manager instance
            top_k: Number of documents to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        super().__init__(config)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        # Template for generating hypothetical documents
        self.hypothetical_doc_template = PromptTemplate(
            input_variables=["question", "chunk_size"],
            template="""Please write a detailed passage that would answer the following question. 
The passage should be approximately {chunk_size} characters long and contain specific details and examples.

Question: {question}

Detailed passage:"""
        )
        
        self.vectorstore: Optional[FAISS] = None
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        
        Args:
            query: User query
            
        Returns:
            Generated hypothetical document
        """
        prompt = self.hypothetical_doc_template.format(
            question=query,
            chunk_size=self.chunk_size
        )
        
        response = self.llm.invoke(prompt)
        hypothetical_doc = response.content if hasattr(response, 'content') else str(response)
        
        self.logger.info(f"Generated hypothetical document of length {len(hypothetical_doc)}")
        return hypothetical_doc
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of documents to index
            **kwargs: Additional parameters
            
        Returns:
            FAISS vector store instance
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        self.logger.info(f"Created vector store with {len(chunks)} chunks")
        return self.vectorstore
    
    def retrieve_with_hyde(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> tuple[str, List[Document]]:
        """
        Retrieve documents using HyDE approach.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (hypothetical_document, retrieved_documents)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k
        
        # Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # Use hypothetical document as the search query
        results = self.vectorstore.similarity_search(
            hypothetical_doc,
            k=k
        )
        
        self.logger.info(f"Retrieved {len(results)} documents using HyDE")
        return hypothetical_doc, results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
        return_hypothetical: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using HyDE retrieval.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_context: Whether to return retrieved context
            return_hypothetical: Whether to return hypothetical document
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing answer and optionally context/hypothetical doc
        """
        # Retrieve using HyDE
        hypothetical_doc, retrieved_docs = self.retrieve_with_hyde(query, top_k)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate final answer
        answer = self._generate_answer(query, context)
        
        result = {
            "answer": answer,
            "num_docs": len(retrieved_docs)
        }
        
        if return_hypothetical:
            result["hypothetical_document"] = hypothetical_doc
        
        if return_context:
            result["context"] = context
            result["retrieved_docs"] = [doc.page_content for doc in retrieved_docs]
        
        return result
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""Based on the following context retrieved using hypothetical document matching, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def compare_retrieval(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Compare HyDE retrieval with standard retrieval.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary comparing both methods
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        # Standard retrieval
        standard_results = self.vectorstore.similarity_search(query, k=top_k)
        
        # HyDE retrieval
        hypothetical_doc, hyde_results = self.retrieve_with_hyde(query, top_k)
        
        return {
            "query": query,
            "hypothetical_document": hypothetical_doc,
            "standard_retrieval": [doc.page_content for doc in standard_results],
            "hyde_retrieval": [doc.page_content for doc in hyde_results],
            "comparison": {
                "standard_count": len(standard_results),
                "hyde_count": len(hyde_results)
            }
        }
