"""Vector store management for RAG system."""
import logging
from typing import List, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage vector database for document embeddings."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store: Optional[Chroma] = None
        self._ensure_vector_db_directory()
    
    def _ensure_vector_db_directory(self):
        """Ensure vector database directory exists."""
        Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=settings.vector_db_path
        )
        
        logger.info(f"Vector store created and persisted to {settings.vector_db_path}")
        
        return self.vector_store
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing vector store from disk.
        
        Returns:
            Chroma vector store instance or None if not found
        """
        try:
            self.vector_store = Chroma(
                persist_directory=settings.vector_db_path,
                embedding_function=self.embeddings
            )
            logger.info(f"Vector store loaded from {settings.vector_db_path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized. Creating new one.")
            self.create_vector_store(documents)
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.vector_store.add_documents(documents)
        logger.info("Documents added and persisted")
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Search for similar documents based on query.
        
        Args:
            query: Search query
            k: Number of results to return (default: from settings)
            
        Returns:
            List of most similar Document objects
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized. Attempting to load.")
            self.load_vector_store()
            
            if not self.vector_store:
                logger.error("Failed to load vector store")
                return []
        
        k = k or settings.top_k_results
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return (default: from settings)
            
        Returns:
            List of tuples (Document, score)
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized. Attempting to load.")
            self.load_vector_store()
            
            if not self.vector_store:
                logger.error("Failed to load vector store")
                return []
        
        k = k or settings.top_k_results
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search with score: {str(e)}")
            return []
    
    def delete_vector_store(self):
        """Delete the vector store."""
        import shutil
        if Path(settings.vector_db_path).exists():
            shutil.rmtree(settings.vector_db_path)
            logger.info(f"Vector store deleted from {settings.vector_db_path}")
        self.vector_store = None
