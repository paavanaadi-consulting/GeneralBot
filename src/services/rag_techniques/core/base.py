"""
Base classes for RAG implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from rag_techniques.core.config import RAGConfig


class BaseRAG(ABC):
    """
    Abstract base class for all RAG implementations
    
    This provides a common interface that all RAG techniques must implement.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG system
        
        Args:
            config: RAG configuration object
        """
        self.config = config or RAGConfig()
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.time_records = {}
    
    @abstractmethod
    def load_documents(self, source: str) -> Any:
        """
        Load documents from a source
        
        Args:
            source: Path to document or data source
            
        Returns:
            Loaded documents
        """
        pass
    
    @abstractmethod
    def create_vector_store(self, documents: Any) -> Any:
        """
        Create a vector store from documents
        
        Args:
            documents: Loaded documents
            
        Returns:
            Vector store instance
        """
        pass
    
    @abstractmethod
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary containing the answer and metadata
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dictionary of metrics
        """
        return {
            "time_records": self.time_records,
            "config": self.config.to_dict(),
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(config={self.config})"


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Any]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        pass


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
