"""
Configuration module for RAG systems
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"


class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"


@dataclass
class RAGConfig:
    """
    Configuration class for RAG systems
    
    Attributes:
        chunk_size: Size of text chunks for splitting documents
        chunk_overlap: Overlap between consecutive chunks
        n_retrieved: Number of chunks to retrieve
        search_type: Type of search (similarity, mmr, similarity_score_threshold)
        model_name: Name of the LLM model
        temperature: LLM temperature for generation
        max_tokens: Maximum tokens for LLM generation
        embedding_provider: Provider for embeddings
        embedding_model: Specific embedding model to use
        vector_store_type: Type of vector store (faiss, chroma)
        custom_params: Additional custom parameters
    """
    
    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval parameters
    n_retrieved: int = 2
    search_type: str = "similarity"
    search_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # LLM parameters
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 4000
    
    # Embedding parameters
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: Optional[str] = None
    
    # Vector store parameters
    vector_store_type: str = "faiss"
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.n_retrieved <= 0:
            raise ValueError("n_retrieved must be positive")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "n_retrieved": self.n_retrieved,
            "search_type": self.search_type,
            "search_kwargs": self.search_kwargs,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "embedding_provider": self.embedding_provider.value,
            "embedding_model": self.embedding_model,
            "vector_store_type": self.vector_store_type,
            "custom_params": self.custom_params,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary"""
        if "embedding_provider" in config_dict:
            config_dict["embedding_provider"] = EmbeddingProvider(
                config_dict["embedding_provider"]
            )
        return cls(**config_dict)
