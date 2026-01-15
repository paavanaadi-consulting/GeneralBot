"""
RAG Techniques Package - A comprehensive toolkit for building advanced RAG systems
"""

__version__ = "0.1.0"
__author__ = "RAG Techniques Contributors"

# Core imports
from .core.simple_rag import SimpleRAG
from .core.config import RAGConfig
from .core.base import BaseRAG

# Technique imports
from .techniques.compression import ContextualCompressionRAG
from .techniques.query_transform import QueryTransformRAG
from .techniques.reranking import RerankingRAG
from .techniques.fusion import FusionRAG
from .techniques.feedback import FeedbackRAG

# Evaluation imports
from .evaluation.metrics import RAGEvaluator

# Utility imports
from .utils.helpers import (
    replace_t_with_space,
    text_wrap,
    show_context,
    encode_pdf,
    encode_from_string,
    read_pdf_to_string,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Core
    "SimpleRAG",
    "RAGConfig",
    "BaseRAG",
    
    # Techniques
    "ContextualCompressionRAG",
    "QueryTransformRAG",
    "RerankingRAG",
    "FusionRAG",
    "FeedbackRAG",
    
    # Evaluation
    "RAGEvaluator",
    
    # Utils
    "replace_t_with_space",
    "text_wrap",
    "show_context",
    "encode_pdf",
    "encode_from_string",
    "read_pdf_to_string",
]
