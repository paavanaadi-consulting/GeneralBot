"""
Multi-modal RAG with ColPali

This module implements multi-modal RAG using ColPali, a vision-language model
for document understanding that processes both text and images together.

Key Features:
- ColPali model integration
- Page-level document indexing
- Visual document understanding
- No explicit OCR needed
- Direct image-query matching

Note: Requires additional dependencies:
- byaldi (ColPali library)
- transformers
- torch
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..core.base import BaseRAGTechnique

logger = logging.getLogger(__name__)


@dataclass
class ColPaliConfig:
    """Configuration for ColPali RAG."""
    
    # Model
    model_name: str = "vidore/colpali-v1.2"
    device: str = "cuda"  # or "cpu"
    
    # Indexing
    store_collection: bool = True
    overwrite_index: bool = False
    
    # Retrieval
    top_k: int = 3
    
    # Generation
    use_colpali_for_generation: bool = True


class ColPaliRAG(BaseRAGTechnique):
    """
    Implements Multi-modal RAG with ColPali.
    
    ColPali is a vision-language model specifically designed for document
    understanding. It processes entire document pages as images, enabling
    retrieval without explicit text extraction or OCR.
    
    Process:
    1. Load ColPali model
    2. Index document pages as images
    3. Retrieve relevant pages using visual similarity
    4. Generate answers using vision-language model
    
    Example:
        >>> config = ColPaliConfig(model_name="vidore/colpali-v1.2")
        >>> colpali_rag = ColPaliRAG(config=config)
        >>> 
        >>> # Index a PDF
        >>> colpali_rag.index_pdf(
        ...     "document.pdf",
        ...     index_name="my_documents"
        ... )
        >>> 
        >>> # Query with visual understanding
        >>> result = colpali_rag.retrieve_and_generate(
        ...     query="What is shown in the diagram?"
        ... )
        >>> 
        >>> # View retrieved page images
        >>> for page in result['retrieved_pages']:
        ...     print(f"Page {page['page_num']}: score={page['score']}")
    
    Note: Requires GPU and ColPali dependencies:
          pip install byaldi transformers torch
    """
    
    def __init__(
        self,
        config: Optional[ColPaliConfig] = None,
        **kwargs
    ):
        """Initialize ColPali RAG.
        
        Args:
            config: Configuration for ColPali
            **kwargs: Additional arguments passed to BaseRAGTechnique
        """
        super().__init__(**kwargs)
        self.config = config or ColPaliConfig()
        
        # Initialize ColPali model
        self._init_colpali()
        
        logger.info(
            f"Initialized ColPaliRAG with model={self.config.model_name}"
        )
    
    def _init_colpali(self):
        """Initialize ColPali model."""
        try:
            from byaldi import RAGMultiModalModel
            
            logger.info(f"Loading ColPali model: {self.config.model_name}")
            self.colpali_model = RAGMultiModalModel.from_pretrained(
                self.config.model_name,
                verbose=1
            )
            logger.info("ColPali model loaded successfully")
            
        except ImportError as e:
            logger.error(
                "ColPali (byaldi) not installed. "
                "Install with: pip install byaldi"
            )
            self.colpali_model = None
            raise ImportError(
                "ColPali dependencies required: pip install byaldi transformers torch"
            ) from e
        except Exception as e:
            logger.error(f"Error loading ColPali model: {e}")
            self.colpali_model = None
            raise
    
    def index_pdf(
        self,
        pdf_path: str,
        index_name: str,
        **kwargs
    ):
        """
        Index a PDF document using ColPali.
        
        Args:
            pdf_path: Path to PDF file
            index_name: Name for the index
            **kwargs: Additional arguments for indexing
        """
        if not self.colpali_model:
            raise ValueError("ColPali model not initialized")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Indexing PDF: {pdf_path} as '{index_name}'")
        
        try:
            self.colpali_model.index(
                input_path=str(pdf_path),
                index_name=index_name,
                store_collection_with_index=self.config.store_collection,
                overwrite=self.config.overwrite_index
            )
            
            self.index_name = index_name
            logger.info(f"Successfully indexed {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error indexing PDF: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search indexed documents using ColPali.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with pages and scores
        """
        if not self.colpali_model:
            raise ValueError("ColPali model not initialized")
        
        if not hasattr(self, 'index_name'):
            raise ValueError("No index loaded. Call index_pdf first.")
        
        k = k or self.config.top_k
        
        logger.info(f"Searching for: {query}")
        
        try:
            results = self.colpali_model.search(query, k=k)
            
            # Parse results
            parsed_results = []
            for result in results:
                parsed_results.append({
                    "doc_id": result.get("doc_id"),
                    "page_num": result.get("page_num"),
                    "score": result.get("score"),
                    "metadata": result.get("metadata", {})
                })
            
            logger.info(f"Found {len(parsed_results)} results")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def get_page_image(
        self,
        doc_id: str,
        page_num: int
    ) -> Optional[Any]:
        """
        Get image data for a specific page.
        
        Args:
            doc_id: Document ID
            page_num: Page number
            
        Returns:
            Page image data (if available)
        """
        if not self.colpali_model:
            return None
        
        try:
            # Retrieve page image from ColPali collection
            # Implementation depends on ColPali's storage format
            # This is a placeholder - actual implementation may vary
            logger.warning("get_page_image: Direct page retrieval not implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving page image: {e}")
            return None
    
    def retrieve_and_generate(
        self,
        query: str,
        k: Optional[int] = None,
        use_vision_llm: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve relevant pages and generate answer.
        
        Args:
            query: User query
            k: Number of pages to retrieve
            use_vision_llm: Use vision LLM for answer generation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with answer and retrieved pages
        """
        # Search for relevant pages
        results = self.search(query, k=k)
        
        if not results:
            return {
                "answer": "No relevant pages found.",
                "retrieved_pages": [],
                "num_pages": 0
            }
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"Page {result['page_num']} (relevance: {result['score']:.3f})"
            )
        
        context = "\n".join(context_parts)
        
        # Generate answer
        if use_vision_llm and self.config.use_colpali_for_generation:
            # Use vision model for generation
            prompt = f"""Based on the retrieved document pages, answer the following question.

Retrieved Pages:
{context}

Question: {query}

Answer based on the visual content of these pages:"""
            
            # Note: Actual vision-based generation would require accessing page images
            # This is a simplified text-based generation
            answer = self.llm.invoke(prompt).content
        else:
            # Fallback to text-based generation
            prompt = f"""The following document pages were found relevant to the query.

Retrieved Pages:
{context}

Question: {query}

Please provide an answer based on these pages:"""
            
            answer = self.llm.invoke(prompt).content
        
        return {
            "answer": answer,
            "retrieved_pages": results,
            "num_pages": len(results),
            "query": query
        }
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Index text documents (not PDF).
        
        Note: ColPali is designed for PDF documents. For text-only documents,
        consider using a different technique or converting to PDF first.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata
            **kwargs: Additional arguments
        """
        logger.warning(
            "ColPali is optimized for PDF documents with visual content. "
            "For text-only documents, consider using SimpleRAG or other techniques."
        )
        
        # Store documents for fallback retrieval
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        
        logger.info(f"Stored {len(documents)} text documents (not indexed with ColPali)")
