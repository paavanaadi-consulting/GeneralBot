"""
Multi-modal RAG with Image Captioning

This module implements multi-modal RAG that processes both text and images
from documents, using image captioning to make visual content searchable.

Key Features:
- PDF text and image extraction
- Image captioning using vision models
- Combined text + image retrieval
- Multi-modal embeddings
- Visual question answering

Note: Requires additional dependencies:
- pymupdf (for PDF processing)
- PIL (for image handling)
- google-generativeai or openai (for image captioning)
"""

import logging
import os
import io
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..core.base import BaseRAGTechnique

logger = logging.getLogger(__name__)


@dataclass
class MultiModalCaptioningConfig:
    """Configuration for Multi-modal RAG with Captioning."""
    
    # Image processing
    extract_images: bool = True
    max_image_size: Tuple[int, int] = (1024, 1024)
    image_quality: int = 85
    
    # Captioning
    caption_model: str = "gpt-4-vision-preview"  # or "gemini-pro-vision"
    caption_prompt: str = "Describe this image in detail, focusing on key information."
    max_caption_length: int = 500
    
    # Integration
    prepend_captions: bool = True
    store_images: bool = False
    image_storage_path: Optional[str] = None


class MultiModalCaptioningRAG(BaseRAGTechnique):
    """
    Implements Multi-modal RAG with image captioning.
    
    This technique extracts both text and images from documents (PDFs),
    generates captions for images using vision models, and enables
    retrieval across both modalities.
    
    Process:
    1. Extract text and images from PDF
    2. Generate captions for each image using vision LLM
    3. Combine text chunks with relevant image captions
    4. Index combined content for retrieval
    5. Answer queries using both text and visual information
    
    Example:
        >>> config = MultiModalCaptioningConfig(
        ...     caption_model="gpt-4-vision-preview",
        ...     extract_images=True
        ... )
        >>> mm_rag = MultiModalCaptioningRAG(config=config)
        >>> 
        >>> # Process PDF with images
        >>> mm_rag.process_pdf("document.pdf")
        >>> 
        >>> # Query about visual content
        >>> result = mm_rag.retrieve_and_generate(
        ...     query="What does the diagram show?"
        ... )
    
    Note: Requires vision model API access (GPT-4V or Gemini Pro Vision)
    """
    
    def __init__(
        self,
        config: Optional[MultiModalCaptioningConfig] = None,
        **kwargs
    ):
        """Initialize Multi-modal Captioning RAG.
        
        Args:
            config: Configuration for multi-modal captioning
            **kwargs: Additional arguments passed to BaseRAGTechnique
        """
        super().__init__(**kwargs)
        self.config = config or MultiModalCaptioningConfig()
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize vision model
        self._init_vision_model()
        
        logger.info(
            f"Initialized MultiModalCaptioningRAG with "
            f"model={self.config.caption_model}"
        )
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError:
            logger.error("PyMuPDF not installed: pip install pymupdf")
            self.fitz = None
        
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            logger.error("PIL not installed: pip install Pillow")
            self.Image = None
    
    def _init_vision_model(self):
        """Initialize vision model for image captioning."""
        if "gpt-4" in self.config.caption_model or "gpt-4o" in self.config.caption_model:
            try:
                from langchain_openai import ChatOpenAI
                self.vision_model = ChatOpenAI(
                    model=self.config.caption_model,
                    max_tokens=self.config.max_caption_length
                )
                logger.info("Initialized GPT-4V for image captioning")
            except ImportError:
                logger.error("OpenAI not available")
                self.vision_model = None
        
        elif "gemini" in self.config.caption_model:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.vision_model = genai.GenerativeModel(self.config.caption_model)
                logger.info("Initialized Gemini for image captioning")
            except ImportError:
                logger.error("Google Generative AI not available")
                self.vision_model = None
        else:
            logger.warning(f"Unknown vision model: {self.config.caption_model}")
            self.vision_model = None
    
    def extract_pdf_content(
        self,
        pdf_path: str
    ) -> Dict[str, Any]:
        """
        Extract text and images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with:
            - text_content: List of text chunks
            - images: List of image data
            - page_mapping: Page numbers for each element
        """
        if not self.fitz:
            raise ImportError("PyMuPDF required: pip install pymupdf")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Extracting content from {pdf_path}")
        
        text_content = []
        images = []
        
        with self.fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                # Extract text
                text = page.get_text().strip()
                if text:
                    text_content.append({
                        "text": text,
                        "page": page_num,
                        "type": "text"
                    })
                
                # Extract images if enabled
                if self.config.extract_images:
                    page_images = page.get_images(full=True)
                    
                    for img_index, img in enumerate(page_images):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            images.append({
                                "data": image_bytes,
                                "page": page_num,
                                "index": img_index,
                                "type": "image"
                            })
                            
                        except Exception as e:
                            logger.warning(f"Failed to extract image on page {page_num}: {e}")
        
        logger.info(
            f"Extracted {len(text_content)} text blocks and "
            f"{len(images)} images"
        )
        
        return {
            "text_content": text_content,
            "images": images,
            "num_pages": len(doc)
        }
    
    def generate_caption(
        self,
        image_data: bytes
    ) -> str:
        """
        Generate caption for an image.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Caption text
        """
        if not self.vision_model:
            logger.warning("No vision model available")
            return "[Image: Caption not available]"
        
        try:
            # Convert bytes to PIL Image
            if self.Image:
                image = self.Image.open(io.BytesIO(image_data))
                
                # Resize if needed
                if image.size[0] > self.config.max_image_size[0] or \
                   image.size[1] > self.config.max_image_size[1]:
                    image.thumbnail(self.config.max_image_size)
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
            else:
                img_byte_arr = image_data
            
            # Generate caption (implementation varies by model)
            if "gpt-4" in self.config.caption_model:
                # OpenAI Vision API
                import base64
                base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
                
                response = self.vision_model.invoke([
                    {
                        "type": "text",
                        "text": self.config.caption_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ])
                
                caption = response.content
            
            elif "gemini" in self.config.caption_model:
                # Gemini Vision API
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(img_byte_arr))
                
                response = self.vision_model.generate_content([
                    self.config.caption_prompt,
                    img
                ])
                caption = response.text
            
            else:
                caption = "[Image: Caption generation not supported]"
            
            logger.debug(f"Generated caption: {caption[:100]}...")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"[Image: Caption generation failed - {str(e)}]"
    
    def process_pdf(
        self,
        pdf_path: str,
        **kwargs
    ):
        """
        Process PDF with text and images, generating captions.
        
        Args:
            pdf_path: Path to PDF file
            **kwargs: Additional arguments for indexing
        """
        # Extract content
        content = self.extract_pdf_content(pdf_path)
        
        # Generate captions for images
        logger.info(f"Generating captions for {len(content['images'])} images...")
        for img_data in content['images']:
            caption = self.generate_caption(img_data['data'])
            img_data['caption'] = caption
        
        # Combine text and image captions
        combined_content = []
        metadatas = []
        
        # Add text content
        for text_item in content['text_content']:
            combined_content.append(text_item['text'])
            metadatas.append({
                "type": "text",
                "page": text_item['page'],
                "source": pdf_path
            })
        
        # Add image captions
        for img_item in content['images']:
            caption_text = f"[IMAGE from page {img_item['page']}]: {img_item['caption']}"
            
            if self.config.prepend_captions:
                combined_content.append(caption_text)
                metadatas.append({
                    "type": "image_caption",
                    "page": img_item['page'],
                    "image_index": img_item['index'],
                    "source": pdf_path
                })
        
        # Index combined content
        logger.info(f"Indexing {len(combined_content)} items...")
        self.index_documents(combined_content, metadatas, **kwargs)
        
        logger.info(
            f"Processed {content['num_pages']} pages, "
            f"{len(content['text_content'])} text blocks, "
            f"{len(content['images'])} images"
        )
    
    def retrieve_and_generate(
        self,
        query: str,
        k: int = 5,
        include_image_captions: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve multi-modal content and generate response.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            include_image_captions: Whether to include image captions
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.retriever:
            raise ValueError("No retriever initialized. Call process_pdf first.")
        
        # Retrieve documents
        docs = self.retriever.invoke(query)[:k]
        
        # Separate text and image captions
        text_docs = []
        image_captions = []
        
        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "image_caption" and include_image_captions:
                image_captions.append(doc)
            else:
                text_docs.append(doc)
        
        # Build context
        context_parts = []
        
        if text_docs:
            context_parts.append("Text Content:")
            for doc in text_docs:
                context_parts.append(f"- {doc.page_content}")
        
        if image_captions:
            context_parts.append("\nVisual Content:")
            for doc in image_captions:
                context_parts.append(f"- {doc.page_content}")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based on the following text and visual content, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.invoke(prompt).content
        
        return {
            "answer": answer,
            "text_documents": text_docs,
            "image_captions": image_captions,
            "num_text": len(text_docs),
            "num_images": len(image_captions),
            "total_retrieved": len(docs)
        }
