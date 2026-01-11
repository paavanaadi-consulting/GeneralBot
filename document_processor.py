"""Document processor for ingesting and chunking documents."""
import os
from typing import List, Dict
from pathlib import Path
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document

from config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG system."""
    
    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.txt', '.text']:
                loader = TextLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            documents = loader.load()
            logger.info(f"Loaded document: {file_path} with {len(documents)} pages/sections")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all loaded Document objects
        """
        all_documents = []
        supported_extensions = ['.pdf', '.txt', '.text', '.docx', '.doc', '.md', '.markdown']
        
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents = self.load_document(str(file_path))
                all_documents.extend(documents)
        
        logger.info(f"Loaded {len(all_documents)} documents from {directory_path}")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunked_documents = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
        return chunked_documents
    
    def process_documents(self, source_path: str) -> List[Document]:
        """Process documents from a file or directory.
        
        Args:
            source_path: Path to file or directory
            
        Returns:
            List of processed and chunked Document objects
        """
        path = Path(source_path)
        
        if path.is_file():
            documents = self.load_document(str(path))
        elif path.is_dir():
            documents = self.load_documents_from_directory(str(path))
        else:
            logger.error(f"Invalid path: {source_path}")
            return []
        
        if not documents:
            logger.warning(f"No documents loaded from {source_path}")
            return []
        
        chunked_documents = self.chunk_documents(documents)
        return chunked_documents
