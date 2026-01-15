"""
Helper utility functions for RAG systems
"""

import textwrap
from typing import List, Any
import fitz  # PyMuPDF


def replace_t_with_space(list_of_documents: List[Any]) -> List[Any]:
    """
    Replaces all tab characters with spaces in document content
    
    Args:
        list_of_documents: List of document objects with page_content attribute
        
    Returns:
        Modified list of documents with tabs replaced by spaces
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents


def text_wrap(text: str, width: int = 120) -> str:
    """
    Wraps text to the specified width
    
    Args:
        text: Input text to wrap
        width: Width at which to wrap text
        
    Returns:
        Wrapped text
    """
    return textwrap.fill(text, width=width)


def show_context(context: List[str]) -> None:
    """
    Display context items in a formatted way
    
    Args:
        context: List of context strings to display
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


def read_pdf_to_string(path: str) -> str:
    """
    Read PDF document and return content as a string
    
    Args:
        path: File path to PDF document
        
    Returns:
        Concatenated text content of all pages
    """
    doc = fitz.open(path)
    content = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        content += page.get_text()
    return content


def chunks_to_string(chunks: List[Any]) -> str:
    """
    Convert document chunks to single string
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Concatenated string of all chunks
    """
    return " ".join([chunk.page_content for chunk in chunks])
