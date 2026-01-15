"""
Contextual Compression Techniques

This module implements contextual compression to reduce retrieved context
while maintaining relevance.
"""

from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.base import BaseRAG
from ..core.config import RAGConfig


class CompressedDocument(BaseModel):
    """Model for compressed document output."""
    compressed_content: str = Field(description="The compressed, relevant content")
    relevance_explanation: str = Field(description="Why this content is relevant")


class ContextualCompressionRAG(BaseRAG):
    """
    RAG with contextual compression.
    
    Reduces the size of retrieved documents by extracting only
    the most relevant portions for the query, improving:
    1. Token efficiency
    2. Context relevance
    3. Answer accuracy
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        content: Optional[str] = None,
        config: Optional[RAGConfig] = None,
        compression_ratio: float = 0.5
    ):
        """
        Initialize contextual compression RAG.
        
        Args:
            pdf_path: Path to PDF document
            content: Text content
            config: RAG configuration
            compression_ratio: Target compression ratio (0-1)
        """
        super().__init__(pdf_path, content, config)
        self.compression_ratio = compression_ratio
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens
        )
    
    def compress_document(
        self,
        query: str,
        document: str,
        max_length: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Compress a document to only relevant portions.
        
        Args:
            query: The search query
            document: Document content to compress
            max_length: Maximum length of compressed content
            
        Returns:
            Dictionary with compressed content and explanation
        """
        if max_length is None:
            max_length = int(len(document) * self.compression_ratio)
        
        prompt = PromptTemplate(
            input_variables=["query", "document", "max_length"],
            template="""
            You are an expert at extracting relevant information.
            
            Given the following query and document, extract only the portions
            that are directly relevant to answering the query.
            
            Query: {query}
            
            Document:
            {document}
            
            Requirements:
            - Keep only content directly relevant to the query
            - Maintain key facts, figures, and context
            - Target length: approximately {max_length} characters
            - Preserve readability and coherence
            
            Provide:
            1. The compressed, relevant content
            2. Brief explanation of what makes this content relevant
            """
        )
        
        chain = prompt | self.llm.with_structured_output(CompressedDocument)
        
        try:
            result = chain.invoke({
                "query": query,
                "document": document,
                "max_length": max_length
            })
            
            return {
                "compressed_content": result.compressed_content,
                "relevance_explanation": result.relevance_explanation,
                "original_length": len(document),
                "compressed_length": len(result.compressed_content),
                "compression_ratio": len(result.compressed_content) / len(document)
            }
        except Exception as e:
            print(f"Error compressing document: {e}")
            # Fallback: return truncated document
            return {
                "compressed_content": document[:max_length],
                "relevance_explanation": "Fallback truncation used",
                "original_length": len(document),
                "compressed_length": min(len(document), max_length),
                "compression_ratio": min(1.0, max_length / len(document))
            }
    
    def compress_documents(
        self,
        query: str,
        documents: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Compress multiple documents.
        
        Args:
            query: The search query
            documents: List of documents to compress
            
        Returns:
            List of compression results
        """
        compressed_docs = []
        
        for doc in documents:
            result = self.compress_document(query, doc.page_content)
            result["metadata"] = doc.metadata
            compressed_docs.append(result)
        
        return compressed_docs
    
    def filter_relevant_sentences(
        self,
        query: str,
        document: str,
        threshold: float = 0.5
    ) -> str:
        """
        Filter document to only relevant sentences.
        
        Args:
            query: The search query
            document: Document content
            threshold: Relevance threshold (0-1)
            
        Returns:
            Filtered document with relevant sentences
        """
        from langchain_openai import OpenAIEmbeddings
        import numpy as np
        
        embeddings = OpenAIEmbeddings()
        
        # Split into sentences
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        
        if not sentences:
            return document
        
        # Embed query and sentences
        query_embedding = np.array(embeddings.embed_query(query))
        sentence_embeddings = np.array(embeddings.embed_documents(sentences))
        
        # Calculate similarities
        similarities = []
        for sent_emb in sentence_embeddings:
            similarity = np.dot(query_embedding, sent_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(sent_emb)
            )
            similarities.append(similarity)
        
        # Filter by threshold
        relevant_sentences = [
            sent for sent, sim in zip(sentences, similarities)
            if sim >= threshold
        ]
        
        if not relevant_sentences:
            # If no sentences pass threshold, return top 3
            top_indices = np.argsort(similarities)[-3:]
            relevant_sentences = [sentences[i] for i in top_indices]
        
        return '. '.join(relevant_sentences) + '.'
    
    def query(
        self,
        query_text: str,
        compression_method: str = "llm",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with contextual compression.
        
        Args:
            query_text: The search query
            compression_method: Method - 'llm' or 'sentence_filter'
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with answer and compression details
        """
        # Retrieve documents
        docs = self.retriever.get_relevant_documents(query_text)
        
        if compression_method == "llm":
            # Compress using LLM
            compressed_docs = self.compress_documents(query_text, docs)
            context = [cd["compressed_content"] for cd in compressed_docs]
            compression_stats = compressed_docs
            
        elif compression_method == "sentence_filter":
            # Filter relevant sentences
            threshold = kwargs.get("threshold", 0.5)
            context = []
            compression_stats = []
            
            for doc in docs:
                filtered = self.filter_relevant_sentences(
                    query_text,
                    doc.page_content,
                    threshold
                )
                context.append(filtered)
                compression_stats.append({
                    "original_length": len(doc.page_content),
                    "compressed_length": len(filtered),
                    "compression_ratio": len(filtered) / len(doc.page_content)
                })
        else:
            raise ValueError(f"Unknown compression method: {compression_method}")
        
        # Generate answer using compressed context
        answer_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Answer the following question based on the provided context.
            
            Question: {query}
            
            Context:
            {context}
            
            Answer:
            """
        )
        
        chain = answer_prompt | self.llm
        result = chain.invoke({
            "query": query_text,
            "context": "\n\n".join(context)
        })
        
        # Calculate overall compression stats
        if compression_method == "llm":
            total_original = sum(cs["original_length"] for cs in compression_stats)
            total_compressed = sum(cs["compressed_length"] for cs in compression_stats)
        else:
            total_original = sum(cs["original_length"] for cs in compression_stats)
            total_compressed = sum(cs["compressed_length"] for cs in compression_stats)
        
        avg_compression_ratio = total_compressed / total_original if total_original > 0 else 0
        
        return {
            "query": query_text,
            "answer": result.content,
            "method": compression_method,
            "context": context,
            "compression_stats": compression_stats,
            "total_original_length": total_original,
            "total_compressed_length": total_compressed,
            "avg_compression_ratio": round(avg_compression_ratio, 3),
            "token_savings": round((1 - avg_compression_ratio) * 100, 1),
            "num_docs": len(docs)
        }
