"""
Document Augmentation: Question Generation for Enhanced Retrieval

This module implements document augmentation through question generation.
By generating relevant questions for each document chunk, the system
improves retrieval accuracy by matching user queries with similar questions.

Key Features:
- Automatic question generation from documents
- Multiple questions per document/chunk
- Enhanced embedding for better retrieval
- Configurable question count
- LLM-based question quality
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.base import BaseRAGTechnique
from ..core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentAugmentationConfig:
    """Configuration for Document Augmentation."""
    
    # Question generation
    questions_per_chunk: int = 3  # Questions to generate per chunk
    max_chunk_size: int = 1000  # Max characters per chunk for questions
    
    # LLM settings
    generation_model: str = "gpt-3.5-turbo"
    generation_temperature: float = 0.7
    
    # Augmentation strategy
    prepend_questions: bool = True  # Add questions to document text
    store_questions_separately: bool = False  # Store as separate entries


class DocumentAugmentation(BaseRAGTechnique):
    """
    Implements Document Augmentation with question generation.
    
    This technique enriches documents by generating relevant questions that
    the document could answer. When a user asks a question, the system can
    better match it to documents that have similar generated questions,
    improving retrieval accuracy.
    
    Process:
    1. Split documents into chunks
    2. Generate N questions per chunk using LLM
    3. Augment chunks with generated questions
    4. Index augmented chunks for retrieval
    5. Retrieve based on query-question similarity
    
    Example:
        >>> config = DocumentAugmentationConfig(questions_per_chunk=5)
        >>> aug_rag = DocumentAugmentation(config=config)
        >>> 
        >>> # Index with augmentation
        >>> aug_rag.index_documents_with_questions(documents)
        >>> 
        >>> # Retrieve augmented docs
        >>> result = aug_rag.retrieve_and_generate(
        ...     query="How does photosynthesis work?",
        ...     k=3
        ... )
        >>> 
        >>> # View generated questions
        >>> for doc in result['documents']:
        ...     print(doc.metadata.get('generated_questions', []))
    """
    
    def __init__(
        self,
        config: Optional[DocumentAugmentationConfig] = None,
        **kwargs
    ):
        """Initialize Document Augmentation.
        
        Args:
            config: Configuration for document augmentation
            **kwargs: Additional arguments passed to BaseRAGTechnique
        """
        super().__init__(**kwargs)
        self.config = config or DocumentAugmentationConfig()
        
        # Initialize question generation LLM
        try:
            from langchain_openai import ChatOpenAI
            self.question_llm = ChatOpenAI(
                model=self.config.generation_model,
                temperature=self.config.generation_temperature
            )
        except ImportError:
            logger.warning("ChatOpenAI not available for question generation")
            self.question_llm = None
        
        logger.info(
            f"Initialized DocumentAugmentation with "
            f"{self.config.questions_per_chunk} questions per chunk"
        )
    
    def generate_questions(
        self,
        document_text: str,
        num_questions: Optional[int] = None
    ) -> List[str]:
        """
        Generate questions that the document could answer.
        
        Args:
            document_text: Text to generate questions for
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if not self.question_llm:
            logger.warning("No question generation LLM available")
            return []
        
        num_questions = num_questions or self.config.questions_per_chunk
        
        # Truncate if too long
        if len(document_text) > self.config.max_chunk_size:
            document_text = document_text[:self.config.max_chunk_size] + "..."
        
        prompt = f"""Based on the following text, generate {num_questions} diverse questions that this text could answer.

Guidelines:
- Questions should be specific and answerable from the text
- Cover different aspects of the content
- Use varied question types (what, how, why, when, etc.)
- Make questions natural and realistic

Text:
{document_text}

Generate exactly {num_questions} questions, one per line:"""
        
        try:
            response = self.question_llm.invoke(prompt)
            questions_text = response.content.strip()
            
            # Parse questions (one per line)
            questions = [
                q.strip().lstrip('0123456789.-) ')
                for q in questions_text.split('\n')
                if q.strip()
            ]
            
            # Filter valid questions
            questions = [
                q for q in questions
                if q and len(q) > 10 and '?' in q
            ]
            
            logger.debug(f"Generated {len(questions)} questions")
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    def augment_document(
        self,
        document_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Augment a single document with generated questions.
        
        Args:
            document_text: Original document text
            metadata: Optional metadata for the document
            
        Returns:
            Dictionary with:
            - augmented_text: Document with questions prepended
            - questions: List of generated questions
            - original_text: Original document text
            - metadata: Updated metadata
        """
        # Generate questions
        questions = self.generate_questions(document_text)
        
        if not questions:
            return {
                "augmented_text": document_text,
                "questions": [],
                "original_text": document_text,
                "metadata": metadata or {}
            }
        
        # Create augmented text
        if self.config.prepend_questions:
            questions_section = "Related Questions:\n" + "\n".join(
                f"- {q}" for q in questions
            )
            augmented_text = f"{questions_section}\n\n{document_text}"
        else:
            augmented_text = document_text
        
        # Update metadata
        augmented_metadata = metadata.copy() if metadata else {}
        augmented_metadata["generated_questions"] = questions
        augmented_metadata["is_augmented"] = True
        
        return {
            "augmented_text": augmented_text,
            "questions": questions,
            "original_text": document_text,
            "metadata": augmented_metadata
        }
    
    def augment_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Augment multiple documents with questions.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of augmented document dictionaries
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        augmented_docs = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            logger.info(f"Augmenting document {i+1}/{len(documents)}")
            
            augmented = self.augment_document(doc, meta)
            augmented_docs.append(augmented)
        
        return augmented_docs
    
    def index_documents_with_questions(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Index documents with question augmentation.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            **kwargs: Additional arguments for indexing
        """
        logger.info(f"Indexing {len(documents)} documents with question augmentation")
        
        # Augment documents
        augmented_docs = self.augment_documents(documents, metadatas)
        
        # Prepare for indexing
        if self.config.store_questions_separately:
            # Create separate entries for questions
            texts_to_index = []
            metadata_to_index = []
            
            for aug_doc in augmented_docs:
                # Add augmented document
                texts_to_index.append(aug_doc["augmented_text"])
                metadata_to_index.append(aug_doc["metadata"])
                
                # Add questions as separate entries (optional)
                for question in aug_doc["questions"]:
                    texts_to_index.append(question)
                    q_meta = aug_doc["metadata"].copy()
                    q_meta["is_question"] = True
                    q_meta["source_doc"] = aug_doc["original_text"][:100]
                    metadata_to_index.append(q_meta)
        else:
            # Just use augmented texts
            texts_to_index = [doc["augmented_text"] for doc in augmented_docs]
            metadata_to_index = [doc["metadata"] for doc in augmented_docs]
        
        # Index using base class method
        self.index_documents(texts_to_index, metadata_to_index, **kwargs)
        
        logger.info(
            f"Indexed {len(texts_to_index)} entries "
            f"({len(documents)} original documents)"
        )
    
    def retrieve_and_generate(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve augmented documents and generate response.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - answer: Generated response
            - documents: Retrieved documents
            - questions: Questions from retrieved documents
            - num_augmented: Number of augmented documents
        """
        if not self.retriever:
            raise ValueError("No retriever initialized. Call index_documents first.")
        
        # Retrieve documents
        logger.info(f"Retrieving {k} augmented documents")
        docs = self.retriever.invoke(query)[:k]
        
        # Extract questions from metadata
        all_questions = []
        num_augmented = 0
        
        for doc in docs:
            if doc.metadata.get("is_augmented"):
                num_augmented += 1
                questions = doc.metadata.get("generated_questions", [])
                all_questions.extend(questions)
        
        # Generate answer
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Based on the following context (which includes relevant questions and answers), please answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.invoke(prompt).content
        
        return {
            "answer": answer,
            "documents": docs,
            "questions": all_questions,
            "num_augmented": num_augmented,
            "total_retrieved": len(docs)
        }
