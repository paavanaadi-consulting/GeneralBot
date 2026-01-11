"""Conversational assistant with RAG capabilities."""
import logging
from typing import List, Dict, Optional
from datetime import datetime

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_manager import LLMManager

logger = logging.getLogger(__name__)


class ConversationalAssistant:
    """AI-powered conversational assistant with RAG."""
    
    def __init__(self):
        """Initialize the conversational assistant."""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        logger.info("Conversational Assistant initialized")
    
    def ingest_documents(self, source_path: str):
        """Ingest documents into the knowledge base.
        
        Args:
            source_path: Path to document file or directory
        """
        logger.info(f"Ingesting documents from {source_path}")
        
        # Process documents
        documents = self.document_processor.process_documents(source_path)
        
        if not documents:
            logger.warning("No documents processed")
            return
        
        # Load existing vector store or create new one
        existing_store = self.vector_store_manager.load_vector_store()
        
        if existing_store:
            # Add to existing store
            self.vector_store_manager.add_documents(documents)
            logger.info(f"Added {len(documents)} document chunks to existing knowledge base")
        else:
            # Create new store
            self.vector_store_manager.create_vector_store(documents)
            logger.info(f"Created knowledge base with {len(documents)} document chunks")
    
    def retrieve_relevant_context(self, query: str, k: Optional[int] = None) -> List:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.vector_store_manager.similarity_search(query, k=k)
    
    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_rag: bool = True
    ) -> Dict[str, any]:
        """Process a user query and generate a response.
        
        Args:
            query: User query
            session_id: Session identifier for conversation history
            use_rag: Whether to use RAG for context retrieval
            
        Returns:
            Dictionary containing response and metadata
        """
        logger.info(f"Processing query: {query[:100]}...")
        start_time = datetime.now()
        
        # Get or create session
        if session_id is None:
            session_id = f"session_{datetime.now().timestamp()}"
        
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # Retrieve relevant context if RAG is enabled
        retrieved_documents = []
        if use_rag:
            retrieved_documents = self.retrieve_relevant_context(query)
            logger.info(f"Retrieved {len(retrieved_documents)} relevant documents")
        
        # Get conversation history for this session
        history = self.conversation_history[session_id]
        
        # Generate response
        if use_rag:
            response = self.llm_manager.generate_response_with_rag(
                query=query,
                retrieved_documents=retrieved_documents,
                conversation_history=history
            )
        else:
            response = self.llm_manager.generate_response(
                query=query,
                conversation_history=history
            )
        
        # Update conversation history
        self.conversation_history[session_id].append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })
        self.conversation_history[session_id].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response metadata
        result = {
            'query': query,
            'response': response,
            'session_id': session_id,
            'processing_time': processing_time,
            'sources_used': len(retrieved_documents),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Response generated in {processing_time:.2f} seconds")
        return result
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages
        """
        return self.conversation_history.get(session_id, [])
    
    def rebuild_knowledge_base(self, source_path: str):
        """Rebuild the knowledge base from scratch.
        
        Args:
            source_path: Path to document file or directory
        """
        logger.info("Rebuilding knowledge base")
        
        # Delete existing vector store
        self.vector_store_manager.delete_vector_store()
        
        # Ingest documents
        self.ingest_documents(source_path)
        
        logger.info("Knowledge base rebuilt successfully")
