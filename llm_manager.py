"""LLM integration for conversation generation."""
import logging
from typing import Optional, List, Dict
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Manage Large Language Model interactions."""
    
    def __init__(self):
        """Initialize LLM manager."""
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration.
        
        Returns:
            ChatOpenAI or AzureChatOpenAI instance
        """
        # Check if Azure OpenAI is configured
        if settings.azure_openai_api_key and settings.azure_openai_endpoint:
            logger.info("Initializing Azure OpenAI")
            return AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                openai_api_key=settings.azure_openai_api_key,
                azure_deployment=settings.azure_openai_deployment_name,
                openai_api_version=settings.azure_openai_api_version,
                temperature=0.7,
                max_tokens=1000
            )
        else:
            logger.info("Initializing OpenAI")
            return ChatOpenAI(
                model=settings.openai_model,
                openai_api_key=settings.openai_api_key,
                temperature=0.7,
                max_tokens=1000
            )
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a response using the LLM.
        
        Args:
            query: User query
            context: Retrieved context from RAG (optional)
            system_prompt: System prompt to guide LLM behavior (optional)
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            Generated response string
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        else:
            default_system_prompt = """You are a helpful AI assistant that provides accurate, 
context-aware responses. You leverage your knowledge and provided context to give 
comprehensive answers. When using provided context, cite it naturally in your responses. 
If you're not sure about something, acknowledge the uncertainty."""
            messages.append(SystemMessage(content=default_system_prompt))
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
        
        # Add current query with context if available
        if context:
            enhanced_query = f"""Context from knowledge base:
{context}

User question: {query}

Please provide a helpful response based on the context and your knowledge."""
            messages.append(HumanMessage(content=enhanced_query))
        else:
            messages.append(HumanMessage(content=query))
        
        try:
            response = self.llm.invoke(messages)
            logger.info(f"Generated response for query: {query[:50]}...")
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def generate_response_with_rag(
        self,
        query: str,
        retrieved_documents: List,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate response using RAG (Retrieval-Augmented Generation).
        
        Args:
            query: User query
            retrieved_documents: Documents retrieved from vector store
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            Generated response string
        """
        # Build context from retrieved documents
        if retrieved_documents:
            context_parts = []
            for i, doc in enumerate(retrieved_documents, 1):
                context_parts.append(f"[Source {i}]\n{doc.page_content}\n")
            context = "\n".join(context_parts)
        else:
            context = None
        
        system_prompt = """You are an intelligent conversational assistant with access to 
organization-specific knowledge. Your role is to provide accurate, context-aware responses 
by combining your reasoning capabilities with information from the company's internal 
documents and databases.

Guidelines:
1. Use the provided context to answer questions accurately
2. If the context doesn't contain the answer, use your general knowledge but make it clear
3. Cite sources naturally when using information from the context
4. Maintain a professional and helpful tone
5. If you're unsure, acknowledge it rather than making up information
6. Provide 24/7 automated support with consistent quality"""
        
        return self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )
