"""Example usage of GeneralBot."""
import os
from assistant import ConversationalAssistant
from logger import setup_logging

# Setup logging
logger = setup_logging()


def example_basic_chat():
    """Example: Basic chat without RAG."""
    print("\n" + "="*60)
    print("Example 1: Basic Chat (No RAG)")
    print("="*60)
    
    assistant = ConversationalAssistant()
    
    result = assistant.chat(
        query="What is artificial intelligence?",
        use_rag=False
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Processing time: {result['processing_time']:.2f}s")


def example_rag_chat():
    """Example: Chat with RAG."""
    print("\n" + "="*60)
    print("Example 2: Chat with RAG")
    print("="*60)
    
    assistant = ConversationalAssistant()
    
    # First, ingest documents
    print("\nIngesting documents...")
    assistant.ingest_documents("./data/documents")
    
    # Now ask questions about the documents
    queries = [
        "What are your business hours?",
        "What services do you offer?",
        "How much does the Enterprise plan cost?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        result = assistant.chat(query=query, use_rag=True)
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Sources used: {result['sources_used']}")
        print(f"Processing time: {result['processing_time']:.2f}s")


def example_conversation_history():
    """Example: Multi-turn conversation with history."""
    print("\n" + "="*60)
    print("Example 3: Multi-turn Conversation")
    print("="*60)
    
    assistant = ConversationalAssistant()
    session_id = "example_session"
    
    # Load documents first
    assistant.ingest_documents("./data/documents")
    
    queries = [
        "What is GeneralBot?",
        "What are its key features?",
        "How can I get started with it?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}")
        result = assistant.chat(
            query=query,
            session_id=session_id,
            use_rag=True
        )
        print(f"User: {query}")
        print(f"Assistant: {result['response']}")
    
    # Show conversation history
    print(f"\n{'='*60}")
    print("Conversation History:")
    history = assistant.get_conversation_history(session_id)
    for msg in history:
        role = msg['role'].upper()
        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"\n{role}: {content}")


def example_document_upload():
    """Example: Adding documents to knowledge base."""
    print("\n" + "="*60)
    print("Example 4: Document Upload")
    print("="*60)
    
    assistant = ConversationalAssistant()
    
    # Check if documents directory exists
    docs_dir = "./data/documents"
    if os.path.exists(docs_dir):
        print(f"\nIngesting documents from: {docs_dir}")
        assistant.ingest_documents(docs_dir)
        print("Documents ingested successfully!")
    else:
        print(f"\nDocuments directory not found: {docs_dir}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("GeneralBot Usage Examples")
    print("="*60)
    print("\nNote: Make sure OPENAI_API_KEY is set in .env file")
    print("      and documents are available in ./data/documents/")
    
    try:
        # Example 1: Basic chat
        # Uncomment to run if you have API key set
        # example_basic_chat()
        
        # Example 2: RAG chat
        # Uncomment to run if you have API key set
        # example_rag_chat()
        
        # Example 3: Multi-turn conversation
        # Uncomment to run if you have API key set
        # example_conversation_history()
        
        # Example 4: Document upload (safe to run without API key)
        example_document_upload()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60)
        print("\nTo run the other examples:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Uncomment the example functions in examples.py")
        print("3. Run: python examples.py")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        logger.error(f"Error in examples: {str(e)}")


if __name__ == "__main__":
    main()
