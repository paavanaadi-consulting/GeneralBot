"""Command-line interface for GeneralBot."""
import argparse
import sys
from pathlib import Path

from assistant import ConversationalAssistant
from logger import setup_logging

logger = setup_logging()


def chat_interactive(assistant: ConversationalAssistant, use_rag: bool = True):
    """Run interactive chat session.
    
    Args:
        assistant: ConversationalAssistant instance
        use_rag: Whether to use RAG for responses
    """
    print("\n" + "="*60)
    print("GeneralBot - AI Conversational Assistant")
    print("="*60)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("Type 'help' for available commands")
    print("="*60 + "\n")
    
    session_id = f"cli_session_{Path.cwd().name}"
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using GeneralBot. Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                assistant.clear_conversation_history(session_id)
                print("\n[Conversation history cleared]")
                continue
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit - End the conversation")
                print("  clear     - Clear conversation history")
                print("  help      - Show this help message")
                continue
            
            # Process query
            print("\nAssistant: ", end="", flush=True)
            result = assistant.chat(
                query=user_input,
                session_id=session_id,
                use_rag=use_rag
            )
            
            print(result['response'])
            
            if use_rag and result['sources_used'] > 0:
                print(f"\n[Used {result['sources_used']} knowledge base sources]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit properly.")
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error(f"Error in chat: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GeneralBot - AI Conversational Assistant with LLM and RAG"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat session')
    chat_parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG and use only LLM'
    )
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into knowledge base')
    ingest_parser.add_argument(
        'path',
        type=str,
        help='Path to document file or directory'
    )
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild knowledge base from scratch')
    rebuild_parser.add_argument(
        'path',
        type=str,
        help='Path to document file or directory'
    )
    
    # Query command (single query)
    query_parser = subparsers.add_parser('query', help='Send a single query')
    query_parser.add_argument(
        'text',
        type=str,
        help='Query text'
    )
    query_parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG and use only LLM'
    )
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize assistant
    assistant = ConversationalAssistant()
    
    try:
        if args.command == 'chat':
            chat_interactive(assistant, use_rag=not args.no_rag)
        
        elif args.command == 'ingest':
            print(f"Ingesting documents from: {args.path}")
            assistant.ingest_documents(args.path)
            print("Documents ingested successfully!")
        
        elif args.command == 'rebuild':
            print(f"Rebuilding knowledge base from: {args.path}")
            assistant.rebuild_knowledge_base(args.path)
            print("Knowledge base rebuilt successfully!")
        
        elif args.command == 'query':
            result = assistant.chat(
                query=args.text,
                use_rag=not args.no_rag
            )
            print(f"\nQuery: {result['query']}")
            print(f"\nResponse: {result['response']}")
            if result['sources_used'] > 0:
                print(f"\nSources used: {result['sources_used']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
        
        elif args.command == 'server':
            print("Starting API server...")
            from main import main as run_server
            run_server()
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"CLI error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
