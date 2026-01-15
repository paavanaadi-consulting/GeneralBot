"""
Example script demonstrating the RAG Techniques package usage
"""

import os
from dotenv import load_dotenv
from rag_techniques import SimpleRAG, RAGConfig
from rag_techniques.core.config import EmbeddingProvider

# Load environment variables
load_dotenv()


def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("RAG Techniques Package - Usage Example")
    print("=" * 60)
    
    # Example 1: Basic Configuration
    print("\n1. Creating Basic RAG Configuration...")
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        n_retrieved=2,
        model_name="gpt-4",
        temperature=0.0,
    )
    print(f"✅ Config created: {config.to_dict()}")
    
    # Example 2: Initialize Simple RAG (commented out - requires PDF)
    print("\n2. Initializing Simple RAG...")
    print("   (Requires PDF file)")
    # rag = SimpleRAG(
    #     pdf_path="../data/Understanding_Climate_Change.pdf",
    #     config=config
    # )
    print("   Usage:")
    print("   rag = SimpleRAG(pdf_path='path/to/document.pdf', config=config)")
    
    # Example 3: Query the system (commented out - requires initialization)
    print("\n3. Querying the RAG System...")
    # result = rag.query("What is climate change?")
    # print(f"Answer: {result['answer']}")
    # print(f"Retrieval Time: {result['retrieval_time']:.2f}s")
    print("   Usage:")
    print("   result = rag.query('What is climate change?')")
    print("   print(result['answer'])")
    
    # Example 4: Configuration from Dictionary
    print("\n4. Creating Config from Dictionary...")
    config_dict = {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "n_retrieved": 3,
        "model_name": "gpt-4",
        "embedding_provider": "openai"
    }
    config2 = RAGConfig.from_dict(config_dict)
    print(f"✅ Config from dict: chunk_size={config2.chunk_size}")
    
    # Example 5: Updating Configuration
    print("\n5. Updating Configuration...")
    config.chunk_size = 1200
    config.n_retrieved = 4
    print(f"✅ Updated: chunk_size={config.chunk_size}, n_retrieved={config.n_retrieved}")
    
    print("\n" + "=" * 60)
    print("✨ Package is ready to use!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set your OpenAI API key in .env file")
    print("2. Prepare a PDF document")
    print("3. Run: python examples/basic_usage.py")
    print("\nFor more examples, see the README.md")


if __name__ == "__main__":
    main()
