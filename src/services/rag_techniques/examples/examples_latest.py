"""
Advanced RAG Techniques - Latest Additions Examples

This script demonstrates the 3 newest RAG techniques:
1. Reliable RAG - Document quality validation
2. Dartboard RAG - Relevance-diversity balance
3. Document Augmentation - Question generation

These examples show how to use these cutting-edge techniques for
production-grade RAG systems.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def example_reliable_rag():
    """
    Example: Reliable RAG with document grading
    
    Use case: When you need to ensure high-quality context and want to
    filter out irrelevant documents before generation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Reliable RAG - Document Quality Validation")
    print("=" * 80)
    
    from rag_techniques import ReliableRAG, ReliableRAGConfig
    
    # Configure reliability settings
    config = ReliableRAGConfig(
        relevance_threshold=0.7,  # Minimum relevance score
        min_relevant_docs=2,      # Need at least 2 relevant docs
        fallback_to_llm=True,     # Use LLM without context if needed
        grading_temperature=0.0   # Deterministic grading
    )
    
    print(f"\nConfiguration:")
    print(f"  - Relevance threshold: {config.relevance_threshold}")
    print(f"  - Minimum relevant docs: {config.min_relevant_docs}")
    print(f"  - Fallback enabled: {config.fallback_to_llm}")
    
    # Initialize
    reliable_rag = ReliableRAG(config=config)
    
    # Sample documents
    documents = [
        "RAG combines retrieval and generation. It retrieves relevant documents and uses them to generate responses.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "RAG systems are powerful for question answering because they ground responses in retrieved documents.",
        "Cats are domesticated mammals that are popular pets worldwide.",
        "Vector databases store embeddings for efficient similarity search in RAG systems."
    ]
    
    print(f"\nIndexing {len(documents)} documents...")
    reliable_rag.index_documents(documents)
    
    # Query
    query = "What is RAG and how does it work?"
    print(f"\nQuery: {query}")
    
    # Retrieve and generate with reliability checking
    result = reliable_rag.retrieve_and_generate(query, k=4)
    
    print(f"\n📊 Reliability Metrics:")
    print(f"  - Total retrieved: {result['num_retrieved']}")
    print(f"  - Relevant docs: {result['num_relevant']}")
    print(f"  - Reliability score: {result['reliability_score']:.2%}")
    print(f"  - Fallback used: {result['fallback_used']}")
    
    print(f"\n📝 Document Gradings:")
    for i, grading in enumerate(result['gradings'][:3], 1):
        print(f"  Doc {i}:")
        print(f"    - Relevant: {grading['is_relevant']}")
        print(f"    - Confidence: {grading['confidence']:.2f}")
        print(f"    - Reason: {grading['explanation'][:80]}...")
    
    print(f"\n💡 Answer:")
    print(f"  {result['answer'][:200]}...")
    
    print("\n✅ Key Benefit: Only uses relevant documents, improving answer quality")


def example_dartboard_rag():
    """
    Example: Dartboard RAG with diversity optimization
    
    Use case: When working with dense knowledge bases where documents
    may overlap, and you want diverse yet relevant results.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Dartboard RAG - Balanced Relevance & Diversity")
    print("=" * 80)
    
    from rag_techniques import DartboardRAG, DartboardConfig
    
    # Configure diversity settings
    config = DartboardConfig(
        relevance_weight=0.6,    # 60% weight on relevance
        diversity_weight=0.4,    # 40% weight on diversity
        initial_k=10,            # Retrieve 10 candidates
        final_k=3                # Return 3 diverse docs
    )
    
    print(f"\nConfiguration:")
    print(f"  - Relevance weight: {config.relevance_weight}")
    print(f"  - Diversity weight: {config.diversity_weight}")
    print(f"  - Initial candidates: {config.initial_k}")
    print(f"  - Final selection: {config.final_k}")
    
    # Initialize
    dartboard = DartboardRAG(config=config)
    
    # Sample documents with overlapping content
    documents = [
        "RAG systems combine retrieval and generation for better answers.",
        "Retrieval-augmented generation merges document retrieval with LLM generation.",
        "RAG techniques retrieve documents and use them to augment LLM responses.",
        "Vector databases enable efficient semantic search for RAG applications.",
        "Embeddings are numerical representations of text used in vector search.",
        "FAISS is a popular library for similarity search in high-dimensional spaces.",
        "Semantic chunking splits documents based on meaning rather than fixed sizes.",
        "Hierarchical RAG uses summaries and detailed chunks in a two-tier structure.",
        "Query transformation rewrites queries for better retrieval results.",
    ]
    
    print(f"\nIndexing {len(documents)} documents (with overlapping content)...")
    dartboard.index_documents(documents)
    
    # Query
    query = "Explain RAG systems and their components"
    print(f"\nQuery: {query}")
    
    # Retrieve with diversity
    result = dartboard.retrieve_with_diversity(query, k=3)
    
    print(f"\n📊 Diversity Metrics:")
    print(f"  - Diversity score: {result['diversity_score']:.3f} (0=identical, 1=completely different)")
    print(f"  - Selected documents: {len(result['documents'])}")
    
    print(f"\n📄 Retrieved Documents (diverse and relevant):")
    for i, (doc, score) in enumerate(zip(result['documents'], result['scores']), 1):
        print(f"\n  Doc {i} (relevance: {score:.3f}):")
        print(f"    {doc.page_content[:100]}...")
    
    # Compare with standard retrieval
    print("\n🔍 Comparison with Standard Retrieval:")
    standard_docs = dartboard.retriever.invoke(query)[:3]
    print(f"  Standard retrieval might return similar documents:")
    for i, doc in enumerate(standard_docs, 1):
        print(f"    - {doc.page_content[:60]}...")
    
    print("\n✅ Key Benefit: Avoids redundant information, provides comprehensive coverage")


def example_document_augmentation():
    """
    Example: Document Augmentation with question generation
    
    Use case: When you want to improve retrieval by matching user queries
    to similar questions your documents can answer.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Document Augmentation - Question Generation")
    print("=" * 80)
    
    from rag_techniques import DocumentAugmentation, DocumentAugmentationConfig
    
    # Configure augmentation settings
    config = DocumentAugmentationConfig(
        questions_per_chunk=3,       # Generate 3 questions per chunk
        prepend_questions=True,      # Add questions to document text
        generation_temperature=0.7   # Creative question generation
    )
    
    print(f"\nConfiguration:")
    print(f"  - Questions per chunk: {config.questions_per_chunk}")
    print(f"  - Prepend to document: {config.prepend_questions}")
    print(f"  - Generation temperature: {config.generation_temperature}")
    
    # Initialize
    aug_rag = DocumentAugmentation(config=config)
    
    # Sample documents
    documents = [
        """RAG (Retrieval-Augmented Generation) is a technique that combines information 
        retrieval with text generation. It first retrieves relevant documents from a 
        knowledge base, then uses those documents as context for a language model to 
        generate more accurate and grounded responses.""",
        
        """Vector embeddings are dense numerical representations of text that capture 
        semantic meaning. They enable similarity search by computing distances in 
        high-dimensional space. Common embedding models include OpenAI's text-embedding-ada-002 
        and sentence-transformers.""",
        
        """Semantic chunking is a technique for splitting documents based on meaning 
        rather than arbitrary token counts. It identifies natural breakpoints in text 
        using embedding similarity, preserving context and improving retrieval accuracy."""
    ]
    
    print(f"\n📝 Augmenting {len(documents)} documents with questions...")
    
    # Show augmentation for one document
    print(f"\nOriginal document excerpt:")
    print(f"  {documents[0][:100]}...")
    
    augmented = aug_rag.augment_document(documents[0])
    
    print(f"\n❓ Generated questions:")
    for i, q in enumerate(augmented['questions'], 1):
        print(f"  {i}. {q}")
    
    print(f"\n📄 Augmented document (first 200 chars):")
    print(f"  {augmented['augmented_text'][:200]}...")
    
    # Index with augmentation
    print(f"\nIndexing documents with question augmentation...")
    aug_rag.index_documents_with_questions(documents)
    
    # Query
    query = "How does semantic meaning help in document splitting?"
    print(f"\nQuery: {query}")
    
    # Retrieve and generate
    result = aug_rag.retrieve_and_generate(query, k=2)
    
    print(f"\n📊 Retrieval Results:")
    print(f"  - Total retrieved: {result['total_retrieved']}")
    print(f"  - Augmented docs: {result['num_augmented']}")
    print(f"  - Questions found: {len(result['questions'])}")
    
    print(f"\n❓ Relevant questions from retrieved docs:")
    for i, q in enumerate(result['questions'][:3], 1):
        print(f"  {i}. {q}")
    
    print(f"\n💡 Answer:")
    print(f"  {result['answer'][:200]}...")
    
    print("\n✅ Key Benefit: Better query-document matching through question similarity")


def example_combining_techniques():
    """
    Example: Combining multiple advanced techniques
    
    Shows how to chain techniques for maximum effectiveness.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Combining Multiple Advanced Techniques")
    print("=" * 80)
    
    print("\n🔗 Technique Combination Strategies:\n")
    
    print("Strategy 1: Diversity + Reliability")
    print("  1. Use DartboardRAG to get diverse candidates")
    print("  2. Use ReliableRAG to validate quality")
    print("  3. Generate from validated diverse set")
    
    print("\nStrategy 2: Augmentation + Adaptive")
    print("  1. Use DocumentAugmentation to enhance retrieval")
    print("  2. Use AdaptiveRAG to select best strategy")
    print("  3. Apply strategy-specific processing")
    
    print("\nStrategy 3: Full Pipeline")
    print("  1. DocumentAugmentation: Enhance documents with questions")
    print("  2. DartboardRAG: Retrieve diverse candidates")
    print("  3. ReliableRAG: Grade and filter for quality")
    print("  4. RerankingRAG: Final precision optimization")
    print("  5. Generate final answer")
    
    print("\n💡 Pseudo-code example:")
    print("""
    # Step 1: Augment and index
    aug_rag = DocumentAugmentation()
    aug_rag.index_documents_with_questions(documents)
    
    # Step 2: Retrieve with diversity
    dartboard = DartboardRAG(retriever=aug_rag.retriever)
    diverse_docs = dartboard.retrieve_with_diversity(query, k=10)
    
    # Step 3: Grade for reliability
    reliable = ReliableRAG()
    gradings = reliable.grade_documents(
        [d.page_content for d in diverse_docs['documents']], 
        query
    )
    relevant_docs = [d for d, g in zip(diverse_docs['documents'], gradings) 
                     if g['is_relevant']]
    
    # Step 4: Final generation
    context = "\\n\\n".join([d.page_content for d in relevant_docs])
    answer = llm.invoke(f"Context: {context}\\n\\nQuestion: {query}")
    """)
    
    print("\n✅ Key Benefit: Modular design allows flexible composition")


def example_use_case_recommendations():
    """
    Recommendations for when to use each technique
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Use Case Recommendations")
    print("=" * 80)
    
    print("\n📋 When to Use Each Technique:\n")
    
    print("1. ReliableRAG")
    print("   ✅ Use when:")
    print("      - Quality and accuracy are critical")
    print("      - You have noisy or heterogeneous documents")
    print("      - Need to explain why documents were selected/rejected")
    print("      - Want to avoid hallucination from irrelevant context")
    print("   ❌ Avoid when:")
    print("      - Speed is critical (grading adds latency)")
    print("      - All documents are high-quality and relevant")
    
    print("\n2. DartboardRAG")
    print("   ✅ Use when:")
    print("      - Documents contain overlapping information")
    print("      - Want comprehensive coverage of a topic")
    print("      - Large knowledge base with redundancy")
    print("      - Need to prevent echo chamber effect")
    print("   ❌ Avoid when:")
    print("      - All documents are unique and distinct")
    print("      - Simple, focused queries needing specific facts")
    
    print("\n3. DocumentAugmentation")
    print("   ✅ Use when:")
    print("      - Users ask questions in varied ways")
    print("      - Documents don't naturally contain question-like text")
    print("      - Want to improve recall for diverse queries")
    print("      - Building FAQ or QA systems")
    print("   ❌ Avoid when:")
    print("      - Documents already question-rich (forums, FAQs)")
    print("      - Computational budget is very limited")
    print("      - Index needs frequent updates (regeneration cost)")
    
    print("\n💡 Quick Selection Guide:")
    print("   - Need quality validation? → ReliableRAG")
    print("   - Have redundant docs? → DartboardRAG")
    print("   - Want better matching? → DocumentAugmentation")
    print("   - Want all benefits? → Combine them!")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("RAG TECHNIQUES - LATEST ADDITIONS SHOWCASE")
    print("=" * 80)
    print("\nDemonstrating 3 cutting-edge RAG techniques:")
    print("1. Reliable RAG - Document quality validation")
    print("2. Dartboard RAG - Relevance-diversity balance")
    print("3. Document Augmentation - Question generation")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("Set it in .env file or environment variables to run examples")
        print("\nShowing structure and concepts only...\n")
    
    try:
        # Run examples
        example_reliable_rag()
        example_dartboard_rag()
        example_document_augmentation()
        example_combining_techniques()
        example_use_case_recommendations()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review FINAL_COVERAGE_REPORT.md for complete technique list")
        print("2. Explore QUICK_REFERENCE.md for all technique examples")
        print("3. Try combining techniques for your specific use case")
        print("4. Check original notebooks for multi-modal examples")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have:")
        print("  1. Set OPENAI_API_KEY in .env file")
        print("  2. Installed all requirements: pip install -r requirements.txt")
        print("  3. Installed the package: pip install -e .")


if __name__ == "__main__":
    main()
