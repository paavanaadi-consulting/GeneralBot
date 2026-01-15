"""
Comprehensive Examples: Advanced RAG Techniques

This script demonstrates all available advanced RAG techniques in the package.
Run each example to see how different techniques work.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Import techniques
from rag_techniques.techniques import (
    SemanticChunkingRAG,
    HierarchicalIndicesRAG,
    HyDERAG,
    RAPTORRAG,
    SelfRAG,
    CorrectiveRAG,
    AdaptiveRAG,
    FeedbackRAG,
    RerankingRAG,
    QueryTransformRAG,
    FusionRAG,
    ContextualCompressionRAG
)

# Sample documents for testing
SAMPLE_DOCS = [
    Document(
        page_content="""
        Climate change is primarily caused by greenhouse gas emissions from human activities.
        The burning of fossil fuels releases carbon dioxide into the atmosphere, trapping heat.
        This leads to global warming, rising sea levels, and extreme weather events.
        """,
        metadata={"source": "climate_intro.txt"}
    ),
    Document(
        page_content="""
        The effects of climate change are already visible worldwide. Arctic ice is melting
        at unprecedented rates, coral reefs are bleaching due to warmer oceans, and extreme
        weather events like hurricanes and droughts are becoming more frequent and severe.
        """,
        metadata={"source": "climate_effects.txt"}
    ),
    Document(
        page_content="""
        Mitigation strategies for climate change include transitioning to renewable energy,
        improving energy efficiency, protecting forests, and developing carbon capture
        technologies. Individual actions like reducing consumption and choosing sustainable
        transportation also contribute to the solution.
        """,
        metadata={"source": "climate_solutions.txt"}
    ),
]

SAMPLE_QUERY = "What are the main causes and effects of climate change?"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_semantic_chunking():
    """Example: Semantic Chunking RAG"""
    print_section("Semantic Chunking RAG")
    
    print("Creating semantic chunks that preserve context...")
    rag = SemanticChunkingRAG(
        breakpoint_type="percentile",
        breakpoint_threshold=90,
        top_k=2
    )
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Get chunk statistics
    stats = rag.get_chunk_stats()
    print(f"Chunk Statistics: {stats}")
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_context=True)
    print(f"\nAnswer: {result['answer']}")
    print(f"Number of chunks used: {result['num_chunks']}")


def example_hierarchical_indices():
    """Example: Hierarchical Indices RAG"""
    print_section("Hierarchical Indices RAG")
    
    print("Building hierarchical index with summaries and chunks...")
    rag = HierarchicalIndicesRAG(
        num_summaries=2,
        num_chunks_per_summary=2,
        chunk_size=500
    )
    
    # Create hierarchical vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Get hierarchy statistics
    stats = rag.get_hierarchy_stats()
    print(f"Hierarchy Statistics: {stats}")
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_context=False)
    print(f"\nAnswer: {result['answer']}")


def example_hyde():
    """Example: Hypothetical Document Embedding (HyDE)"""
    print_section("Hypothetical Document Embedding (HyDE)")
    
    print("Generating hypothetical document for improved retrieval...")
    rag = HyDERAG(top_k=2)
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query with HyDE
    result = rag.query(
        SAMPLE_QUERY,
        return_hypothetical=True,
        return_context=False
    )
    
    print(f"Hypothetical Document (first 200 chars):")
    print(f"{result['hypothetical_document'][:200]}...")
    print(f"\nAnswer: {result['answer']}")


def example_raptor():
    """Example: RAPTOR (Recursive Abstractive Processing)"""
    print_section("RAPTOR - Recursive Abstractive Processing")
    
    print("Building hierarchical tree structure...")
    rag = RAPTORRAG(
        max_levels=2,
        n_clusters=2,
        top_k=3
    )
    
    # Build tree and create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Get tree statistics
    stats = rag.get_tree_stats()
    print(f"Tree Statistics: {stats}")
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_context=False)
    print(f"\nAnswer: {result['answer']}")
    print(f"Levels used: {result['levels_used']}")


def example_self_rag():
    """Example: Self-RAG (Self-Reflective RAG)"""
    print_section("Self-RAG - Self-Reflective Retrieval")
    
    print("Implementing self-reflective retrieval with quality checks...")
    rag = SelfRAG(
        top_k=2,
        relevance_threshold=0.7
    )
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query with full reflections
    result = rag.query(SAMPLE_QUERY, return_reflections=True)
    
    print(f"Retrieval Decision: {result['reflections']['retrieval_decision']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Quality Score: {result['reflections']['utility']}")


def example_crag():
    """Example: Corrective RAG (CRAG)"""
    print_section("Corrective RAG (CRAG)")
    
    print("Implementing corrective retrieval with relevance evaluation...")
    rag = CorrectiveRAG(
        top_k=2,
        high_relevance_threshold=0.7,
        low_relevance_threshold=0.3,
        enable_web_search=False  # Disable for demo
    )
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_metadata=True)
    
    print(f"Strategy Used: {result['strategy']}")
    print(f"Max Relevance Score: {result['max_relevance_score']:.2f}")
    print(f"Correction Applied: {result['metadata']['correction_applied']}")
    print(f"\nAnswer: {result['answer']}")


def example_adaptive():
    """Example: Adaptive RAG"""
    print_section("Adaptive RAG")
    
    print("Classifying query and selecting optimal strategy...")
    rag = AdaptiveRAG()
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_classification=True)
    
    print(f"Query Type: {result['query_classification']['query_type']}")
    print(f"Strategy Selected: {result['query_classification']['strategy']}")
    print(f"Confidence: {result['query_classification']['confidence']}")
    print(f"\nAnswer: {result['answer']}")


def example_feedback():
    """Example: Feedback Loop RAG"""
    print_section("Feedback Loop RAG")
    
    print("Implementing iterative refinement with feedback...")
    rag = FeedbackRAG(max_iterations=2)
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query with feedback
    result = rag.query(SAMPLE_QUERY, return_history=True)
    
    print(f"Iterations: {result['iterations']}")
    print(f"Final Quality: {result['final_quality']}")
    print(f"\nFinal Answer: {result['answer']}")


def example_reranking():
    """Example: Reranking RAG"""
    print_section("Reranking RAG")
    
    print("Retrieving and reranking documents...")
    rag = RerankingRAG(initial_k=5, final_k=2)
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_scores=True)
    
    print(f"Initial retrieval: {result['initial_docs']} docs")
    print(f"After reranking: {result['final_docs']} docs")
    print(f"\nAnswer: {result['answer']}")


def example_query_transform():
    """Example: Query Transformation RAG"""
    print_section("Query Transformation RAG")
    
    print("Transforming query for better retrieval...")
    rag = QueryTransformRAG(
        transformation_type="multi_query",
        num_variants=2
    )
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_transformations=True)
    
    print("Query Variants:")
    for i, variant in enumerate(result['query_variants'], 1):
        print(f"  {i}. {variant}")
    print(f"\nAnswer: {result['answer']}")


def example_fusion():
    """Example: Fusion RAG"""
    print_section("Fusion RAG")
    
    print("Fusing results from multiple query variants...")
    rag = FusionRAG(num_queries=2, top_k=3)
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_fusion_details=True)
    
    print("Generated Queries:")
    for i, q in enumerate(result['generated_queries'], 1):
        print(f"  {i}. {q}")
    print(f"\nFused Documents: {result['num_fused_docs']}")
    print(f"Answer: {result['answer']}")


def example_compression():
    """Example: Contextual Compression RAG"""
    print_section("Contextual Compression RAG")
    
    print("Compressing context to relevant information...")
    rag = ContextualCompressionRAG(initial_k=5, compressed_k=2)
    
    # Create vectorstore
    rag.create_vectorstore(SAMPLE_DOCS)
    
    # Query
    result = rag.query(SAMPLE_QUERY, return_compression_info=True)
    
    print(f"Initial documents: {result['initial_docs']}")
    print(f"After compression: {result['compressed_docs']}")
    print(f"Compression ratio: {result['compression_ratio']:.2%}")
    print(f"\nAnswer: {result['answer']}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "🚀" * 40)
    print(" " * 15 + "ADVANCED RAG TECHNIQUES SHOWCASE")
    print("🚀" * 40)
    
    examples = [
        ("Semantic Chunking", example_semantic_chunking),
        ("Hierarchical Indices", example_hierarchical_indices),
        ("HyDE", example_hyde),
        ("RAPTOR", example_raptor),
        ("Self-RAG", example_self_rag),
        ("Corrective RAG", example_crag),
        ("Adaptive RAG", example_adaptive),
        ("Feedback Loop", example_feedback),
        ("Reranking", example_reranking),
        ("Query Transformation", example_query_transform),
        ("Fusion", example_fusion),
        ("Contextual Compression", example_compression),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "✅" * 40)
    print(" " * 20 + "ALL EXAMPLES COMPLETED")
    print("✅" * 40 + "\n")


def interactive_menu():
    """Interactive menu for running specific examples."""
    examples = {
        '1': ("Semantic Chunking", example_semantic_chunking),
        '2': ("Hierarchical Indices", example_hierarchical_indices),
        '3': ("HyDE", example_hyde),
        '4': ("RAPTOR", example_raptor),
        '5': ("Self-RAG", example_self_rag),
        '6': ("Corrective RAG", example_crag),
        '7': ("Adaptive RAG", example_adaptive),
        '8': ("Feedback Loop", example_feedback),
        '9': ("Reranking", example_reranking),
        '10': ("Query Transformation", example_query_transform),
        '11': ("Fusion", example_fusion),
        '12': ("Contextual Compression", example_compression),
        'all': ("All Examples", run_all_examples),
    }
    
    print("\n" + "📚" * 40)
    print(" " * 15 + "RAG TECHNIQUES - INTERACTIVE MENU")
    print("📚" * 40 + "\n")
    
    print("Select an example to run:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  q. Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == 'q':
        print("Goodbye!")
        return False
    
    if choice in examples:
        name, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice. Please try again.")
    
    return True


if __name__ == "__main__":
    import sys
    
    # Check if running in non-interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        run_all_examples()
    else:
        # Interactive mode
        while interactive_menu():
            print("\n" + "-" * 80)
            input("Press Enter to continue...")
