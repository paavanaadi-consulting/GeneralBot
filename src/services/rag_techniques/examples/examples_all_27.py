"""
Complete Usage Examples for All 27 RAG Techniques

This script demonstrates how to use each of the 27 implemented RAG techniques.
Run different examples by uncommenting the relevant sections.

Author: RAG Techniques Package
Date: December 2024
"""

import os
from typing import List, Dict

# Ensure API key is set
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Warning: OPENAI_API_KEY not set. Set it to run examples.")
    print("   export OPENAI_API_KEY='your-key-here'")


# =============================================================================
# CATEGORY 1: CORE RAG (1 technique)
# =============================================================================

def example_01_simple_rag():
    """Example 1: SimpleRAG - Basic retrieval-augmented generation."""
    print("\n" + "="*60)
    print("Example 1: SimpleRAG")
    print("="*60)
    
    from rag_techniques import SimpleRAG
    
    rag = SimpleRAG()
    
    # Add documents
    documents = [
        "Paris is the capital of France and known for the Eiffel Tower.",
        "London is the capital of the United Kingdom.",
        "Tokyo is the capital of Japan and the most populous city."
    ]
    rag.add_documents(documents)
    
    # Query
    result = rag.retrieve_and_generate("What is the capital of France?")
    print(f"\nQuery: What is the capital of France?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result.get('sources', []))} documents retrieved")


# =============================================================================
# CATEGORY 2: QUERY ENHANCEMENT (4 techniques)
# =============================================================================

def example_02_query_transform():
    """Example 2: QueryTransformRAG - Multiple query strategies."""
    print("\n" + "="*60)
    print("Example 2: QueryTransformRAG")
    print("="*60)
    
    from rag_techniques import QueryTransformRAG
    from rag_techniques.techniques.query_transform import QueryTransformConfig
    
    config = QueryTransformConfig(
        strategy="multi_query",  # or "decomposition", "step_back", "hype"
        num_queries=3
    )
    
    rag = QueryTransformRAG(config=config)
    rag.add_documents([
        "Climate change is causing global warming.",
        "Renewable energy reduces carbon emissions.",
        "Solar and wind are clean energy sources."
    ])
    
    result = rag.retrieve_and_generate("What are solutions to climate change?")
    print(f"\nOriginal Query: What are solutions to climate change?")
    print(f"Generated Queries: {result.get('transformed_queries', [])}")
    print(f"Answer: {result['answer']}")


def example_03_fusion():
    """Example 3: FusionRAG - Reciprocal Rank Fusion."""
    print("\n" + "="*60)
    print("Example 3: FusionRAG")
    print("="*60)
    
    from rag_techniques import FusionRAG
    from rag_techniques.techniques.fusion import FusionConfig
    
    config = FusionConfig(num_queries=3, rrf_k=60)
    rag = FusionRAG(config=config)
    
    rag.add_documents([
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "AI includes ML, DL, and NLP."
    ])
    
    result = rag.retrieve_and_generate("Explain AI and its subfields")
    print(f"\nQuery: Explain AI and its subfields")
    print(f"Fusion Score: {result.get('fusion_score', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_04_hyde():
    """Example 4: HyDERAG - Hypothetical Document Embedding."""
    print("\n" + "="*60)
    print("Example 4: HyDERAG")
    print("="*60)
    
    from rag_techniques import HyDERAG
    from rag_techniques.techniques.hyde import HyDEConfig
    
    config = HyDEConfig(num_hypothetical_docs=2)
    rag = HyDERAG(config=config)
    
    rag.add_documents([
        "The Great Wall of China is over 13,000 miles long.",
        "The Colosseum in Rome was built around 70-80 AD.",
        "Machu Picchu is an ancient Incan city in Peru."
    ])
    
    result = rag.retrieve_and_generate("Tell me about ancient monuments")
    print(f"\nQuery: Tell me about ancient monuments")
    print(f"Hypothetical Docs Generated: {result.get('num_hypothetical', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_05_feedback():
    """Example 5: FeedbackRAG - Iterative refinement."""
    print("\n" + "="*60)
    print("Example 5: FeedbackRAG")
    print("="*60)
    
    from rag_techniques import FeedbackRAG
    from rag_techniques.techniques.feedback import FeedbackConfig
    
    config = FeedbackConfig(max_iterations=3, quality_threshold=0.7)
    rag = FeedbackRAG(config=config)
    
    rag.add_documents([
        "Python is a high-level programming language.",
        "JavaScript is used for web development.",
        "Java is object-oriented and platform-independent."
    ])
    
    result = rag.retrieve_and_generate("Compare Python and JavaScript")
    print(f"\nQuery: Compare Python and JavaScript")
    print(f"Iterations: {result.get('iterations', 1)}")
    print(f"Final Quality: {result.get('quality_score', 'N/A')}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# CATEGORY 3: CONTEXT ENHANCEMENT (4 techniques)
# =============================================================================

def example_06_reranking():
    """Example 6: RerankingRAG - Cross-encoder reranking."""
    print("\n" + "="*60)
    print("Example 6: RerankingRAG")
    print("="*60)
    
    from rag_techniques import RerankingRAG
    from rag_techniques.techniques.reranking import RerankingConfig
    
    config = RerankingConfig(
        reranker="cross-encoder",
        top_n_after_rerank=3
    )
    rag = RerankingRAG(config=config)
    
    rag.add_documents([
        "Electric cars reduce emissions.",
        "Hybrid vehicles use both gas and electricity.",
        "Gasoline cars are traditional vehicles."
    ])
    
    result = rag.retrieve_and_generate("What are eco-friendly vehicles?")
    print(f"\nQuery: What are eco-friendly vehicles?")
    print(f"Reranked Results: {result.get('rerank_scores', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_07_compression():
    """Example 7: ContextualCompressionRAG - Relevant extraction."""
    print("\n" + "="*60)
    print("Example 7: ContextualCompressionRAG")
    print("="*60)
    
    from rag_techniques import ContextualCompressionRAG
    from rag_techniques.techniques.compression import CompressionConfig
    
    config = CompressionConfig(compression_method="llm")
    rag = ContextualCompressionRAG(config=config)
    
    rag.add_documents([
        "The human brain contains approximately 86 billion neurons. "
        "Each neuron can form thousands of connections called synapses.",
        "Memory formation involves strengthening synaptic connections "
        "through a process called long-term potentiation."
    ])
    
    result = rag.retrieve_and_generate("How many neurons in the brain?")
    print(f"\nQuery: How many neurons in the brain?")
    print(f"Compression Ratio: {result.get('compression_ratio', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_08_contextual_headers():
    """Example 8: ContextualChunkHeadersRAG - Hierarchical context."""
    print("\n" + "="*60)
    print("Example 8: ContextualChunkHeadersRAG")
    print("="*60)
    
    from rag_techniques import ContextualChunkHeadersRAG
    
    rag = ContextualChunkHeadersRAG()
    
    # Structured document with headers
    structured_doc = """
    # Company Overview
    Our company was founded in 2020.
    
    ## Product Line
    We offer three main products.
    
    ## Financial Performance
    Revenue grew 50% last year.
    """
    
    rag.add_documents([structured_doc])
    result = rag.retrieve_and_generate("What's the revenue growth?")
    print(f"\nQuery: What's the revenue growth?")
    print(f"Answer: {result['answer']}")


def example_09_document_augmentation():
    """Example 9: DocumentAugmentation - Metadata enrichment."""
    print("\n" + "="*60)
    print("Example 9: DocumentAugmentation")
    print("="*60)
    
    from rag_techniques import DocumentAugmentation
    from rag_techniques.techniques.document_augmentation import DocumentAugmentationConfig
    
    config = DocumentAugmentationConfig(
        generate_questions=True,
        generate_summaries=True
    )
    rag = DocumentAugmentation(config=config)
    
    rag.add_documents([
        "Quantum computing uses qubits instead of bits.",
        "Qubits can exist in superposition states."
    ])
    
    result = rag.retrieve_and_generate("What are qubits?")
    print(f"\nQuery: What are qubits?")
    print(f"Augmentations: {result.get('augmentations', 'N/A')}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# CATEGORY 4: CHUNKING STRATEGIES (2 techniques)
# =============================================================================

def example_10_semantic_chunking():
    """Example 10: SemanticChunkingRAG - Context-aware splitting."""
    print("\n" + "="*60)
    print("Example 10: SemanticChunkingRAG")
    print("="*60)
    
    from rag_techniques import SemanticChunkingRAG
    from rag_techniques.techniques.semantic_chunking import SemanticChunkingConfig
    
    config = SemanticChunkingConfig(
        similarity_threshold=0.7,
        min_chunk_size=100
    )
    rag = SemanticChunkingRAG(config=config)
    
    long_doc = """
    Machine learning is a field of AI. It enables computers to learn patterns.
    
    Deep learning is a subset of ML. It uses neural networks with many layers.
    
    Natural language processing helps computers understand human language.
    """
    
    rag.add_documents([long_doc])
    result = rag.retrieve_and_generate("Explain machine learning")
    print(f"\nQuery: Explain machine learning")
    print(f"Chunks Created: {result.get('num_chunks', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_11_proposition_chunking():
    """Example 11: PropositionChunkingRAG - Atomic propositions."""
    print("\n" + "="*60)
    print("Example 11: PropositionChunkingRAG")
    print("="*60)
    
    from rag_techniques import PropositionChunkingRAG
    
    rag = PropositionChunkingRAG()
    
    rag.add_documents([
        "Einstein developed relativity. He won the Nobel Prize in 1921. "
        "His work revolutionized physics."
    ])
    
    result = rag.retrieve_and_generate("When did Einstein win Nobel Prize?")
    print(f"\nQuery: When did Einstein win Nobel Prize?")
    print(f"Propositions: {result.get('num_propositions', 'N/A')}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# CATEGORY 5: HIERARCHICAL RETRIEVAL (2 techniques)
# =============================================================================

def example_12_hierarchical():
    """Example 12: HierarchicalIndicesRAG - Two-tier indexing."""
    print("\n" + "="*60)
    print("Example 12: HierarchicalIndicesRAG")
    print("="*60)
    
    from rag_techniques import HierarchicalIndicesRAG
    from rag_techniques.techniques.hierarchical import HierarchicalConfig
    
    config = HierarchicalConfig(
        use_summaries=True,
        summary_chunk_size=500
    )
    rag = HierarchicalIndicesRAG(config=config)
    
    rag.add_documents([
        "Chapter 1: Introduction to Biology. Biology is the study of life...",
        "Chapter 2: Cell Structure. Cells are the basic units of life..."
    ])
    
    result = rag.retrieve_and_generate("What is biology?")
    print(f"\nQuery: What is biology?")
    print(f"Hierarchy Levels: {result.get('levels_searched', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_13_raptor():
    """Example 13: RAPTORRAG - Recursive tree structure."""
    print("\n" + "="*60)
    print("Example 13: RAPTORRAG")
    print("="*60)
    
    from rag_techniques import RAPTORRAG
    from rag_techniques.techniques.raptor import RAPTORConfig
    
    config = RAPTORConfig(
        num_levels=3,
        cluster_size=3
    )
    rag = RAPTORRAG(config=config)
    
    rag.add_documents([
        "Topic 1: Climate patterns affect weather.",
        "Topic 2: Oceans regulate temperature.",
        "Topic 3: Forests absorb carbon dioxide."
    ])
    
    result = rag.retrieve_and_generate("How is climate regulated?")
    print(f"\nQuery: How is climate regulated?")
    print(f"Tree Levels: {result.get('tree_levels', 'N/A')}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# CATEGORY 6: ADAPTIVE SYSTEMS (3 techniques)
# =============================================================================

def example_14_adaptive():
    """Example 14: AdaptiveRAG - Dynamic strategy selection."""
    print("\n" + "="*60)
    print("Example 14: AdaptiveRAG")
    print("="*60)
    
    from rag_techniques import AdaptiveRAG
    from rag_techniques.techniques.adaptive import AdaptiveConfig
    
    config = AdaptiveConfig(
        enable_classification=True,
        complexity_threshold=0.5
    )
    rag = AdaptiveRAG(config=config)
    
    rag.add_documents([
        "Water boils at 100°C.",
        "The water cycle includes evaporation and condensation."
    ])
    
    result = rag.retrieve_and_generate("Explain the water cycle")
    print(f"\nQuery: Explain the water cycle")
    print(f"Selected Strategy: {result.get('strategy', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_15_self_rag():
    """Example 15: SelfRAG - Self-reflective evaluation."""
    print("\n" + "="*60)
    print("Example 15: SelfRAG")
    print("="*60)
    
    from rag_techniques import SelfRAG
    from rag_techniques.techniques.self_rag import SelfRAGConfig
    
    config = SelfRAGConfig(
        enable_reflection=True,
        quality_threshold=0.7
    )
    rag = SelfRAG(config=config)
    
    rag.add_documents([
        "The moon orbits Earth.",
        "Earth orbits the Sun."
    ])
    
    result = rag.retrieve_and_generate("What does Earth orbit?")
    print(f"\nQuery: What does Earth orbit?")
    print(f"Reflection Score: {result.get('reflection_score', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_16_crag():
    """Example 16: CorrectiveRAG - Dynamic correction."""
    print("\n" + "="*60)
    print("Example 16: CorrectiveRAG (CRAG)")
    print("="*60)
    
    from rag_techniques import CorrectiveRAG
    from rag_techniques.techniques.crag import CRAGConfig
    
    config = CRAGConfig(
        enable_web_search=False,  # Set to True for actual web search
        relevance_threshold=0.6
    )
    rag = CorrectiveRAG(config=config)
    
    rag.add_documents([
        "Some historical information here.",
        "More context about topics."
    ])
    
    result = rag.retrieve_and_generate("What happened in 2023?")
    print(f"\nQuery: What happened in 2023?")
    print(f"Correction Applied: {result.get('corrected', False)}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# CATEGORY 7: GRAPH-BASED RETRIEVAL (2 techniques)
# =============================================================================

def example_17_graph_rag():
    """Example 17: GraphRAG - Knowledge graph retrieval."""
    print("\n" + "="*60)
    print("Example 17: GraphRAG")
    print("="*60)
    
    from rag_techniques import GraphRAG
    from rag_techniques.techniques.graph_rag import GraphRAGConfig
    
    config = GraphRAGConfig(
        enable_entity_extraction=True,
        max_graph_depth=2
    )
    rag = GraphRAG(config=config)
    
    rag.add_documents([
        "Albert Einstein worked at Princeton University.",
        "Princeton University is in New Jersey.",
        "New Jersey is in the United States."
    ])
    
    result = rag.retrieve_and_generate("Where did Einstein work?")
    print(f"\nQuery: Where did Einstein work?")
    print(f"Graph Nodes: {result.get('num_nodes', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_18_reliable_rag():
    """Example 18: ReliableRAG - Citation and verification."""
    print("\n" + "="*60)
    print("Example 18: ReliableRAG")
    print("="*60)
    
    from rag_techniques import ReliableRAG
    from rag_techniques.techniques.reliable_rag import ReliableRAGConfig
    
    config = ReliableRAGConfig(
        enable_citations=True,
        confidence_threshold=0.7
    )
    rag = ReliableRAG(config=config)
    
    rag.add_documents([
        "Study from 2023 shows X.",
        "Research confirms Y."
    ])
    
    result = rag.retrieve_and_generate("What does research show?")
    print(f"\nQuery: What does research show?")
    print(f"Citations: {result.get('citations', [])}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# CATEGORY 8: SPECIALIZED TECHNIQUES (3 techniques)
# =============================================================================

def example_19_dartboard():
    """Example 19: DartboardRAG - Multi-granularity retrieval."""
    print("\n" + "="*60)
    print("Example 19: DartboardRAG")
    print("="*60)
    
    from rag_techniques import DartboardRAG
    from rag_techniques.techniques.dartboard import DartboardConfig
    
    config = DartboardConfig(
        granularity_levels=3,
        expansion_factor=2
    )
    rag = DartboardRAG(config=config)
    
    rag.add_documents([
        "Detail 1: Specific fact. Context: Broader information."
    ])
    
    result = rag.retrieve_and_generate("Tell me about this topic")
    print(f"\nQuery: Tell me about this topic")
    print(f"Granularity Levels: {result.get('levels_used', 'N/A')}")
    print(f"Answer: {result['answer']}")


def example_20_multimodal_captioning():
    """Example 20: MultiModalCaptioningRAG - Image captioning."""
    print("\n" + "="*60)
    print("Example 20: MultiModalCaptioningRAG")
    print("="*60)
    
    from rag_techniques import MultiModalCaptioningRAG
    from rag_techniques.techniques.multimodal_captioning import MultiModalCaptioningConfig
    
    config = MultiModalCaptioningConfig(
        extract_images=True,
        caption_model="gpt-4-vision-preview"
    )
    rag = MultiModalCaptioningRAG(config=config)
    
    print("\nNote: Requires PDF with images. Example:")
    print("  rag.process_pdf('document.pdf')")
    print("  result = rag.retrieve_and_generate('What does the diagram show?')")


def example_21_colpali():
    """Example 21: ColPaliRAG - Vision-language document understanding."""
    print("\n" + "="*60)
    print("Example 21: ColPaliRAG")
    print("="*60)
    
    from rag_techniques import ColPaliRAG
    from rag_techniques.techniques.multimodal_colpali import ColPaliConfig
    
    config = ColPaliConfig(
        model_name="vidore/colpali-v1.2",
        device="cpu"  # or "cuda"
    )
    
    print("\nNote: Requires ColPali dependencies. Example:")
    print("  rag = ColPaliRAG(config=config)")
    print("  rag.index_pdf('document.pdf', index_name='docs')")
    print("  result = rag.retrieve_and_generate('Find tables')")


# =============================================================================
# CATEGORY 9: AGENTIC SYSTEMS (1 technique)
# =============================================================================

def example_22_agentic():
    """Example 22: AgenticRAG - Agent-based reasoning."""
    print("\n" + "="*60)
    print("Example 22: AgenticRAG")
    print("="*60)
    
    from rag_techniques import AgenticRAG
    from rag_techniques.techniques.agentic_rag import AgenticRAGConfig
    
    config = AgenticRAGConfig(
        enable_query_analysis=True,
        enable_multi_turn=True,
        max_reasoning_steps=5
    )
    rag = AgenticRAG(config=config)
    
    rag.add_documents([
        "Product A costs $100 and has feature X.",
        "Product B costs $150 and has features X and Y."
    ])
    
    result = rag.retrieve_and_generate(
        "Compare products and recommend one",
        conversation_history=[]
    )
    print(f"\nQuery: Compare products and recommend one")
    print(f"Reasoning Steps: {result.get('reasoning_steps', 'N/A')}")
    print(f"Answer: {result['answer']}")


# =============================================================================
# MAIN MENU
# =============================================================================

def print_menu():
    """Print the example menu."""
    print("\n" + "="*60)
    print("RAG Techniques - Complete Examples (27 Techniques)")
    print("="*60)
    print("\nCATEGORY 1: CORE RAG")
    print("  1. SimpleRAG")
    print("\nCATEGORY 2: QUERY ENHANCEMENT")
    print("  2. QueryTransformRAG")
    print("  3. FusionRAG")
    print("  4. HyDERAG")
    print("  5. FeedbackRAG")
    print("\nCATEGORY 3: CONTEXT ENHANCEMENT")
    print("  6. RerankingRAG")
    print("  7. ContextualCompressionRAG")
    print("  8. ContextualChunkHeadersRAG")
    print("  9. DocumentAugmentation")
    print("\nCATEGORY 4: CHUNKING STRATEGIES")
    print(" 10. SemanticChunkingRAG")
    print(" 11. PropositionChunkingRAG")
    print("\nCATEGORY 5: HIERARCHICAL RETRIEVAL")
    print(" 12. HierarchicalIndicesRAG")
    print(" 13. RAPTORRAG")
    print("\nCATEGORY 6: ADAPTIVE SYSTEMS")
    print(" 14. AdaptiveRAG")
    print(" 15. SelfRAG")
    print(" 16. CorrectiveRAG (CRAG)")
    print("\nCATEGORY 7: GRAPH-BASED")
    print(" 17. GraphRAG")
    print(" 18. ReliableRAG")
    print("\nCATEGORY 8: SPECIALIZED")
    print(" 19. DartboardRAG")
    print(" 20. MultiModalCaptioningRAG")
    print(" 21. ColPaliRAG")
    print("\nCATEGORY 9: AGENTIC")
    print(" 22. AgenticRAG")
    print("\n  0. Run all examples")
    print("  q. Quit")
    print("="*60)


def main():
    """Main function to run examples."""
    examples = {
        1: example_01_simple_rag,
        2: example_02_query_transform,
        3: example_03_fusion,
        4: example_04_hyde,
        5: example_05_feedback,
        6: example_06_reranking,
        7: example_07_compression,
        8: example_08_contextual_headers,
        9: example_09_document_augmentation,
        10: example_10_semantic_chunking,
        11: example_11_proposition_chunking,
        12: example_12_hierarchical,
        13: example_13_raptor,
        14: example_14_adaptive,
        15: example_15_self_rag,
        16: example_16_crag,
        17: example_17_graph_rag,
        18: example_18_reliable_rag,
        19: example_19_dartboard,
        20: example_20_multimodal_captioning,
        21: example_21_colpali,
        22: example_22_agentic,
    }
    
    while True:
        print_menu()
        choice = input("\nSelect example number (or 0 for all, q to quit): ").strip()
        
        if choice.lower() == 'q':
            print("\nExiting. Thank you!")
            break
        
        try:
            num = int(choice)
            if num == 0:
                print("\nRunning all examples...")
                for i in sorted(examples.keys()):
                    try:
                        examples[i]()
                    except Exception as e:
                        print(f"\n❌ Error in example {i}: {str(e)}")
                print("\n✅ All examples completed!")
            elif num in examples:
                try:
                    examples[num]()
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
            else:
                print("\n❌ Invalid choice. Please try again.")
        except ValueError:
            print("\n❌ Invalid input. Please enter a number.")
        
        input("\nPress Enter to continue...")


if __name__ == '__main__':
    main()
