"""
Comprehensive Example Usage of RAG Techniques Package

This script demonstrates all the major features and techniques available
in the rag_techniques package.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Import the package
from rag_techniques import (
    # Core
    SimpleRAG,
    RAGConfig,
    # Techniques
    FeedbackRAG,
    QueryTransformRAG,
    RerankingRAG,
    ContextualCompressionRAG,
    FusionRAG,
    # Evaluation
    RAGEvaluator,
)


def example_1_simple_rag():
    """Example 1: Basic RAG implementation"""
    print("\n" + "="*60)
    print("Example 1: Simple RAG")
    print("="*60)
    
    # Initialize with default configuration
    rag = SimpleRAG(pdf_path="../data/Understanding_Climate_Change.pdf")
    
    # Query the system
    result = rag.query("What is the greenhouse effect?")
    
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Number of context docs: {result['num_docs']}")


def example_2_custom_config():
    """Example 2: RAG with custom configuration"""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    config = RAGConfig(
        chunk_size=500,
        chunk_overlap=100,
        n_retrieved=3,
        model_name="gpt-4o",
        temperature=0.1
    )
    
    # Initialize RAG with custom config
    rag = SimpleRAG(
        pdf_path="../data/Understanding_Climate_Change.pdf",
        config=config
    )
    
    result = rag.query("What causes climate change?")
    print(f"Answer: {result['answer'][:200]}...")


def example_3_feedback_rag():
    """Example 3: RAG with Feedback Loop"""
    print("\n" + "="*60)
    print("Example 3: Feedback RAG")
    print("="*60)
    
    # Initialize Feedback RAG
    rag = FeedbackRAG(
        pdf_path="../data/Understanding_Climate_Change.pdf",
        feedback_file="my_feedback.json"
    )
    
    # Query
    result = rag.query("What is global warming?", collect_feedback=True)
    print(f"Answer: {result['answer'][:200]}...")
    
    # Simulate user feedback
    rag.add_feedback(
        query=result['query'],
        response=result['answer'],
        relevance=5,
        quality=4,
        comments="Good answer, very informative"
    )
    
    # Get feedback statistics
    stats = rag.get_feedback_stats()
    print(f"\nFeedback Stats: {stats}")


def example_4_query_transformation():
    """Example 4: Query Transformation Techniques"""
    print("\n" + "="*60)
    print("Example 4: Query Transformation")
    print("="*60)
    
    rag = QueryTransformRAG(
        pdf_path="../data/Understanding_Climate_Change.pdf"
    )
    
    query = "How does CO2 affect temperature?"
    
    # Method 1: Query Rewriting
    result1 = rag.query(query, method="rewrite")
    print(f"Method: Query Rewrite")
    print(f"Transformed: {result1['transformed_queries']}")
    print(f"Answer: {result1['answer'][:150]}...\n")


def example_5_reranking():
    """Example 5: Document Reranking"""
    print("\n" + "="*60)
    print("Example 5: Reranking")
    print("="*60)
    
    rag = RerankingRAG(
        pdf_path="../data/Understanding_Climate_Change.pdf",
        rerank_top_k=10
    )
    
    query = "What are the effects of climate change?"
    
    # LLM-based reranking
    result = rag.query(query, rerank_method="llm")
    print(f"Method: LLM Reranking")
    print(f"Relevance scores: {result['relevance_scores']}")
    print(f"Answer: {result['answer'][:150]}...")


def example_6_contextual_compression():
    """Example 6: Contextual Compression"""
    print("\n" + "="*60)
    print("Example 6: Contextual Compression")
    print("="*60)
    
    rag = ContextualCompressionRAG(
        pdf_path="../data/Understanding_Climate_Change.pdf",
        compression_ratio=0.3
    )
    
    query = "What is the Paris Agreement?"
    
    # LLM-based compression
    result = rag.query(query, compression_method="llm")
    print(f"Compression Stats:")
    print(f"  Original length: {result['total_original_length']}")
    print(f"  Compressed length: {result['total_compressed_length']}")
    print(f"  Compression ratio: {result['avg_compression_ratio']}")
    print(f"  Token savings: {result['token_savings']}%")


def example_7_fusion_retrieval():
    """Example 7: Fusion Retrieval"""
    print("\n" + "="*60)
    print("Example 7: Fusion Retrieval")
    print("="*60)
    
    rag = FusionRAG(
        pdf_path="../data/Understanding_Climate_Change.pdf"
    )
    
    query = "What causes rising sea levels?"
    
    # Reciprocal Rank Fusion
    result = rag.query(query, method="rrf")
    print(f"Method: Reciprocal Rank Fusion")
    print(f"Retrieval info: {result['retrieval_info']}")
    print(f"Answer: {result['answer'][:150]}...")


def example_8_evaluation():
    """Example 8: RAG Evaluation"""
    print("\n" + "="*60)
    print("Example 8: RAG Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(model_name="gpt-4o")
    
    # Run a query
    rag = SimpleRAG(pdf_path="../data/Understanding_Climate_Change.pdf")
    result = rag.query("What is climate change?")
    
    # Evaluate the result
    evaluation = evaluator.evaluate_end_to_end(
        query=result['query'],
        answer=result['answer'],
        context=result['context']
    )
    
    print(f"Evaluation Results:")
    print(f"  Overall Score: {evaluation['overall_score']}")
    print(f"  Retrieval Relevance: {evaluation['retrieval']['mean_relevance']:.3f}")
    print(f"  Faithfulness: {evaluation['faithfulness']['faithfulness_score']:.3f}")
    print(f"  Answer Relevance: {evaluation['answer_relevance']['relevance_score']:.3f}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("RAG TECHNIQUES PACKAGE - COMPREHENSIVE EXAMPLES")
    print("="*60)
    
    # Run examples (comment out any you don't want to run)
    example_1_simple_rag()
    example_2_custom_config()
    example_3_feedback_rag()
    example_4_query_transformation()
    example_5_reranking()
    example_6_contextual_compression()
    example_7_fusion_retrieval()
    example_8_evaluation()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY in your .env file")
    else:
        main()
