"""
RAG Evaluation Metrics

This module provides metrics for evaluating RAG system performance.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


class RelevanceScore(BaseModel):
    """Model for relevance evaluation."""
    score: float = Field(description="Relevance score 0-1")
    reasoning: str = Field(description="Explanation")


class FaithfulnessScore(BaseModel):
    """Model for faithfulness evaluation."""
    score: float = Field(description="Faithfulness score 0-1")
    reasoning: str = Field(description="Explanation")


class AnswerQualityScore(BaseModel):
    """Model for answer quality evaluation."""
    score: float = Field(description="Quality score 0-1")
    reasoning: str = Field(description="Explanation")


class RAGEvaluator:
    """
    Evaluator for RAG system performance.
    
    Provides metrics for:
    1. Retrieval quality (precision, recall, relevance)
    2. Generation quality (faithfulness, answer relevance)
    3. End-to-end performance
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize evaluator.
        
        Args:
            model_name: LLM model for evaluation
        """
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.embeddings = OpenAIEmbeddings()
    
    def evaluate_retrieval_relevance(
        self,
        query: str,
        retrieved_docs: List[str],
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate relevance of retrieved documents.
        
        Args:
            query: The search query
            retrieved_docs: List of retrieved document texts
            use_llm: Whether to use LLM for evaluation
            
        Returns:
            Dictionary with relevance metrics
        """
        if use_llm:
            return self._llm_evaluate_relevance(query, retrieved_docs)
        else:
            return self._embedding_evaluate_relevance(query, retrieved_docs)
    
    def _llm_evaluate_relevance(
        self,
        query: str,
        retrieved_docs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate relevance using LLM."""
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            Rate how relevant this document is to the query on a scale of 0 to 1.
            
            Query: {query}
            
            Document: {document}
            
            Provide a score (0.0 = not relevant, 1.0 = highly relevant) and reasoning.
            """
        )
        
        chain = prompt | self.llm.with_structured_output(RelevanceScore)
        
        scores = []
        for doc in retrieved_docs:
            try:
                result = chain.invoke({"query": query, "document": doc[:1000]})
                scores.append(result.score)
            except Exception as e:
                print(f"Error evaluating relevance: {e}")
                scores.append(0.0)
        
        return {
            "individual_scores": scores,
            "mean_relevance": np.mean(scores),
            "min_relevance": np.min(scores),
            "max_relevance": np.max(scores),
            "num_relevant": sum(1 for s in scores if s >= 0.7),
            "num_docs": len(retrieved_docs)
        }
    
    def _embedding_evaluate_relevance(
        self,
        query: str,
        retrieved_docs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate relevance using embeddings."""
        query_emb = np.array(self.embeddings.embed_query(query))
        doc_embs = np.array(self.embeddings.embed_documents(retrieved_docs))
        
        # Calculate cosine similarities
        similarities = []
        for doc_emb in doc_embs:
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            similarities.append(float(sim))
        
        return {
            "individual_scores": similarities,
            "mean_relevance": np.mean(similarities),
            "min_relevance": np.min(similarities),
            "max_relevance": np.max(similarities),
            "num_relevant": sum(1 for s in similarities if s >= 0.7),
            "num_docs": len(retrieved_docs)
        }
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if answer is faithful to context.
        
        Args:
            answer: Generated answer
            context: Source context documents
            
        Returns:
            Dictionary with faithfulness metrics
        """
        prompt = PromptTemplate(
            input_variables=["answer", "context"],
            template="""
            Evaluate if the answer is faithful to (supported by) the provided context.
            
            Context:
            {context}
            
            Answer:
            {answer}
            
            Rate faithfulness from 0 to 1:
            - 1.0 = Answer is fully supported by context
            - 0.5 = Answer is partially supported
            - 0.0 = Answer contradicts or is not supported by context
            
            Provide score and reasoning.
            """
        )
        
        chain = prompt | self.llm.with_structured_output(FaithfulnessScore)
        
        try:
            result = chain.invoke({
                "answer": answer,
                "context": "\n\n".join(context[:3])  # Limit context length
            })
            
            return {
                "faithfulness_score": result.score,
                "reasoning": result.reasoning,
                "is_faithful": result.score >= 0.7
            }
        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")
            return {
                "faithfulness_score": 0.0,
                "reasoning": f"Error: {e}",
                "is_faithful": False
            }
    
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if answer is relevant to query.
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            Dictionary with answer relevance metrics
        """
        prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template="""
            Evaluate if the answer properly addresses the query.
            
            Query: {query}
            
            Answer: {answer}
            
            Rate answer relevance from 0 to 1:
            - 1.0 = Answer directly and completely addresses the query
            - 0.5 = Answer partially addresses the query
            - 0.0 = Answer does not address the query
            
            Provide score and reasoning.
            """
        )
        
        chain = prompt | self.llm.with_structured_output(AnswerQualityScore)
        
        try:
            result = chain.invoke({"query": query, "answer": answer})
            
            return {
                "relevance_score": result.score,
                "reasoning": result.reasoning,
                "is_relevant": result.score >= 0.7
            }
        except Exception as e:
            print(f"Error evaluating answer relevance: {e}")
            return {
                "relevance_score": 0.0,
                "reasoning": f"Error: {e}",
                "is_relevant": False
            }
    
    def evaluate_end_to_end(
        self,
        query: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive end-to-end evaluation.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Retrieval evaluation
        retrieval_metrics = self.evaluate_retrieval_relevance(query, context)
        
        # Generation evaluation
        faithfulness_metrics = self.evaluate_faithfulness(answer, context)
        answer_relevance_metrics = self.evaluate_answer_relevance(query, answer)
        
        results = {
            "retrieval": retrieval_metrics,
            "faithfulness": faithfulness_metrics,
            "answer_relevance": answer_relevance_metrics
        }
        
        # Ground truth comparison if provided
        if ground_truth:
            # Calculate semantic similarity between answer and ground truth
            answer_emb = np.array(self.embeddings.embed_query(answer))
            truth_emb = np.array(self.embeddings.embed_query(ground_truth))
            
            similarity = np.dot(answer_emb, truth_emb) / (
                np.linalg.norm(answer_emb) * np.linalg.norm(truth_emb)
            )
            
            results["ground_truth_similarity"] = float(similarity)
        
        # Overall score (weighted average)
        overall_score = (
            retrieval_metrics["mean_relevance"] * 0.3 +
            faithfulness_metrics["faithfulness_score"] * 0.4 +
            answer_relevance_metrics["relevance_score"] * 0.3
        )
        
        results["overall_score"] = round(overall_score, 3)
        
        return results
    
    def batch_evaluate(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple query-answer pairs.
        
        Args:
            queries: List of queries
            answers: List of answers
            contexts: List of context lists
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not (len(queries) == len(answers) == len(contexts)):
            raise ValueError("Queries, answers, and contexts must have same length")
        
        all_results = []
        for i, (query, answer, context) in enumerate(zip(queries, answers, contexts)):
            gt = ground_truths[i] if ground_truths else None
            result = self.evaluate_end_to_end(query, answer, context, gt)
            all_results.append(result)
        
        # Aggregate metrics
        aggregate = {
            "num_evaluations": len(all_results),
            "mean_retrieval_relevance": np.mean([r["retrieval"]["mean_relevance"] for r in all_results]),
            "mean_faithfulness": np.mean([r["faithfulness"]["faithfulness_score"] for r in all_results]),
            "mean_answer_relevance": np.mean([r["answer_relevance"]["relevance_score"] for r in all_results]),
            "mean_overall_score": np.mean([r["overall_score"] for r in all_results]),
            "individual_results": all_results
        }
        
        return aggregate
