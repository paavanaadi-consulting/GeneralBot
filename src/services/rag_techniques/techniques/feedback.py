"""
Feedback Loop RAG Implementation

This module implements a RAG system with integrated user feedback loop
for continuous improvement of retrieval and response quality.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.base import BaseRAG
from ..core.config import RAGConfig
from ..utils.helpers import encode_from_string, read_pdf_to_string


class Response(BaseModel):
    """Response model for LLM structured output."""
    answer: str = Field(..., description="Yes or No answer")


class FeedbackRAG(BaseRAG):
    """
    RAG system with feedback loop for continuous improvement.
    
    This implementation:
    1. Collects user feedback on responses
    2. Adjusts document relevance scores based on feedback
    3. Fine-tunes the vector index periodically
    4. Learns from each interaction to improve future results
    
    Attributes:
        feedback_file: Path to store feedback data
        qa_chain: RetrievalQA chain for answering queries
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        content: Optional[str] = None,
        config: Optional[RAGConfig] = None,
        feedback_file: str = "feedback_data.json"
    ):
        """
        Initialize FeedbackRAG system.
        
        Args:
            pdf_path: Path to PDF document
            content: Text content (alternative to pdf_path)
            config: RAG configuration
            feedback_file: Path to store feedback data
        """
        super().__init__(pdf_path, content, config)
        self.feedback_file = feedback_file
        
        # Initialize QA chain
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever
        )
    
    def query(self, query_text: str, collect_feedback: bool = False) -> Dict[str, Any]:
        """
        Query the RAG system with optional feedback adjustment.
        
        Args:
            query_text: The question to answer
            collect_feedback: Whether to return feedback collection prompt
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Load previous feedback
        feedback_data = self.load_feedback()
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(query_text)
        
        # Adjust relevance scores based on feedback
        if feedback_data:
            docs = self._adjust_relevance_scores(query_text, docs, feedback_data)
        
        # Generate answer
        response = self.qa_chain(query_text)["result"]
        
        result = {
            "query": query_text,
            "answer": response,
            "context": [doc.page_content for doc in docs],
            "num_docs": len(docs)
        }
        
        if collect_feedback:
            result["feedback_prompt"] = (
                "Please rate this response:\n"
                "1. Relevance (1-5): How relevant was the answer?\n"
                "2. Quality (1-5): How good was the answer quality?\n"
                "3. Comments (optional): Any additional feedback"
            )
        
        return result
    
    def add_feedback(
        self,
        query: str,
        response: str,
        relevance: int,
        quality: int,
        comments: str = ""
    ) -> None:
        """
        Store user feedback for a query-response pair.
        
        Args:
            query: The original query
            response: The system's response
            relevance: Relevance score (1-5)
            quality: Quality score (1-5)
            comments: Optional text comments
        """
        feedback = {
            "query": query,
            "response": response,
            "relevance": int(relevance),
            "quality": int(quality),
            "comments": comments
        }
        
        # Append to feedback file
        with open(self.feedback_file, "a") as f:
            json.dump(feedback, f)
            f.write("\n")
    
    def load_feedback(self) -> List[Dict[str, Any]]:
        """
        Load all stored feedback data.
        
        Returns:
            List of feedback dictionaries
        """
        feedback_data = []
        if not os.path.exists(self.feedback_file):
            return feedback_data
        
        try:
            with open(self.feedback_file, "r") as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line.strip()))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading feedback: {e}")
        
        return feedback_data
    
    def _adjust_relevance_scores(
        self,
        query: str,
        docs: List[Any],
        feedback_data: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Adjust document relevance scores based on past feedback.
        
        Args:
            query: Current query
            docs: Retrieved documents
            feedback_data: Historical feedback
            
        Returns:
            Documents with adjusted relevance scores
        """
        # Create relevance checking prompt
        relevance_prompt = PromptTemplate(
            input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
            template="""
            Determine if the following feedback response is relevant to the current query and document content.
            
            Current query: {query}
            Feedback query: {feedback_query}
            Document content: {doc_content}
            Feedback response: {feedback_response}
            
            Is this feedback relevant? Respond with only 'Yes' or 'No'.
            """
        )
        
        relevance_chain = relevance_prompt | self.llm.with_structured_output(Response)
        
        # Process each document
        for doc in docs:
            # Initialize relevance score if not present
            if 'relevance_score' not in doc.metadata:
                doc.metadata['relevance_score'] = 1.0
            
            relevant_feedback = []
            
            # Check each feedback entry
            for feedback in feedback_data:
                input_data = {
                    "query": query,
                    "feedback_query": feedback['query'],
                    "doc_content": doc.page_content[:1000],
                    "feedback_response": feedback['response']
                }
                
                try:
                    result = relevance_chain.invoke(input_data).answer.lower()
                    if result == 'yes':
                        relevant_feedback.append(feedback)
                except Exception as e:
                    print(f"Error checking relevance: {e}")
                    continue
            
            # Adjust score based on relevant feedback
            if relevant_feedback:
                avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
                # Scale: 3 is neutral on a 1-5 scale
                doc.metadata['relevance_score'] *= (avg_relevance / 3.0)
        
        # Re-rank documents by adjusted scores
        return sorted(docs, key=lambda x: x.metadata.get('relevance_score', 1.0), reverse=True)
    
    def fine_tune_index(self, min_relevance: int = 4, min_quality: int = 4) -> None:
        """
        Fine-tune the vector index by incorporating high-quality feedback.
        
        This periodically updates the vector store with queries and responses
        that received good feedback ratings.
        
        Args:
            min_relevance: Minimum relevance score to include (1-5)
            min_quality: Minimum quality score to include (1-5)
        """
        feedback_data = self.load_feedback()
        
        # Filter for high-quality responses
        good_responses = [
            f for f in feedback_data
            if f['relevance'] >= min_relevance and f['quality'] >= min_quality
        ]
        
        if not good_responses:
            print("No high-quality feedback to incorporate.")
            return
        
        # Create additional documents from good Q&A pairs
        additional_texts = " ".join([
            f"{f['query']} {f['response']}"
            for f in good_responses
        ])
        
        # Combine with original content
        original_content = self.content if self.content else read_pdf_to_string(self.pdf_path)
        all_texts = f"{original_content} {additional_texts}"
        
        # Recreate vector store
        self.vectorstore = encode_from_string(
            all_texts,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Update retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.n_retrieved}
        )
        
        # Update QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever
        )
        
        print(f"Index fine-tuned with {len(good_responses)} high-quality feedback entries.")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        feedback_data = self.load_feedback()
        
        if not feedback_data:
            return {"total": 0}
        
        total = len(feedback_data)
        avg_relevance = sum(f['relevance'] for f in feedback_data) / total
        avg_quality = sum(f['quality'] for f in feedback_data) / total
        high_quality = sum(1 for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4)
        
        return {
            "total": total,
            "avg_relevance": round(avg_relevance, 2),
            "avg_quality": round(avg_quality, 2),
            "high_quality_count": high_quality,
            "high_quality_percentage": round(high_quality / total * 100, 1)
        }
