"""
Query Transformation Techniques

This module implements various query transformation methods to improve
retrieval quality by reformulating user queries.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.base import BaseRAG
from ..core.config import RAGConfig


class QueryRewrite(BaseModel):
    """Model for query rewrite output."""
    rewritten_query: str = Field(description="The rewritten query")


class MultiQuery(BaseModel):
    """Model for multiple query generation."""
    queries: List[str] = Field(description="List of generated queries")


class StepBackQuery(BaseModel):
    """Model for step-back prompting."""
    step_back_query: str = Field(description="Higher-level abstraction query")


class QueryTransformRAG(BaseRAG):
    """
    RAG with query transformation techniques.
    
    Implements multiple strategies:
    1. Query Rewriting - Reformulate the query for better retrieval
    2. Multi-Query - Generate multiple perspectives of the same question
    3. Step-Back Prompting - Create higher-level abstraction
    4. HyDE (Hypothetical Document Embeddings) - Generate hypothetical answers
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        content: Optional[str] = None,
        config: Optional[RAGConfig] = None
    ):
        super().__init__(pdf_path, content, config)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens
        )
    
    def query_rewrite(self, query: str) -> str:
        """
        Rewrite the query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert at reformulating user queries to improve information retrieval.
            
            Original query: {query}
            
            Rewrite this query to be more specific and search-friendly while preserving the original intent.
            Consider:
            - Adding relevant context
            - Using more precise terminology
            - Breaking down complex questions
            
            Rewritten query:
            """
        )
        
        chain = prompt | self.llm.with_structured_output(QueryRewrite)
        result = chain.invoke({"query": query})
        return result.rewritten_query
    
    def multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query perspectives.
        
        Args:
            query: Original query
            num_queries: Number of queries to generate
            
        Returns:
            List of generated queries
        """
        prompt = PromptTemplate(
            input_variables=["query", "num_queries"],
            template="""
            You are an AI assistant helping to generate multiple perspectives of a question.
            
            Original question: {query}
            
            Generate {num_queries} different versions of this question that:
            - Approach the topic from different angles
            - Use different wordings and phrasings
            - Maintain the same core information need
            
            Return them as a list.
            """
        )
        
        chain = prompt | self.llm.with_structured_output(MultiQuery)
        result = chain.invoke({"query": query, "num_queries": num_queries})
        return result.queries
    
    def step_back_query(self, query: str) -> str:
        """
        Generate a step-back (higher-level) query.
        
        Args:
            query: Original specific query
            
        Returns:
            Higher-level abstraction query
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert at abstraction and reasoning.
            
            Given this specific question: {query}
            
            Generate a higher-level, more general question that would help establish
            fundamental concepts needed to answer the original question.
            
            For example:
            - Specific: "What is the boiling point of water at 2000m altitude?"
            - Step-back: "How does altitude affect the physical properties of water?"
            
            Step-back question:
            """
        )
        
        chain = prompt | self.llm.with_structured_output(StepBackQuery)
        result = chain.invoke({"query": query})
        return result.step_back_query
    
    def hyde_query(self, query: str) -> str:
        """
        Generate hypothetical document (HyDE technique).
        
        Creates a hypothetical answer that can be used for retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Hypothetical document/answer
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert assistant. Generate a detailed, hypothetical answer to the following question.
            This answer will be used for semantic search, so make it comprehensive and well-structured.
            
            Question: {query}
            
            Hypothetical Answer:
            """
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        return result.content
    
    def query(
        self,
        query_text: str,
        method: str = "rewrite",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with transformation method.
        
        Args:
            query_text: Original query
            method: Transformation method - 'rewrite', 'multi', 'step_back', 'hyde'
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary with answer and metadata
        """
        transformed_queries = []
        
        if method == "rewrite":
            transformed = self.query_rewrite(query_text)
            transformed_queries = [transformed]
            
        elif method == "multi":
            num_queries = kwargs.get("num_queries", 3)
            transformed_queries = self.multi_query(query_text, num_queries)
            
        elif method == "step_back":
            step_back = self.step_back_query(query_text)
            # Use both original and step-back query
            transformed_queries = [query_text, step_back]
            
        elif method == "hyde":
            hypothetical_doc = self.hyde_query(query_text)
            # Search using hypothetical document
            docs = self.vectorstore.similarity_search(hypothetical_doc, k=self.config.n_retrieved)
            context = [doc.page_content for doc in docs]
            
            # Generate final answer
            answer_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                Answer the following question based on the provided context.
                
                Question: {query}
                
                Context:
                {context}
                
                Answer:
                """
            )
            chain = answer_prompt | self.llm
            result = chain.invoke({
                "query": query_text,
                "context": "\n\n".join(context)
            })
            
            return {
                "query": query_text,
                "method": method,
                "hypothetical_document": hypothetical_doc,
                "answer": result.content,
                "context": context,
                "num_docs": len(docs)
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # For non-HyDE methods, retrieve documents for all transformed queries
        all_docs = []
        for tq in transformed_queries:
            docs = self.retriever.get_relevant_documents(tq)
            all_docs.extend(docs)
        
        # Remove duplicates based on content
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        # Get top-k documents
        unique_docs = unique_docs[:self.config.n_retrieved]
        context = [doc.page_content for doc in unique_docs]
        
        # Generate answer
        answer_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Answer the following question based on the provided context.
            
            Question: {query}
            
            Context:
            {context}
            
            Answer:
            """
        )
        chain = answer_prompt | self.llm
        result = chain.invoke({
            "query": query_text,
            "context": "\n\n".join(context)
        })
        
        return {
            "query": query_text,
            "method": method,
            "transformed_queries": transformed_queries,
            "answer": result.content,
            "context": context,
            "num_docs": len(unique_docs)
        }
