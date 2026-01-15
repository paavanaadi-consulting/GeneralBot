"""
Proposition Chunking

This module implements proposition-based chunking where documents are broken down
into atomic, factual, self-contained propositions for more granular retrieval.
Based on research by Chen et al. (2023).
"""

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class PropositionQuality(BaseModel):
    """Quality assessment for a proposition."""
    accuracy: float = Field(description="Accuracy score (0-1)")
    clarity: float = Field(description="Clarity score (0-1)")
    completeness: float = Field(description="Completeness score (0-1)")
    conciseness: float = Field(description="Conciseness score (0-1)")
    overall: float = Field(description="Overall quality score (0-1)")


class PropositionChunkingRAG(BaseRAGTechnique):
    """
    Proposition-based chunking for granular retrieval.
    
    Breaks down documents into atomic, factual, self-contained propositions.
    Each proposition is:
    - Atomic: Contains one piece of information
    - Factual: Based on facts, not opinions
    - Self-contained: Can be understood without additional context
    - Concise: Brief and to the point
    
    Attributes:
        embeddings: OpenAI embeddings model
        vectorstore: FAISS vector store for propositions
        propositions: List of generated propositions
        quality_threshold: Minimum quality score for propositions
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        quality_threshold: float = 0.7,
        top_k: int = 5,
        initial_chunk_size: int = 1000
    ):
        """
        Initialize Proposition Chunking RAG.
        
        Args:
            config: Configuration manager instance
            quality_threshold: Minimum quality score (0-1)
            top_k: Number of propositions to retrieve
            initial_chunk_size: Size for initial document chunking
        """
        super().__init__(config)
        self.quality_threshold = quality_threshold
        self.top_k = top_k
        self.initial_chunk_size = initial_chunk_size
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        self.vectorstore: Optional[FAISS] = None
        self.propositions: List[Document] = []
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.proposition_prompt = PromptTemplate(
            input_variables=["chunk"],
            template="""Break down the following text into atomic, factual, self-contained propositions.

Each proposition should:
1. Contain one piece of information
2. Be factual and objective
3. Be self-contained (understandable without context)
4. Be concise

Text: {chunk}

Generate propositions as a numbered list:"""
        )
        
        self.quality_check_prompt = PromptTemplate(
            input_variables=["proposition"],
            template="""Evaluate the following proposition on these criteria (score 0-1 for each):

Proposition: {proposition}

1. Accuracy: Is it factually correct?
2. Clarity: Is it clear and unambiguous?
3. Completeness: Is it self-contained?
4. Conciseness: Is it brief and to the point?

Provide scores and overall quality."""
        )
    
    def generate_propositions(
        self,
        chunk: str
    ) -> List[str]:
        """
        Generate propositions from a text chunk.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            List of proposition strings
        """
        prompt = self.proposition_prompt.format(chunk=chunk)
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse numbered list
        propositions = []
        for line in response_text.split('\n'):
            line = line.strip()
            # Remove numbering (e.g., "1.", "2)", etc.)
            import re
            cleaned = re.sub(r'^\d+[\.)]\s*', '', line)
            if cleaned and len(cleaned) > 10:  # Filter very short lines
                propositions.append(cleaned)
        
        self.logger.info(f"Generated {len(propositions)} propositions from chunk")
        return propositions
    
    def check_quality(
        self,
        proposition: str
    ) -> PropositionQuality:
        """
        Check quality of a proposition.
        
        Args:
            proposition: Proposition to evaluate
            
        Returns:
            PropositionQuality object with scores
        """
        prompt = self.quality_check_prompt.format(proposition=proposition)
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse scores (simplified - in production, use structured output)
        import re
        
        def extract_score(text: str, keyword: str) -> float:
            """Extract score for a keyword from text."""
            pattern = rf'{keyword}[:\s]+(\d*\.?\d+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
                except:
                    pass
            return 0.7  # Default score
        
        accuracy = extract_score(response_text, 'accuracy')
        clarity = extract_score(response_text, 'clarity')
        completeness = extract_score(response_text, 'completeness')
        conciseness = extract_score(response_text, 'conciseness')
        
        # Calculate overall score
        overall = (accuracy + clarity + completeness + conciseness) / 4.0
        
        return PropositionQuality(
            accuracy=accuracy,
            clarity=clarity,
            completeness=completeness,
            conciseness=conciseness,
            overall=overall
        )
    
    def process_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[Document]:
        """
        Process documents into quality-checked propositions.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            List of proposition documents
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Initial chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.initial_chunk_size,
            chunk_overlap=200
        )
        initial_chunks = text_splitter.split_documents(documents)
        
        self.logger.info(f"Split into {len(initial_chunks)} initial chunks")
        
        # Generate and quality-check propositions
        all_propositions = []
        for chunk_idx, chunk in enumerate(initial_chunks):
            # Generate propositions
            propositions = self.generate_propositions(chunk.page_content)
            
            # Quality check each proposition
            for prop_idx, prop_text in enumerate(propositions):
                quality = self.check_quality(prop_text)
                
                if quality.overall >= self.quality_threshold:
                    prop_doc = Document(
                        page_content=prop_text,
                        metadata={
                            "source_chunk": chunk_idx,
                            "proposition_id": f"{chunk_idx}_{prop_idx}",
                            "quality_score": quality.overall,
                            "accuracy": quality.accuracy,
                            "clarity": quality.clarity,
                            "completeness": quality.completeness,
                            "conciseness": quality.conciseness,
                            **chunk.metadata
                        }
                    )
                    all_propositions.append(prop_doc)
                else:
                    self.logger.debug(
                        f"Proposition {prop_idx} from chunk {chunk_idx} "
                        f"failed quality check (score: {quality.overall:.2f})"
                    )
        
        self.propositions = all_propositions
        self.logger.info(
            f"Created {len(all_propositions)} quality-checked propositions "
            f"from {len(initial_chunks)} chunks"
        )
        
        return all_propositions
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        Create FAISS vector store from propositions.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            FAISS vector store instance
        """
        # Process into propositions
        propositions = self.process_documents(documents)
        
        if not propositions:
            raise ValueError("No valid propositions generated")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=propositions,
            embedding=self.embeddings
        )
        
        self.logger.info("Created proposition-based vector store")
        return self.vectorstore
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant propositions.
        
        Args:
            query: Query string
            top_k: Number of propositions to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of relevant proposition documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k
        
        # Retrieve similar propositions
        results = self.vectorstore.similarity_search(query, k=k)
        
        self.logger.info(f"Retrieved {len(results)} propositions")
        return results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_propositions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using proposition-based retrieval.
        
        Args:
            query: Query string
            top_k: Number of propositions to retrieve
            return_propositions: Whether to return proposition details
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer and proposition metadata
        """
        # Retrieve propositions
        propositions = self.retrieve(query, top_k=top_k)
        
        # Prepare context from propositions
        context = "\n\n".join([
            f"- {prop.page_content}"
            for prop in propositions
        ])
        
        # Generate answer
        answer = self._generate_answer(query, context)
        
        result = {
            "answer": answer,
            "num_propositions": len(propositions)
        }
        
        if return_propositions:
            result["propositions"] = [
                {
                    "content": prop.page_content,
                    "quality_score": prop.metadata.get("quality_score", 0),
                    "source_chunk": prop.metadata.get("source_chunk"),
                    "proposition_id": prop.metadata.get("proposition_id")
                }
                for prop in propositions
            ]
        
        return result
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using proposition-based context."""
        prompt = f"""Answer the question using the following factual propositions.
Each proposition is atomic and self-contained.

Propositions:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_proposition_stats(self) -> Dict[str, Any]:
        """
        Get statistics about propositions.
        
        Returns:
            Dictionary with proposition statistics
        """
        if not self.propositions:
            return {"error": "No propositions generated yet"}
        
        quality_scores = [
            prop.metadata.get("quality_score", 0)
            for prop in self.propositions
        ]
        
        return {
            "total_propositions": len(self.propositions),
            "avg_quality_score": sum(quality_scores) / len(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "quality_threshold": self.quality_threshold,
            "avg_proposition_length": sum(
                len(prop.page_content) for prop in self.propositions
            ) / len(self.propositions)
        }
