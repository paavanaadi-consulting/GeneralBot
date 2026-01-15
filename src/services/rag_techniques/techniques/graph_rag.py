"""
Graph RAG - Graph-Enhanced Retrieval-Augmented Generation

This module implements GraphRAG which constructs a knowledge graph from documents
and uses graph traversal for enhanced retrieval and context-aware answering.
"""

import networkx as nx
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import defaultdict
import heapq
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from ..core.base import BaseRAGTechnique
from ..core.config import ConfigManager


class GraphRAG(BaseRAGTechnique):
    """
    Graph-enhanced RAG with knowledge graph construction and traversal.
    
    Creates a knowledge graph where:
    - Nodes represent document chunks
    - Edges represent relationships (semantic similarity, shared concepts)
    - Traversal follows relevant paths to gather context
    
    Attributes:
        embeddings: OpenAI embeddings model
        vectorstore: FAISS vector store
        graph: NetworkX graph structure
        node_content: Mapping of node IDs to content
        concepts: Extracted concepts per node
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        max_graph_depth: int = 2,
        top_k_initial: int = 3
    ):
        """
        Initialize Graph RAG.
        
        Args:
            config: Configuration manager instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            similarity_threshold: Threshold for creating edges
            max_graph_depth: Maximum depth for graph traversal
            top_k_initial: Number of initial nodes to retrieve
        """
        super().__init__(config)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_graph_depth = max_graph_depth
        self.top_k_initial = top_k_initial
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key
        )
        
        self.vectorstore: Optional[FAISS] = None
        self.graph = nx.Graph()
        self.node_content: Dict[int, str] = {}
        self.concepts: Dict[int, Set[str]] = {}
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract 3-5 key concepts or entities from the following text.
Return them as a comma-separated list.

Text: {text}

Concepts:"""
        )
        
        self.completeness_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Given the question and current context, is the context sufficient to provide a complete answer?
Answer with 'COMPLETE' or 'INCOMPLETE'.

Question: {query}
Context: {context}

Assessment:"""
        )
    
    def extract_concepts(self, text: str) -> Set[str]:
        """
        Extract key concepts from text.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            Set of concept strings
        """
        # Limit text length
        text_preview = text[:500]
        
        prompt = self.concept_extraction_prompt.format(text=text_preview)
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse concepts
        concepts = set()
        for concept in response_text.split(','):
            concept = concept.strip().lower()
            if concept and len(concept) > 2:
                concepts.add(concept)
        
        return concepts
    
    def build_knowledge_graph(
        self,
        documents: List[Document]
    ) -> nx.Graph:
        """
        Build knowledge graph from documents.
        
        Args:
            documents: List of document chunks
            
        Returns:
            NetworkX graph
        """
        self.graph = nx.Graph()
        
        # Add nodes
        for idx, doc in enumerate(documents):
            self.graph.add_node(idx)
            self.node_content[idx] = doc.page_content
            
            # Extract concepts
            concepts = self.extract_concepts(doc.page_content)
            self.concepts[idx] = concepts
        
        self.logger.info(f"Added {len(documents)} nodes to graph")
        
        # Add edges based on concept similarity
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                # Calculate concept overlap
                concepts_i = self.concepts[i]
                concepts_j = self.concepts[j]
                
                if concepts_i and concepts_j:
                    overlap = len(concepts_i & concepts_j)
                    union = len(concepts_i | concepts_j)
                    
                    if union > 0:
                        similarity = overlap / union
                        
                        if similarity >= self.similarity_threshold:
                            self.graph.add_edge(i, j, weight=similarity)
        
        num_edges = self.graph.number_of_edges()
        self.logger.info(f"Added {num_edges} edges to graph")
        
        return self.graph
    
    def traverse_graph(
        self,
        query: str,
        start_nodes: List[int],
        max_depth: Optional[int] = None
    ) -> Tuple[List[int], List[str]]:
        """
        Traverse graph to gather relevant context.
        
        Args:
            query: User query
            start_nodes: Initial nodes to start from
            max_depth: Maximum traversal depth
            
        Returns:
            Tuple of (visited_node_ids, traversal_path)
        """
        depth = max_depth or self.max_graph_depth
        visited = set()
        context_nodes = []
        path = []
        
        # Priority queue: (priority, node_id)
        # Lower priority = higher relevance
        pq = [(0, node_id) for node_id in start_nodes]
        heapq.heapify(pq)
        
        visited_concepts = set()
        
        while pq and len(context_nodes) < self.top_k_initial * 3:
            priority, node_id = heapq.heappop(pq)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            context_nodes.append(node_id)
            path.append(f"Node {node_id}")
            
            # Check if we have enough context
            current_context = "\n".join([self.node_content[n] for n in context_nodes])
            if self._is_context_complete(query, current_context):
                self.logger.info(f"Found complete answer after {len(context_nodes)} nodes")
                break
            
            # Explore neighbors
            if len(path) < depth:
                node_concepts = self.concepts.get(node_id, set())
                visited_concepts.update(node_concepts)
                
                for neighbor in self.graph.neighbors(node_id):
                    if neighbor not in visited:
                        # Calculate priority based on edge weight and concept novelty
                        edge_weight = self.graph[node_id][neighbor]['weight']
                        neighbor_concepts = self.concepts.get(neighbor, set())
                        new_concepts = neighbor_concepts - visited_concepts
                        
                        # Lower priority = more relevant
                        priority_score = 1.0 - (edge_weight * 0.7 + len(new_concepts) * 0.3)
                        heapq.heappush(pq, (priority_score, neighbor))
        
        self.logger.info(f"Traversed {len(context_nodes)} nodes")
        return context_nodes, path
    
    def _is_context_complete(self, query: str, context: str) -> bool:
        """Check if context is sufficient to answer query."""
        if len(context) < 100:  # Too short
            return False
        
        prompt = self.completeness_check_prompt.format(
            query=query,
            context=context[:1000]  # Limit for efficiency
        )
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return 'COMPLETE' in response_text.upper()
    
    def create_vectorstore(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        Create vector store and knowledge graph.
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            FAISS vector store instance
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Build knowledge graph
        self.build_knowledge_graph(chunks)
        
        self.logger.info("Created vector store and knowledge graph")
        return self.vectorstore
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_graph_info: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query using graph-enhanced retrieval.
        
        Args:
            query: Query string
            top_k: Number of initial nodes to retrieve
            return_graph_info: Whether to return graph traversal info
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer and graph metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        k = top_k or self.top_k_initial
        
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=k)
        
        # Map documents to node IDs
        start_nodes = []
        for doc in initial_docs:
            # Find matching node by content
            for node_id, content in self.node_content.items():
                if content == doc.page_content:
                    start_nodes.append(node_id)
                    break
        
        if not start_nodes:
            # Fallback: use first k nodes
            start_nodes = list(range(min(k, len(self.node_content))))
        
        # Traverse graph
        visited_nodes, traversal_path = self.traverse_graph(query, start_nodes)
        
        # Gather context from visited nodes
        context = "\n\n".join([
            f"[Node {node_id}]: {self.node_content[node_id]}"
            for node_id in visited_nodes
        ])
        
        # Generate answer
        answer = self._generate_answer(query, context)
        
        result = {
            "answer": answer,
            "num_nodes_visited": len(visited_nodes),
            "num_initial_nodes": len(start_nodes)
        }
        
        if return_graph_info:
            result["graph_info"] = {
                "start_nodes": start_nodes,
                "visited_nodes": visited_nodes,
                "traversal_path": traversal_path,
                "graph_stats": {
                    "total_nodes": self.graph.number_of_nodes(),
                    "total_edges": self.graph.number_of_edges(),
                    "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
                }
            }
        
        return result
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using graph-traversed context."""
        prompt = f"""Answer the question using the following context gathered through knowledge graph traversal.
Each piece of context is labeled with its graph node ID.

Context from Knowledge Graph:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return {"error": "Graph not built yet"}
        
        degrees = dict(self.graph.degree())
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": sum(degrees.values()) / len(degrees),
            "max_degree": max(degrees.values()) if degrees else 0,
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "num_connected_components": nx.number_connected_components(self.graph),
            "similarity_threshold": self.similarity_threshold
        }
    
    def visualize_graph(self, query: Optional[str] = None, save_path: Optional[str] = None):
        """
        Visualize the knowledge graph (requires matplotlib).
        
        Args:
            query: Optional query to highlight traversal path
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_color='lightblue',
                node_size=500,
                alpha=0.7
            )
            
            # Draw edges with weights
            edges = self.graph.edges()
            weights = [self.graph[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(
                self.graph, pos,
                width=[w * 3 for w in weights],
                alpha=0.5
            )
            
            # Draw labels
            labels = {i: f"N{i}" for i in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            plt.title("Knowledge Graph")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved graph visualization to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            self.logger.warning("matplotlib not available for visualization")
