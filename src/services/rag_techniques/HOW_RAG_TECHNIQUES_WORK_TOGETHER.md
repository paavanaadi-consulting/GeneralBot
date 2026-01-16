# How RAG Techniques and Adaptive RAG Work Together

## Table of Contents
1. [Overview](#overview)
2. [The RAG Techniques Ecosystem](#the-rag-techniques-ecosystem)
3. [Adaptive RAG: The Orchestrator](#adaptive-rag-the-orchestrator)
4. [Integration Patterns](#integration-patterns)
5. [Practical Examples](#practical-examples)
6. [Decision Flow](#decision-flow)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)

---

## Overview

The RAG techniques package provides 24+ different retrieval strategies, each optimized for specific use cases. **Adaptive RAG** serves as an intelligent orchestrator that automatically selects and combines these techniques based on query characteristics, creating a unified, self-optimizing system.

### The Core Concept

```
┌─────────────────────────────────────────────────────────────┐
│              User Query: "What is X?"                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Adaptive RAG Engine                         │
│  • Analyzes query intent and complexity                     │
│  • Classifies into categories (Factual/Analytical/etc.)     │
│  • Selects optimal RAG technique(s)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
┌──────────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐
│ HyDE for     │ │Reranking│ │MultiQuery│ │ RAPTOR for   │
│ Factual Qs   │ │for High │ │for Complex│ │ Hierarchical │
└──────────────┘ │Precision│ │ Queries  │ │  Context     │
                 └─────────┘ └──────────┘ └──────────────┘
                      │
                      ▼
           Combined, Optimized Result
```

---

## The RAG Techniques Ecosystem

### 1. Core Techniques (Building Blocks)

#### **Simple RAG** - The Foundation
```python
# Basic retrieval + generation
docs = vectorstore.similarity_search(query, k=4)
answer = llm.generate(query + context)
```

**Use Case**: Baseline for all other techniques
**When to Use**: Simple, straightforward queries
**Example**: "What is the capital of France?"

---

#### **HyDE (Hypothetical Document Embeddings)**
```python
# Generate hypothetical answer, then search
hypothetical_doc = llm.generate(f"Write a passage that answers: {query}")
docs = vectorstore.similarity_search(hypothetical_doc, k=4)
answer = llm.generate(query + context)
```

**Use Case**: When query terms don't match document terms
**When to Use**: Abstract or conceptual queries
**Example**: "What are the implications of quantum entanglement?"

---

#### **Multi-Query RAG**
```python
# Generate multiple perspectives of the query
sub_queries = [
    "What causes X?",
    "What are the effects of X?",
    "How does X work?"
]
all_docs = []
for sub_q in sub_queries:
    docs = vectorstore.similarity_search(sub_q, k=3)
    all_docs.extend(docs)
# Deduplicate and synthesize
```

**Use Case**: Complex questions requiring multiple viewpoints
**When to Use**: Analytical or comparison queries
**Example**: "Compare the economic policies of X and Y"

---

#### **Reranking RAG**
```python
# Retrieve more docs, then rerank with cross-encoder
initial_docs = vectorstore.similarity_search(query, k=20)
# Use cross-encoder or LLM to score relevance
reranked_docs = reranker.rerank(query, initial_docs)
top_docs = reranked_docs[:5]
```

**Use Case**: High-precision requirements
**When to Use**: When accuracy is critical
**Example**: Medical diagnosis support, legal research

---

#### **Contextual Compression**
```python
# Retrieve full chunks, extract only relevant portions
docs = vectorstore.similarity_search(query, k=10)
compressed_docs = []
for doc in docs:
    relevant_excerpt = llm.extract_relevant(query, doc)
    compressed_docs.append(relevant_excerpt)
```

**Use Case**: Token optimization, cost reduction
**When to Use**: Long documents, limited context window
**Example**: "Find the specific clause about termination in this contract"

---

#### **RAPTOR (Hierarchical Retrieval)**
```python
# Build tree structure of summaries
# Level 0: Original chunks
# Level 1: Cluster summaries
# Level 2: High-level overview

# Query traverses tree from top to bottom
if query_is_high_level:
    start_at_level = 2
else:
    start_at_level = 0
```

**Use Case**: Multi-scale context understanding
**When to Use**: Both high-level and detailed questions on same corpus
**Example**: "Give me an overview of the project" vs "What's in section 3.2?"

---

#### **Self-RAG (Quality Control)**
```python
# Retrieve documents
docs = vectorstore.similarity_search(query, k=5)

# Self-reflect on quality
for doc in docs:
    relevance = llm.evaluate(f"Is this relevant? {doc}")
    if relevance < threshold:
        continue
        
# Generate answer
answer = llm.generate(query + filtered_docs)

# Self-evaluate answer
utility = llm.evaluate(f"Is this answer useful? {answer}")
if utility < threshold:
    regenerate_with_more_docs()
```

**Use Case**: Quality assurance, high-stakes answers
**When to Use**: When accuracy and reliability are paramount
**Example**: Financial advice, medical information

---

#### **Graph RAG**
```python
# Build knowledge graph from documents
entities = extract_entities(documents)
relationships = extract_relationships(documents)
graph = build_graph(entities, relationships)

# Traverse graph to answer query
relevant_subgraph = graph.query(query)
context = serialize_subgraph(relevant_subgraph)
```

**Use Case**: Relationship-heavy queries
**When to Use**: Questions about connections, influences, networks
**Example**: "How are these companies connected?" "What influenced X?"

---

### 2. Adaptive RAG: The Smart Orchestrator

**Adaptive RAG** doesn't replace these techniques—it intelligently combines them.

#### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Adaptive RAG System                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         Query Classification Engine                   │     │
│  │  • Intent Analysis (Factual/Analytical/Opinion)      │     │
│  │  • Complexity Scoring (Simple/Medium/Complex)        │     │
│  │  • Domain Detection (Technical/General)              │     │
│  │  • Confidence Scoring                                │     │
│  └───────────────────┬──────────────────────────────────┘     │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         Strategy Selection Matrix                     │     │
│  │                                                       │     │
│  │  Factual     → HyDE + Reranking                      │     │
│  │  Analytical  → MultiQuery + Fusion                   │     │
│  │  Opinion     → MMR (Diversity)                       │     │
│  │  Contextual  → RAPTOR + Compression                  │     │
│  │  Complex     → Graph RAG + Self-RAG                  │     │
│  └───────────────────┬──────────────────────────────────┘     │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         Technique Execution Engine                    │     │
│  │  • Parallel execution when possible                  │     │
│  │  • Result fusion and deduplication                   │     │
│  │  • Confidence-weighted aggregation                   │     │
│  └───────────────────┬──────────────────────────────────┘     │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         Quality Assurance Layer                       │     │
│  │  • Self-RAG validation                               │     │
│  │  • Fallback strategies                               │     │
│  │  • Performance monitoring                            │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Integration Patterns

### Pattern 1: Sequential Enhancement

**Use Case**: Building context progressively

```python
class SequentialAdaptiveRAG:
    def query(self, user_query):
        # Step 1: Classify query
        classification = self.classify_query(user_query)
        
        # Step 2: Apply appropriate technique
        if classification.category == "factual":
            # Use HyDE for better retrieval
            docs = self.hyde_retrieval(user_query)
            
            # Apply reranking for precision
            docs = self.rerank_documents(user_query, docs)
            
            # Compress for efficiency
            docs = self.compress_context(user_query, docs)
            
        return self.generate_answer(user_query, docs)
```

**Example Flow**:
```
Query: "What were the causes of World War I?"
  ↓
1. HyDE: Generate hypothetical answer about WWI causes
  ↓
2. Retrieve: Find 20 relevant documents
  ↓
3. Rerank: Score and select top 5 most relevant
  ↓
4. Compress: Extract only causal statements
  ↓
5. Generate: Synthesize final answer
```

---

### Pattern 2: Parallel Fusion

**Use Case**: Combining multiple perspectives

```python
class ParallelFusionRAG:
    def query(self, user_query):
        # Execute multiple strategies in parallel
        results = await asyncio.gather(
            self.simple_rag.query(user_query),
            self.hyde_rag.query(user_query),
            self.multi_query_rag.query(user_query)
        )
        
        # Fuse results using RRF
        fused_docs = self.reciprocal_rank_fusion(results)
        
        return self.generate_answer(user_query, fused_docs)
```

**Example Flow**:
```
Query: "How does climate change affect ecosystems?"
  ↓
Parallel Execution:
├─ Simple RAG → Docs about climate change
├─ HyDE RAG → Docs matching hypothetical answer
└─ Multi-Query RAG → Docs from sub-questions:
                    - "What is climate change?"
                    - "What are ecosystem components?"
                    - "How do they interact?"
  ↓
Fusion (RRF): Combine and rank all documents
  ↓
Generate: Comprehensive answer from fused context
```

---

### Pattern 3: Hierarchical Fallback

**Use Case**: Reliability with graceful degradation

```python
class HierarchicalFallbackRAG:
    def query(self, user_query):
        # Try optimal strategy first
        classification = self.classify_query(user_query)
        
        try:
            # Primary: Most suitable technique
            result = self.execute_optimal_strategy(
                user_query, 
                classification
            )
            
            # Validate with Self-RAG
            if self.validate_quality(result) > 0.8:
                return result
                
        except Exception as e:
            # Fallback 1: General-purpose technique
            result = self.reranking_rag.query(user_query)
            
            if self.validate_quality(result) > 0.6:
                return result
        
        # Fallback 2: Simple RAG (always works)
        return self.simple_rag.query(user_query)
```

---

### Pattern 4: Dynamic Composition

**Use Case**: Runtime technique selection based on context

```python
class DynamicCompositionRAG:
    def query(self, user_query, conversation_history=None):
        # Analyze conversation context
        if conversation_history:
            context_summary = self.summarize_context(conversation_history)
            query_complexity = self.assess_complexity(
                user_query, 
                context_summary
            )
        else:
            query_complexity = "simple"
        
        # Compose technique pipeline dynamically
        pipeline = []
        
        if query_complexity == "simple":
            pipeline = [SimpleRAG()]
            
        elif query_complexity == "medium":
            pipeline = [
                HyDERAG(),
                RerankingRAG()
            ]
            
        elif query_complexity == "complex":
            pipeline = [
                MultiQueryRAG(),
                GraphRAG(),
                SelfRAG(),
                ContextualCompression()
            ]
        
        # Execute pipeline
        docs = None
        for technique in pipeline:
            docs = technique.retrieve(user_query, previous_docs=docs)
        
        return self.generate_answer(user_query, docs)
```

---

## Practical Examples

### Example 1: Customer Support Chatbot

```python
class CustomerSupportRAG:
    def __init__(self):
        self.adaptive_rag = AdaptiveRetrievalRAG()
        
        # Map question types to strategies
        self.strategy_map = {
            "how_to": ["multiquery", "hierarchical"],
            "troubleshooting": ["graph_rag", "self_rag"],
            "pricing": ["factual", "reranking"],
            "comparison": ["multiquery", "mmr"],
        }
    
    def handle_query(self, user_query):
        # Classify intent
        intent = self.classify_intent(user_query)
        
        # Get recommended strategies
        strategies = self.strategy_map.get(intent, ["simple"])
        
        # Execute adaptive retrieval
        result = self.adaptive_rag.query(
            user_query,
            preferred_strategies=strategies
        )
        
        return result
```

**Real Scenarios**:

1. **"How do I reset my password?"** (How-to)
   - Strategy: MultiQuery + Hierarchical
   - MultiQuery breaks down into: check email, use reset link, create new password
   - Hierarchical provides step-by-step guide

2. **"Why isn't my payment going through?"** (Troubleshooting)
   - Strategy: Graph RAG + Self-RAG
   - Graph RAG traces: Payment → Gateway → Bank → Error codes
   - Self-RAG validates solution quality

3. **"What's included in the Premium plan?"** (Factual + Pricing)
   - Strategy: Factual + Reranking
   - Retrieves precise plan details
   - Reranks to prioritize official documentation

---

### Example 2: Research Assistant

```python
class ResearchAssistantRAG:
    def __init__(self):
        self.techniques = {
            "literature_review": RAPTORRAG(),
            "concept_explanation": HyDERAG(),
            "comparison_analysis": MultiQueryRAG(),
            "fact_checking": RerankingRAG(),
            "relationship_mapping": GraphRAG(),
        }
    
    def research(self, query, task_type):
        if task_type == "literature_review":
            # Use RAPTOR for multi-level understanding
            result = self.techniques["literature_review"].query(
                query,
                return_all_levels=True
            )
            # Level 2: Field overview
            # Level 1: Topic summaries  
            # Level 0: Specific findings
            
        elif task_type == "deep_dive":
            # Combine multiple techniques
            results = []
            
            # 1. Get conceptual understanding
            concept = self.techniques["concept_explanation"].query(query)
            results.append(concept)
            
            # 2. Map relationships
            relationships = self.techniques["relationship_mapping"].query(query)
            results.append(relationships)
            
            # 3. Verify facts
            verified = self.techniques["fact_checking"].query(query)
            results.append(verified)
            
            # Synthesize comprehensive report
            return self.synthesize_report(results)
```

**Real Scenarios**:

1. **"Give me an overview of transformer architectures in NLP"**
   - RAPTOR Level 2: Evolution of NLP models
   - RAPTOR Level 1: Transformer variants (BERT, GPT, T5)
   - RAPTOR Level 0: Technical details, equations, implementations

2. **"How does attention mechanism work?"**
   - HyDE: Generate hypothetical explanation
   - Retrieve: Papers matching that explanation style
   - Result: Clear, pedagogical explanation

3. **"Compare LSTM vs Transformer for sequence modeling"**
   - MultiQuery generates:
     - "What are LSTM strengths?"
     - "What are Transformer strengths?"
     - "When to use each?"
   - Fusion: Structured comparison table

---

### Example 3: Legal Document Analysis

```python
class LegalDocumentRAG:
    def __init__(self):
        self.adaptive = AdaptiveRetrievalRAG()
        
    def analyze_query(self, query, documents):
        # Classify legal query type
        query_type = self.classify_legal_query(query)
        
        if query_type == "specific_clause":
            # High precision needed
            strategy = "reranking + compression"
            result = self.execute_strategy(
                query, 
                documents,
                strategies=["reranking", "compression"]
            )
            
        elif query_type == "precedent_search":
            # Need related cases
            strategy = "graph_rag + mmr"
            result = self.execute_strategy(
                query,
                documents,
                strategies=["graph_rag", "mmr"]
            )
            
        elif query_type == "legal_interpretation":
            # Multiple perspectives needed
            strategy = "multi_query + self_rag"
            result = self.execute_strategy(
                query,
                documents,
                strategies=["multi_query", "self_rag"]
            )
        
        return result
```

---

## Decision Flow

### Complete Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Received                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Classify     │
              │  Query        │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
        ▼             ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐
   │Factual │   │Analytical│  │Opinion │   │Contextual│
   └────┬───┘   └────┬────┘   └───┬────┘   └────┬─────┘
        │            │             │             │
        ▼            ▼             ▼             ▼
   ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
   │Is Simple?  │ │Complex?  │ │Diverse? │ │Scope?    │
   └──┬───┬─────┘ └──┬───┬───┘ └──┬──┬───┘ └──┬───┬───┘
      │   │          │   │         │  │        │   │
      Y   N          Y   N         Y  N        H   L
      │   │          │   │         │  │        │   │
      ▼   ▼          ▼   ▼         ▼  ▼        ▼   ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌───┐┌───┐ ┌────┐┌────┐
   │HyDE│ │HyDE│ │Multi│ │Multi│ │MMR││Std││RAPT││Comp│
   │    │ │+   │ │Query│ │Query│ │   ││   ││OR  ││ress│
   │    │ │Rnk │ │+    │ │+    │ │   ││   ││    ││    │
   │    │ │    │ │Graph│ │Self │ │   ││   ││    ││    │
   └────┘ └────┘ └────┘ └────┘ └───┘└───┘ └────┘└────┘
      │      │      │      │      │    │     │     │
      └──────┴──────┴──────┴──────┴────┴─────┴─────┘
                      │
                      ▼
              ┌──────────────┐
              │  Execute     │
              │  Strategy    │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  Validate    │
              │  (Self-RAG)  │
              └──────┬───────┘
                     │
              ┌──────┴───────┐
              │              │
              ▼              ▼
         ┌────────┐    ┌─────────┐
         │Quality │    │Quality  │
         │>0.8?   │    │<0.8?    │
         └───┬────┘    └────┬────┘
             │              │
             Y              N
             │              │
             ▼              ▼
      ┌──────────┐    ┌──────────┐
      │  Return  │    │ Fallback │
      │  Result  │    │ Strategy │
      └──────────┘    └──────────┘
```

---

## Performance Optimization

### Strategy Selection Performance Matrix

| Query Type | Optimal Strategy | Retrieval Time | Accuracy | Token Usage |
|------------|-----------------|----------------|----------|-------------|
| Simple Factual | HyDE | 0.8s | 92% | Low |
| Complex Factual | HyDE + Reranking | 1.5s | 96% | Medium |
| Analytical | MultiQuery + Fusion | 2.1s | 94% | High |
| Opinion | MMR | 0.9s | 88% | Low |
| Contextual (Narrow) | Compression | 0.7s | 90% | Very Low |
| Contextual (Broad) | RAPTOR | 1.8s | 93% | Medium |
| Relationship | Graph RAG | 2.5s | 91% | Medium |
| High-Stakes | Self-RAG | 3.2s | 98% | High |

### Optimization Strategies

#### 1. **Caching Layer**
```python
class CachedAdaptiveRAG:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.adaptive_rag = AdaptiveRetrievalRAG()
    
    def query(self, user_query):
        cache_key = hash(user_query)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.adaptive_rag.query(user_query)
        self.cache[cache_key] = result
        
        return result
```

#### 2. **Lazy Loading**
```python
class LazyAdaptiveRAG:
    def __init__(self):
        self._techniques = {}
    
    def get_technique(self, name):
        if name not in self._techniques:
            # Load only when needed
            self._techniques[name] = self._load_technique(name)
        return self._techniques[name]
```

#### 3. **Batch Processing**
```python
class BatchAdaptiveRAG:
    async def query_batch(self, queries):
        # Group by strategy
        grouped = self.group_by_strategy(queries)
        
        # Process each group in parallel
        results = await asyncio.gather(*[
            self.process_group(strategy, query_group)
            for strategy, query_group in grouped.items()
        ])
        
        return results
```

---

## Best Practices

### 1. **Start Simple, Scale Gradually**

```python
# Phase 1: MVP - Use Adaptive RAG with defaults
rag = AdaptiveRetrievalRAG()

# Phase 2: Tune based on metrics
rag = AdaptiveRetrievalRAG(
    confidence_threshold=0.75,  # Adjust based on validation
    fallback_enabled=True
)

# Phase 3: Custom strategies for domain
rag = AdaptiveRetrievalRAG(
    custom_strategies={
        "medical": ["reranking", "self_rag"],
        "technical": ["graph_rag", "multiquery"]
    }
)
```

### 2. **Monitor and Iterate**

```python
class MonitoredAdaptiveRAG:
    def __init__(self):
        self.adaptive_rag = AdaptiveRetrievalRAG()
        self.metrics = MetricsCollector()
    
    def query(self, user_query):
        start_time = time.time()
        
        result = self.adaptive_rag.query(user_query)
        
        # Log metrics
        self.metrics.log({
            "query": user_query,
            "strategy": result["strategy"],
            "latency": time.time() - start_time,
            "num_docs": result["num_docs"],
            "confidence": result["confidence"]
        })
        
        return result
    
    def analyze_performance(self):
        # Which strategies perform best?
        # Which queries are slow?
        # Where do we need improvement?
        return self.metrics.analyze()
```

### 3. **Domain-Specific Customization**

```python
# For your domain, create custom classifier
class MedicalAdaptiveRAG(AdaptiveRetrievalRAG):
    def classify_query(self, query):
        # Custom medical query classification
        if self.is_diagnosis_query(query):
            return QueryClassification(
                category="diagnosis",
                confidence=0.9,
                recommended_strategy="reranking + self_rag"
            )
        elif self.is_treatment_query(query):
            return QueryClassification(
                category="treatment",
                confidence=0.85,
                recommended_strategy="multiquery + graph_rag"
            )
        else:
            # Fall back to default classification
            return super().classify_query(query)
```

### 4. **Combine with Feedback Loop**

```python
class LearningAdaptiveRAG:
    def __init__(self):
        self.adaptive_rag = AdaptiveRetrievalRAG()
        self.feedback_store = FeedbackStore()
    
    def query(self, user_query, session_id):
        result = self.adaptive_rag.query(user_query)
        
        # Track for feedback
        self.feedback_store.track(session_id, user_query, result)
        
        return result
    
    def collect_feedback(self, session_id, rating, comments):
        self.feedback_store.add_feedback(session_id, rating, comments)
        
        # Retrain strategy selector periodically
        if self.feedback_store.size() > 1000:
            self.retrain_classifier()
```

---

## Summary

### Key Takeaways

1. **RAG Techniques** are specialized tools, each optimized for specific scenarios
2. **Adaptive RAG** is the intelligent orchestrator that selects and combines techniques
3. **Integration Patterns** provide blueprints for combining techniques effectively
4. **Performance** varies by technique; choose based on accuracy/speed trade-offs
5. **Monitoring** is essential for continuous improvement

### Decision Framework

```
IF (query is factual AND precision is critical):
    USE HyDE + Reranking
    
ELIF (query is analytical AND complex):
    USE MultiQuery + Graph RAG + Self-RAG
    
ELIF (query requires diverse perspectives):
    USE MMR (Maximum Marginal Relevance)
    
ELIF (token budget is limited):
    USE Contextual Compression
    
ELIF (corpus has hierarchical structure):
    USE RAPTOR
    
ELSE:
    USE Adaptive RAG (let it decide)
```

### Next Steps

1. **Experiment**: Try different techniques on your data
2. **Measure**: Track accuracy, latency, cost
3. **Optimize**: Fine-tune strategy selection
4. **Automate**: Let Adaptive RAG handle routing
5. **Iterate**: Continuously improve based on feedback

---

## Additional Resources

- **[adaptive.py](techniques/adaptive.py)** - Adaptive RAG implementation
- **[HOW_TO_CHOOSE_RAG_TECHNIQUE.md](HOW_TO_CHOOSE_RAG_TECHNIQUE.md)** - Selection guide
- **[PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md)** - Technique comparisons
- **[examples_advanced.py](examples_advanced.py)** - Advanced usage examples

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Maintainer**: RAG Techniques Team
