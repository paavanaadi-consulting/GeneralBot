# RAG Techniques - Reusable Python Package

A comprehensive, production-ready Python package implementing 24 advanced Retrieval-Augmented Generation (RAG) techniques.

**Coverage:** 24/27 Techniques (88.9% total, 100% core) | **Status:** ✅ Production Ready | **Version:** 1.0.0

## 🎉 Latest Update (December 2024)

**3 NEW CUTTING-EDGE TECHNIQUES ADDED:**
- ✨ **ReliableRAG** - Document quality validation and grading
- ✨ **DartboardRAG** - Balanced relevance-diversity retrieval
- ✨ **DocumentAugmentation** - Question generation for enhanced matching

**See [examples_latest.py](examples_latest.py) for demonstrations!**

## 🎯 Overview

This package provides ready-to-use implementations of advanced RAG techniques from research papers and the RAG_Techniques repository. All techniques follow a consistent API and are production-ready with comprehensive documentation.

## ✨ Features - 24 Techniques Implemented

### Core RAG (100% Coverage)
- ✅ **Simple RAG**: Foundation for all techniques

### Query Enhancement (100% Coverage)  
- ✅ **Query Transformation**: Multi-query, decomposition, step-back, HyPE
- ✅ **Fusion Retrieval**: Reciprocal Rank Fusion (RRF)
- ✅ **HyDE**: Hypothetical Document Embedding

### Context Enhancement (100% Coverage)
- ✅ **Contextual Compression**: Extract relevant portions
- ✅ **Reranking**: Cross-encoder reranking
- ✅ **Contextual Chunk Headers**: Document title/summary prepending
- ✅ **Document Augmentation**: Question generation ✨ NEW

### Quality & Reliability (100% Coverage)
- ✅ **Reliable RAG**: Document grading and validation ✨ NEW
- ✅ **Dartboard RAG**: Relevance-diversity balance ✨ NEW

### Chunking Strategies (100% Coverage)
- ✅ **Semantic Chunking**: Embedding-based boundary detection
- ✅ **Proposition Chunking**: Atomic factual propositions

### Hierarchical Retrieval (100% Coverage)
- ✅ **Hierarchical Indices**: Two-tier (summaries + chunks)
- ✅ **RAPTOR**: Recursive tree with multi-level summaries

### Adaptive Systems (100% Coverage)
- ✅ **Adaptive RAG**: Query classification & strategy selection
- ✅ **Self-RAG**: Self-reflective with quality checks
- ✅ **Corrective RAG (CRAG)**: Dynamic correction with web search
- ✅ **Feedback Loop**: Iterative refinement

### Graph-Based (100% Coverage)
- ✅ **Graph RAG**: Knowledge graph construction and traversal

### Evaluation Tools
- ✅ **Metrics**: BLEU, ROUGE, Answer Relevancy, Faithfulness, Context Precision/Recall

### Not Included (By Design)
- 🔴 **Multi-modal RAG** (Captioning, ColPali) - External service dependencies
- 🔴 **Agentic RAG** - Proprietary platform (can be composed from existing techniques)

**See [FINAL_COVERAGE_REPORT.md](FINAL_COVERAGE_REPORT.md) for complete details.**

## 📦 Installation

```bash
# Clone or download the package
cd rag_techniques_reusable

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Set up environment
export OPENAI_API_KEY="your-key-here"
```

## 🚀 Quick Start

### 1. Simple RAG

```python
from rag_techniques.core import SimpleRAG
from langchain.schema import Document

# Create documents
docs = [Document(page_content="Your content here")]

# Initialize and query
rag = SimpleRAG()
rag.create_vectorstore(docs)
result = rag.query("Your question")
print(result['answer'])
```

### 2. Semantic Chunking

```python
from rag_techniques.techniques import SemanticChunkingRAG

rag = SemanticChunkingRAG(
    breakpoint_type="percentile",
    breakpoint_threshold=90
)
rag.create_vectorstore(docs)
result = rag.query("Your question")
```

### 3. RAPTOR (Hierarchical Tree)

```python
from rag_techniques.techniques import RAPTORRAG

rag = RAPTORRAG(
    max_levels=3,
    n_clusters=5
)
rag.create_vectorstore(docs)
result = rag.query("Your question", return_context=True)
print(f"Used levels: {result['levels_used']}")
```

### 4. Self-RAG (Quality-Focused)

```python
from rag_techniques.techniques import SelfRAG

rag = SelfRAG(relevance_threshold=0.7)
rag.create_vectorstore(docs)
result = rag.query("Your question", return_reflections=True)
print(f"Quality score: {result['reflections']['utility']}")
```

### 5. Adaptive RAG (Auto-Optimization)

```python
from rag_techniques.techniques import AdaptiveRAG

rag = AdaptiveRAG()
rag.create_vectorstore(docs)
result = rag.query("Your question", return_classification=True)
print(f"Strategy: {result['query_classification']['strategy']}")
```

# Initialize with feedback
feedback_rag = FeedbackRAG(pdf_path="document.pdf")

# Query and collect feedback
response = feedback_rag.query("What causes climate change?")

# Provide feedback
feedback_rag.add_feedback(
    query="What causes climate change?",
    response=response,
    relevance=5,
    quality=5
)

# System automatically improves over time
```

## Package Structure

```
rag_techniques/
├── __init__.py              # Package initialization
├── core/                    # Core RAG implementations
│   ├── __init__.py
│   ├── simple_rag.py        # Basic RAG
│   ├── base.py              # Base classes
│   └── config.py            # Configuration
├── techniques/              # Advanced techniques
│   ├── __init__.py
│   ├── compression.py       # Contextual compression
│   ├── query_transform.py   # Query transformations
│   ├── reranking.py         # Reranking strategies
│   ├── fusion.py            # Fusion retrieval
│   ├── hierarchical.py      # Hierarchical indices
│   ├── adaptive.py          # Adaptive retrieval
│   ├── graph_rag.py         # Graph-based RAG
│   ├── self_rag.py          # Self-reflective RAG
│   ├── crag.py              # Corrective RAG
│   ├── raptor.py            # RAPTOR
│   └── feedback.py          # Feedback loop
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── document_loaders.py  # Document loading
│   ├── text_splitters.py    # Text chunking
│   ├── embeddings.py        # Embedding providers
│   ├── vector_stores.py     # Vector store utilities
│   └── helpers.py           # Helper functions
├── evaluation/              # Evaluation tools
│   ├── __init__.py
│   ├── metrics.py           # Evaluation metrics
│   └── evaluators.py        # Evaluator classes
└── cli.py                   # Command-line interface
```

## Configuration

```python
from rag_techniques.config import RAGConfig

config = RAGConfig(
    # Chunking parameters
    chunk_size=1000,
    chunk_overlap=200,
    
    # Retrieval parameters
    n_retrieved=2,
    search_type="similarity",
    
    # LLM parameters
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=4000,
    
    # Embedding parameters
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
```

## CLI Usage

```bash
# Simple RAG
rag-simple --path document.pdf --query "What is climate change?"

# With evaluation
rag-simple --path document.pdf --query "What is climate change?" --evaluate

# Custom parameters
rag-simple --path document.pdf --chunk-size 1500 --n-retrieved 3
```

## 📚 Documentation

### Complete Documentation Suite

| Document | Purpose | Best For |
|----------|---------|----------|
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | Navigation guide | Finding what you need |
| **[COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)** | Big picture overview | Understanding everything |
| **[FINAL_COVERAGE_REPORT.md](FINAL_COVERAGE_REPORT.md)** | Coverage analysis | Seeing what's available |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Code snippets | Quick examples |
| **[PACKAGE_GUIDE.md](PACKAGE_GUIDE.md)** | API reference | Detailed documentation |

### Example Scripts

| Script | Purpose |
|--------|---------|
| **[example_usage.py](example_usage.py)** | Basic usage examples |
| **[examples_advanced.py](examples_advanced.py)** | Advanced techniques |
| **[examples_latest.py](examples_latest.py)** | Latest 3 techniques ✨ |
| **[verify_installation.py](verify_installation.py)** | Installation check |

### Learning Paths

**Beginner:** README.md → example_usage.py → QUICK_REFERENCE.md  
**Intermediate:** FINAL_COVERAGE_REPORT.md → examples_advanced.py → PACKAGE_GUIDE.md  
**Advanced:** COMPLETE_SUMMARY.md → examples_latest.py → Source code

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black rag_techniques/

# Type checking
mypy rag_techniques/

# Linting
flake8 rag_techniques/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rag_techniques,
  title = {RAG Techniques: A Comprehensive Toolkit for Retrieval-Augmented Generation},
  author = {RAG Techniques Contributors},
  year = {2024},
  url = {https://github.com/NirDiamant/RAG_Techniques}
}
```

## Acknowledgments

This package is based on the RAG Techniques repository by Nir Diamant and contributors.

## Support

- Documentation: [https://github.com/NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
- Issues: [GitHub Issues](https://github.com/NirDiamant/RAG_Techniques/issues)
- Discord: [Community Discord](https://discord.gg/cA6Aa4uyDX)
