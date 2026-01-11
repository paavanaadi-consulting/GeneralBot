# RAG (Retrieval-Augmented Generation) MVP

A simple yet powerful RAG pipeline that demonstrates how to build an AI-powered question-answering system using your own documents.

---

## 🎯 What is This?

This project lets you:
- Upload your own PDF documents
- Ask questions about the content
- Get accurate answers with source citations
- Understand how LLMs can work with your private data

**Technologies:** Python, OpenAI API, ChromaDB, PyPDF

---

## 📚 Documentation

This project documentation is split into focused guides:

### Getting Started
- **[SETUP.md](SETUP.md)** - Installation and configuration guide
- **[USAGE.md](USAGE.md)** - How to run and use the pipeline
- **[TESTING.md](TESTING.md)** - Complete testing checklist

### Understanding RAG
- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** - Architecture and concepts explained
- **[FAQ.md](FAQ.md)** - Common questions and answers

### Reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[COSTS.md](COSTS.md)** - Pricing and cost optimization
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Ideas for improvement and scaling

---

## ⚡ Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install openai==1.3.7 chromadb==0.4.22 pypdf==3.17.4 python-dotenv==1.0.0

# 2. Configure API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 3. Add PDFs to documents/ folder
mkdir documents
# ... add your PDFs ...

# 4. Run the pipeline
python test_rag.py
```

For detailed instructions, see [SETUP.md](SETUP.md)

---

## 📁 Project Structure

```
rag-mvp/
├── README.md              # This file
├── SETUP.md               # Installation guide
├── USAGE.md               # Usage instructions
├── TESTING.md             # Testing guide
├── HOW_IT_WORKS.md        # Architecture explanation
├── FAQ.md                 # Questions & answers
├── TROUBLESHOOTING.md     # Problem solving
├── COSTS.md               # Cost breakdown
├── NEXT_STEPS.md          # Future improvements
│
├── rag_pipeline.py        # Main RAG implementation
├── test_rag.py            # Test script
├── test_env.py            # Environment verification
├── .env                   # API keys (create this)
├── .gitignore             # Git ignore rules
│
├── venv/                  # Virtual environment
├── documents/             # Your PDF files
└── chroma_db/             # Vector database (auto-created)
```

---

## 🎓 Learning Path

**New to RAG?** Read in this order:

1. [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - Understand the concepts
2. [SETUP.md](SETUP.md) - Get it running
3. [USAGE.md](USAGE.md) - Use the pipeline
4. [FAQ.md](FAQ.md) - Deepen your understanding
5. [NEXT_STEPS.md](NEXT_STEPS.md) - Explore improvements

---

## 💡 Key Features

- **Local-first:** Runs on your MacBook, no cloud infrastructure needed
- **Simple:** ~200 lines of well-commented Python code
- **Educational:** Clear architecture with detailed explanations
- **Production-ready concepts:** Demonstrates patterns used in real systems
- **Low cost:** ~$2-5 for initial setup and testing

---

## 🚀 What You'll Learn

- How embeddings convert text to vectors
- Semantic search vs keyword search
- Vector databases and similarity search
- LLM prompt engineering for RAG
- Document chunking strategies
- Production considerations for AI systems

---

## 📊 Performance (MacBook Air M3, 24GB RAM)

| Operation | Time | Cost |
|-----------|------|------|
| Process 10 PDFs (1000 chunks) | 2-3 min | $0.60 |
| Single query | 3-6 sec | $0.02 |
| Total setup cost | - | $1-2 |

---

## ⚠️ Prerequisites

- macOS (tested on M3 MacBook Air)
- Python 3.9 or 3.10
- OpenAI API account ($5-10 budget)
- 5-10 PDF documents to test with

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with the code
- Try different models and parameters
- Share improvements and insights

---

## 📄 License

Educational project - use freely for learning.

---

**Ready to start?** → [SETUP.md](SETUP.md)

**Questions?** → [FAQ.md](FAQ.md)

**Issues?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
