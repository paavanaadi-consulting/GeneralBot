# Testing Guide

Complete testing sequence to verify your RAG pipeline works correctly.

---

## Testing Checklist

- [ ] Environment setup verified
- [ ] API key loaded successfully
- [ ] Documents folder populated
- [ ] PDF text extraction works
- [ ] Embedding generation successful
- [ ] Full pipeline runs without errors
- [ ] ChromaDB populated with chunks
- [ ] Queries return relevant results
- [ ] Interactive testing works
- [ ] Costs within expected range
- [ ] Answer quality acceptable

---

## Step 1: Environment Verification

### Verify Virtual Environment

```bash
# Activate venv
source venv/bin/activate

# Should see (venv) in prompt
```

### Verify Python Version

```bash
python --version
# Should show: Python 3.9.x or 3.10.x
```

### Verify Package Imports

```bash
python -c "import openai; import chromadb; import pypdf; print('✓ All imports successful')"
```

**Expected:** `✓ All imports successful`

**If fails:** Review [SETUP.md](SETUP.md) Step 3

---

## Step 2: API Key Verification

### Check .env File

```bash
# Verify file exists
ls -la .env

# Check content
cat .env
# Should show: OPENAI_API_KEY=sk-proj-...
```

### Test API Key Loading

```bash
python test_env.py
```

**Expected output:**
```
API Key loaded: sk-proj-TA...
Key length: 164
```

**If fails:** Review [SETUP.md](SETUP.md) Step 6

---

## Step 3: Document Preparation

### Check Documents Folder

```bash
# List documents
ls -la documents/

# Count PDFs
ls documents/*.pdf | wc -l
# Should show: 5 or more
```

### Verify PDF Files

```bash
# Check file types
file documents/*.pdf
# Each should show: PDF document, version X.X
```

**Expected:** 5-10 valid PDF files

**If no PDFs:** Add PDFs to documents/ folder (see [SETUP.md](SETUP.md))

---

## Step 4: Test Document Ingestion

Create `test_ingestion.py`:

```python
from rag_pipeline import load_pdf, chunk_text
from pathlib import Path

# Test loading one PDF
pdf_files = list(Path("documents").glob("*.pdf"))
if pdf_files:
    print(f"Testing with: {pdf_files[0].name}")
    
    # Extract text
    text = load_pdf(pdf_files[0])
    print(f"✓ Extracted {len(text)} characters")
    
    # Chunk it
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"✓ Created {len(chunks)} chunks")
    print(f"\nSample chunk:\n{chunks[0][:200]}...")
else:
    print("❌ No PDF files found")
```

Run it:
```bash
python test_ingestion.py
```

**Expected output:**
```
Testing with: document1.pdf
✓ Extracted 15234 characters
✓ Created 47 chunks

Sample chunk:
This is the beginning of the document text...
```

**If fails:** PDF may be corrupted or scanned images only

---

## Step 5: Test Embedding Generation

Create `test_embedding.py`:

```python
from rag_pipeline import get_embedding

# Test embedding
test_text = "This is a test sentence for embedding generation."

try:
    embedding = get_embedding(test_text)
    print(f"✓ Generated embedding")
    print(f"  Dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Type: {type(embedding)}")
except Exception as e:
    print(f"❌ Embedding failed: {e}")
```

Run it:
```bash
python test_embedding.py
```

**Expected output:**
```
✓ Generated embedding
  Dimensions: 1536
  First 5 values: [0.0234, -0.0156, 0.0432, -0.0089, 0.0267]
  Type: <class 'list'>
```

**Cost:** ~$0.001
**If fails:** Check API key and internet connection

---

## Step 6: Full Pipeline Test

```bash
python test_rag.py
```

### Expected Sequence

**1. Document Processing:**
```
Setting up RAG pipeline...
Found 5 PDF files
Processing: document1.pdf
  - Created 47 chunks
Processing: document2.pdf
  - Created 52 chunks
...
```

**2. Embedding & Storage:**
```
Embedding and storing 245 chunks...
  Embedding chunk 0/245...
  Embedding chunk 10/245...
  ...
✓ Stored 245 chunks in ChromaDB
```

**3. Query Testing:**
```
Testing queries...
Searching for: 'What are the main topics?'

Retrieved chunks:
--- Chunk 1 (from document1.pdf) ---
...

ANSWER:
Based on the documents provided, the main topics include...
```

**Time:** 2-5 minutes  
**Cost:** $0.50-2.00

---

## Step 7: Verify ChromaDB Storage

### Check Database Created

```bash
# Check database exists
ls -la chroma_db/

# Check size
du -sh chroma_db/
# Should show size (e.g., 15MB for 250 chunks)
```

### Test Database Access

Create `test_chromadb.py`:

```python
import chromadb

# Connect to database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="documents")

# Check stats
count = collection.count()
print(f"✓ ChromaDB contains {count} chunks")

# Sample retrieval
results = collection.peek(limit=3)
print(f"✓ Sample documents:")
for i, doc in enumerate(results['documents'][:3]):
    print(f"  {i+1}. {doc[:100]}...")
```

Run it:
```bash
python test_chromadb.py
```

**Expected output:**
```
✓ ChromaDB contains 245 chunks
✓ Sample documents:
  1. This document discusses the implementation...
  2. Data preprocessing is a critical step...
  3. The model architecture consists of...
```

---

## Step 8: Interactive Query Testing

Create `interactive_test.py`:

```python
from rag_pipeline import ask_question

print("Interactive RAG Testing")
print("=" * 60)
print("Enter questions (or 'quit' to exit)\n")

while True:
    query = input("\nYour question: ")
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if query.strip():
        try:
            ask_question(query)
        except Exception as e:
            print(f"Error: {e}")
```

Run it:
```bash
python interactive_test.py
```

**Test with various questions:**
```
Your question: What is this document about?
Your question: Summarize the key findings
Your question: What methodology is used?
```

---

## Step 9: Cost Tracking

### Check OpenAI Usage

1. Go to: https://platform.openai.com/usage
2. Check today's usage
3. Verify costs

**Expected after full test:**
- Embeddings: $0.50-1.00
- Chat completions: $0.50-1.00
- **Total: $1-2**

---

## Step 10: Quality Assessment

### Test Different Query Types

```python
test_questions = [
    # Factual
    "What year was this published?",
    "Who are the authors?",
    
    # Comprehension
    "What is the main argument?",
    "What methodology is used?",
    
    # Synthesis
    "How do these documents relate?",
    "What are common themes?",
    
    # Negative tests (should say not in context)
    "What is the capital of France?",
    "Tell me about quantum physics",
]
```

### Quality Metrics

| Metric | How to Test | Expected |
|--------|-------------|----------|
| **Relevance** | Specific questions | On-topic answers |
| **Accuracy** | Verify against PDFs | No hallucinations |
| **Citations** | Check sources | Documents cited |
| **Coverage** | Various topics | Multiple docs used |

---

## Automated Testing

### Create Test Suite

Create `test_suite.py`:

```python
from rag_pipeline import ask_question, search_documents
import chromadb

def test_retrieval():
    """Test that retrieval returns results"""
    results = search_documents("test query")
    assert len(results['documents'][0]) > 0
    print("✓ Retrieval test passed")

def test_database():
    """Test ChromaDB has content"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="documents")
    count = collection.count()
    assert count > 0
    print(f"✓ Database test passed ({count} chunks)")

def test_answer_generation():
    """Test end-to-end answer generation"""
    answer = ask_question("What is the main topic?")
    assert len(answer) > 10
    print("✓ Answer generation test passed")

if __name__ == "__main__":
    print("Running test suite...\n")
    test_database()
    test_retrieval()
    test_answer_generation()
    print("\n✓ All tests passed!")
```

Run it:
```bash
python test_suite.py
```

---

## Performance Benchmarks

**Hardware:** MacBook Air M3, 24GB RAM

| Operation | Expected Time |
|-----------|---------------|
| Load 1 PDF (50 pages) | 2-3 sec |
| Chunk 1 PDF | <1 sec |
| Embed 100 chunks | 30-60 sec |
| Store in ChromaDB | <5 sec |
| Single query | 3-6 sec |

**If significantly slower:** Check internet connection or API rate limits

---

## Common Test Failures

### No PDF files found

```bash
# Add PDFs to documents/
cp /path/to/*.pdf documents/
```

### Embedding fails

- Check API key in .env
- Verify internet connection
- Check OpenAI status: https://status.openai.com

### ChromaDB errors

```bash
# Delete and recreate
rm -rf chroma_db/
python test_rag.py
```

### Out of memory

- Process fewer documents
- Reduce batch size
- Restart Python

---

## Regression Testing

After making changes, re-run:

```bash
# Quick test
python test_embedding.py
python test_chromadb.py

# Full test
python test_rag.py
```

---

## Next Steps

After testing passes:

- **Understand what you built:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
- **Learn to use it:** [USAGE.md](USAGE.md)
- **Improve it:** [NEXT_STEPS.md](NEXT_STEPS.md)

---

**Tests failing?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Ready to use?** → [USAGE.md](USAGE.md)
