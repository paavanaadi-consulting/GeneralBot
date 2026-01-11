# Usage Guide

How to run and use the RAG pipeline.

---

## First Time Setup

### Run the Complete Pipeline

```bash
# Make sure you're in rag-mvp directory
# Make sure venv is activated (you should see (venv) in prompt)

python test_rag.py
```

### What Happens

**1. Document Processing**
```
Setting up RAG pipeline...
Found 5 PDF files
Processing: document1.pdf
  - Created 47 chunks
Processing: document2.pdf
  - Created 52 chunks
...
```

**2. Embedding & Storage**
```
Embedding and storing 245 chunks...
  Embedding chunk 0/245...
  Embedding chunk 10/245...
  ...
✓ Stored 245 chunks in ChromaDB
```

**3. Query Testing**
```
Testing queries...
Searching for: 'What are the main topics?'

Retrieved chunks:
--- Chunk 1 (from document1.pdf) ---
...

ANSWER:
Based on the documents provided...
```

**Time:** 2-5 minutes depending on number of documents  
**Cost:** ~$0.50-2.00

---

## Interactive Usage

### Python REPL

```bash
python
```

```python
from rag_pipeline import ask_question

# Ask questions about your documents
ask_question("What are the key findings?")
ask_question("Summarize the methodology")
ask_question("What recommendations are made?")
```

### Custom Script

Create `my_queries.py`:

```python
from rag_pipeline import ask_question

questions = [
    "What are the main topics?",
    "What methodology is used?",
    "What are the conclusions?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print('='*60)
    ask_question(q)
```

Run it:
```bash
python my_queries.py
```

---

## Customizing Queries

### Edit test_rag.py

```python
# Modify the questions list based on YOUR documents
questions = [
    "What are the main topics covered in these documents?",
    "Can you summarize the key points?",
    
    # Add your own questions:
    "What recommendations are provided?",
    "What is the methodology described?",
    "Who are the authors?",
]
```

### Tips for Good Questions

**✅ Good questions:**
- "What is the remote work policy for full-time employees?"
- "What are the three main findings in the study?"
- "How does the methodology differ from previous approaches?"

**❌ Vague questions:**
- "What's the policy?" (which policy?)
- "Tell me about it" (about what?)
- "Summarize" (too broad)

---

## Understanding the Output

### Query Output Format

```
Searching for: 'What are the main topics?'

Retrieved chunks:
--- Chunk 1 (from document1.pdf) ---
[First 200 characters of relevant text...]

--- Chunk 2 (from document3.pdf) ---
[First 200 characters of relevant text...]

--- Chunk 3 (from document2.pdf) ---
[First 200 characters of relevant text...]

==================================================
ANSWER:
==================================================
Based on the documents provided, the main topics include:
1. [Topic 1]
2. [Topic 2]
...
```

### What Each Part Means

**"Searching for"** - Your question being processed

**"Retrieved chunks"** - Most relevant text pieces found
- Shows which document each came from
- Preview of the content
- These are sent to the LLM as context

**"ANSWER"** - LLM-generated response
- Based on retrieved chunks
- Should cite sources
- May say "not enough information" if context insufficient

---

## Adding New Documents

### Option 1: Re-run Everything

```bash
# Delete existing database
rm -rf chroma_db/

# Add new PDFs to documents/ folder
cp /path/to/new.pdf documents/

# Re-run setup
python test_rag.py
```

**Cost:** ~$0.50-2.00 to re-embed all documents

### Option 2: Incremental Update (Advanced)

Create `add_document.py`:

```python
from rag_pipeline import load_pdf, chunk_text, store_in_chromadb
from pathlib import Path

# Process single new document
new_pdf = "documents/new_document.pdf"
text = load_pdf(Path(new_pdf))
chunks = chunk_text(text)

# Generate IDs and metadata
doc_id = len(list(Path("documents").glob("*.pdf"))) - 1
ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]
metadata = [{"source": "new_document.pdf", "chunk_index": i} 
            for i in range(len(chunks))]

# Store
store_in_chromadb(chunks, metadata, ids)
print(f"✓ Added {len(chunks)} chunks from new document")
```

---

## Adjusting Parameters

### Change Chunk Size

In `rag_pipeline.py`, line ~32:

```python
# Smaller chunks (more precise, less context)
chunks = chunk_text(text, chunk_size=300, overlap=50)

# Default
chunks = chunk_text(text, chunk_size=500, overlap=50)

# Larger chunks (more context, less precise)
chunks = chunk_text(text, chunk_size=800, overlap=100)
```

**Effect:**
- Smaller = better for specific facts
- Larger = better for comprehensive topics

### Change Number of Retrieved Chunks

In `rag_pipeline.py`, line ~109:

```python
# Retrieve fewer chunks (faster, less context)
results = search_documents(query, n_results=2)

# Default
results = search_documents(query, n_results=3)

# Retrieve more chunks (more context, slower, costs more)
results = search_documents(query, n_results=5)
```

**Effect:**
- More chunks = better for complex questions
- Fewer chunks = faster, cheaper

### Change LLM Model

In `rag_pipeline.py`, line ~144:

```python
# Cheaper, faster (default)
model="gpt-3.5-turbo"

# Better quality, slower, expensive
model="gpt-4"

# Cheaper OpenAI model
model="gpt-3.5-turbo-16k"  # Larger context window
```

**Cost comparison:**
- GPT-3.5-turbo: $0.50/$1.50 per 1M tokens
- GPT-4: $30/$60 per 1M tokens

---

## Monitoring Usage

### Check OpenAI Costs

1. Go to https://platform.openai.com/usage
2. View today's usage
3. Check costs by model

### Expected Costs Per Query

| Component | Cost |
|-----------|------|
| Question embedding | $0.00002 |
| Answer generation | $0.01-0.02 |
| **Total per query** | **~$0.02** |

### Cost-Saving Tips

1. **Cache common queries** - Store results for repeated questions
2. **Use cheaper models** - GPT-3.5 for simple Q&A
3. **Retrieve fewer chunks** - Only get what you need
4. **Batch operations** - Process multiple documents at once

---

## Working with Different File Types

This MVP only supports PDFs. For other formats:

### Text Files (.txt, .md)

```python
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
```

### Word Documents (.docx)

```bash
pip install python-docx
```

```python
from docx import Document

def load_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
```

### Web Pages

```bash
pip install beautifulsoup4 requests
```

```python
import requests
from bs4 import BeautifulSoup

def load_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()
```

---

## Example Workflows

### Research Paper Analysis

```python
# Questions to ask about research papers
questions = [
    "What is the main research question?",
    "What methodology was used?",
    "What were the key findings?",
    "What are the limitations mentioned?",
    "What future work is suggested?",
]
```

### Policy Document Q&A

```python
# Questions about company policies
questions = [
    "What is the remote work policy?",
    "What expenses can be reimbursed?",
    "What are the vacation day allowances?",
    "What is the dress code?",
]
```

### Technical Documentation

```python
# Questions about technical docs
questions = [
    "How do I install the software?",
    "What are the system requirements?",
    "How do I configure authentication?",
    "What are common troubleshooting steps?",
]
```

---

## Best Practices

### Document Preparation

✅ **Do:**
- Use text-based PDFs (not scanned images)
- Include 5-10 documents minimum
- Use documents on related topics
- Keep documents under 100 pages each

❌ **Don't:**
- Use password-protected PDFs
- Mix completely unrelated topics
- Include corrupted files

### Question Formulation

✅ **Do:**
- Be specific and clear
- Ask one thing at a time
- Provide context if needed
- Use domain-appropriate terms

❌ **Don't:**
- Ask multiple unrelated questions
- Use ambiguous pronouns
- Expect information not in documents

### Quality Control

After getting an answer:
1. Check if sources are cited
2. Verify facts against original PDFs
3. Note if answer seems incomplete
4. Try rephrasing if results are poor

---

## Next Steps

- **Understand the architecture:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
- **Run comprehensive tests:** [TESTING.md](TESTING.md)
- **Learn about improvements:** [NEXT_STEPS.md](NEXT_STEPS.md)

---

**Having issues?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Questions about concepts?** → [FAQ.md](FAQ.md)
