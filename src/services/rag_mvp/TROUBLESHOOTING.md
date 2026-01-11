# Troubleshooting Guide

Solutions to common issues and errors.

---

## Installation Issues

### ModuleNotFoundError: No module named 'openai'

**Cause:** Virtual environment not activated or packages not installed

**Solution:**
```bash
# Activate venv
source venv/bin/activate

# Reinstall packages
pip install openai==1.3.7 chromadb==0.4.22 pypdf==3.17.4 python-dotenv==1.0.0

# Verify
python -c "import openai; print('Success')"
```

---

### TypeError: __init__() got an unexpected keyword argument 'proxies'

**Cause:** Version conflict between openai and httpx libraries

**Solution:**
```bash
# Uninstall conflicting packages
pip uninstall openai httpx httpcore -y

# Install compatible version
pip install openai==1.3.7

# Verify
python -c "from openai import OpenAI; print('Success')"
```

---

### AttributeError: 'np.float_' was removed in NumPy 2.0

**Cause:** ChromaDB incompatible with NumPy 2.0

**Solution:**
```bash
# Downgrade NumPy
pip uninstall numpy -y
pip install "numpy<2.0"

# Verify
python -c "import chromadb; print('Success')"
```

---

### NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+

**Cause:** urllib3 v2 compatibility warning (harmless)

**Solution (optional):**
```bash
# Downgrade to avoid warning
pip install "urllib3<2.0"
```

**Note:** This is just a warning, not an error. Code will work fine.

---

## API Key Issues

### OpenAIError: The api_key client option must be set

**Cause:** API key not loaded from .env file

**Solutions:**

**1. Check .env file format:**
```bash
cat .env

# Should show exactly:
# OPENAI_API_KEY=sk-proj-...

# Common mistakes:
# ❌ OPEN_API_KEY=...     (missing "AI")
# ❌ OPENAI_API_KEY = ... (extra spaces)
# ❌ "sk-proj-..."        (quotes around key)
# ✅ OPENAI_API_KEY=sk-proj-...
```

**2. Verify loading:**
```bash
python test_env.py
# Should show: API Key loaded: sk-proj-TA...
```

**3. Hardcode temporarily (testing only):**
```python
# In rag_pipeline.py, line 14
client = OpenAI(api_key="sk-proj-your-actual-key")
```
**⚠️ Never commit this to git!**

---

### AuthenticationError: Incorrect API key

**Cause:** Invalid or expired API key

**Solution:**
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Update .env file
4. Restart script

---

### RateLimitError: Rate limit exceeded

**Cause:** Too many API requests

**Solution:**
```python
# Add delays between requests
import time

for chunk in chunks:
    embedding = get_embedding(chunk)
    time.sleep(0.1)  # 100ms delay
```

Or upgrade your OpenAI plan

---

## Document Processing Issues

### Found 0 PDF files

**Cause:** No PDFs in documents folder or wrong location

**Solution:**
```bash
# Check folder
ls -la documents/

# Verify .pdf extension (case-sensitive)
ls documents/*.pdf

# Add PDFs
cp /path/to/files/*.pdf documents/
```

---

### PDF text extraction returns empty string

**Cause:** PDF is scanned images, not text

**Solution:**

**Option 1:** Use text-based PDFs

**Option 2:** Add OCR support
```bash
pip install pytesseract pdf2image
```

```python
from pdf2image import convert_from_path
import pytesseract

def load_scanned_pdf(file_path):
    images = convert_from_path(file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text
```

---

### PdfReadError: EOF marker not found

**Cause:** Corrupted PDF file

**Solution:**
```python
# Add error handling
def load_pdf_safe(file_path):
    try:
        return load_pdf(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""
```

---

## ChromaDB Issues

### Collection already exists error

**Cause:** Trying to create collection that exists

**Solution:**
```python
# Use get_or_create instead of create
collection = chroma_client.get_or_create_collection(
    name="documents"
)
```

---

### ChromaDB database locked

**Cause:** Another process using database

**Solution:**
```bash
# Close all Python processes
pkill python

# Or delete and recreate
rm -rf chroma_db/
python test_rag.py
```

---

### ChromaDB corrupted

**Symptoms:** Random errors, missing data

**Solution:**
```bash
# Delete database
rm -rf chroma_db/

# Re-run setup
python test_rag.py
```

---

## Query Issues

### No results returned

**Cause:** Database empty or query too specific

**Debug:**
```python
# Check database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("documents")
print(f"Total chunks: {collection.count()}")

# If 0, re-run setup
```

**Solution:**
```bash
python test_rag.py
```

---

### Results not relevant

**Cause:** Poor retrieval quality

**Solutions:**

**1. Retrieve more chunks:**
```python
# In rag_pipeline.py
results = search_documents(query, n_results=5)  # vs 3
```

**2. Rephrase query:**
```python
# Vague: "What's the policy?"
# Better: "What is the remote work policy for employees?"
```

**3. Check chunk size:**
```python
# Try larger chunks
chunks = chunk_text(text, chunk_size=800, overlap=100)
```

---

### LLM says "not in context" but answer is in documents

**Cause:** Relevant chunks not retrieved

**Debug:**
```python
results = search_documents(query)
print("Retrieved chunks:")
for doc in results['documents'][0]:
    print(doc[:200])
# Check if relevant info is here
```

**Solutions:**
- Retrieve more chunks (n_results=10)
- Improve query phrasing
- Check if info actually in documents

---

## Performance Issues

### Embedding very slow

**Cause:** Network latency or API rate limits

**Solutions:**

**1. Batch requests:**
```python
# Embed multiple at once
response = client.embeddings.create(
    input=[chunk1, chunk2, chunk3],
    model="text-embedding-3-small"
)
```

**2. Use local embeddings:**
```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("text")
```

---

### Queries take too long

**Symptoms:** >10 seconds per query

**Causes & Solutions:**

**1. Too many chunks retrieved:**
```python
# Reduce from 10 to 3
results = search_documents(query, n_results=3)
```

**2. Slow LLM:**
```python
# Use faster model
model="gpt-3.5-turbo"  # vs "gpt-4"
```

**3. Large chunks:**
```python
# Reduce chunk size
chunk_size=500  # vs 1000
```

---

### Out of memory

**Cause:** Processing too much data at once

**Solutions:**

**1. Process in batches:**
```python
def process_documents_batch(files, batch_size=5):
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        # Process batch
```

**2. Reduce chunk overlap:**
```python
chunk_text(text, chunk_size=500, overlap=25)  # vs 50
```

**3. Clear memory:**
```python
import gc
gc.collect()
```

---

## Cost Issues

### Unexpected high costs

**Check usage:**
1. Go to https://platform.openai.com/usage
2. View breakdown by model
3. Check token counts

**Common causes:**

**1. Re-embedding same documents:**
```bash
# Don't delete chroma_db unless necessary
# Only re-run setup when adding new documents
```

**2. Too many retrieved chunks:**
```python
# Costs scale with tokens sent to LLM
n_results=3  # vs 10
```

**3. Using expensive model:**
```python
model="gpt-3.5-turbo"  # $0.50/$1.50 per 1M
# vs
model="gpt-4"  # $30/$60 per 1M
```

**Set usage limits:**
- https://platform.openai.com/account/billing/limits
- Set to $10 or $20 for safety

---

## Import Errors

### ImportError: cannot import name 'OpenAI'

**Cause:** Wrong openai version

**Solution:**
```bash
pip uninstall openai -y
pip install openai==1.3.7
```

---

### ImportError: No module named 'pypdf'

**Cause:** Package name changed

**Solution:**
```bash
# New name is pypdf (not PyPDF2)
pip install pypdf==3.17.4
```

---

## Environment Issues

### Virtual environment not activating

**Symptoms:** (venv) not showing in prompt

**Solution:**
```bash
# Deactivate if active
deactivate

# Delete and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

---

### Wrong Python version

**Check version:**
```bash
python --version
# Should be 3.9.x or 3.10.x
```

**Solution:**
```bash
# Install correct version via Homebrew
brew install python@3.10

# Create venv with specific version
python3.10 -m venv venv
source venv/bin/activate
```

---

## Runtime Errors

### KeyError: 'documents'

**Cause:** ChromaDB result format changed

**Solution:**
```python
# Check result structure
results = search_documents(query)
print(results.keys())

# Adjust code accordingly
```

---

### Connection timeout

**Cause:** Network issues or API down

**Solutions:**

**1. Check internet:**
```bash
ping api.openai.com
```

**2. Check API status:**
- https://status.openai.com

**3. Add retry logic:**
```python
import time

def get_embedding_with_retry(text, retries=3):
    for i in range(retries):
        try:
            return get_embedding(text)
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
                continue
            raise
```

---

## Debug Mode

### Enable verbose logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Print intermediate results

```python
# In rag_pipeline.py

def ask_question(query):
    print(f"DEBUG: Query = {query}")
    
    results = search_documents(query)
    print(f"DEBUG: Retrieved {len(results['documents'][0])} chunks")
    
    for i, doc in enumerate(results['documents'][0]):
        print(f"DEBUG: Chunk {i}: {doc[:100]}")
    
    answer = generate_answer(query, results)
    print(f"DEBUG: Answer length = {len(answer)}")
    
    return answer
```

---

## Getting Help

### Information to provide

When asking for help, include:

1. **Error message** (full traceback)
2. **What you tried** (commands run)
3. **Environment info:**
   ```bash
   python --version
   pip list | grep -E "openai|chromadb|pypdf"
   ```
4. **Relevant code** (what you changed)

### Check logs

```bash
# Run with verbose output
python -v test_rag.py 2>&1 | tee debug.log
```

---

## Clean Slate

If nothing works, start fresh:

```bash
# 1. Deactivate and delete everything
deactivate
cd ..
rm -rf rag-mvp

# 2. Start from scratch
mkdir rag-mvp && cd rag-mvp

# 3. Follow SETUP.md exactly
python3 -m venv venv
source venv/bin/activate
pip install openai==1.3.7 "numpy<2.0" chromadb==0.4.22 pypdf==3.17.4 python-dotenv==1.0.0

# 4. Configure .env
echo "OPENAI_API_KEY=your-key" > .env

# 5. Add documents and run
mkdir documents
# ... add PDFs ...
python test_rag.py
```

---

**Still having issues?** Check [FAQ.md](FAQ.md) for concept questions.

**Ready to continue?** See [USAGE.md](USAGE.md) or [NEXT_STEPS.md](NEXT_STEPS.md).
