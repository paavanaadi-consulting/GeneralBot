# Next Steps

Ideas for improving and scaling your RAG pipeline.

---

## Immediate Improvements

### 1. Add Error Handling

**Current:** Crashes on errors  
**Better:** Graceful degradation

```python
def load_pdf_safe(file_path):
    try:
        return load_pdf(file_path)
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return ""

def get_embedding_with_retry(text, retries=3):
    for i in range(retries):
        try:
            return get_embedding(text)
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            raise
```

### 2. Improve Chunking Strategy

**Current:** Fixed 500-word chunks  
**Better:** Semantic chunking

```python
def chunk_by_paragraphs(text, target_size=500):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        if current_size + para_words > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [para]
            current_size = para_words
        else:
            current_chunk.append(para)
            current_size += para_words
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

**Benefits:** Maintains logical boundaries, better context

### 3. Add Source Citations

**Current:** Mentions source  
**Better:** Clickable citations

```python
def generate_answer_with_citations(query, context_chunks):
    # Build context with citation markers
    context_parts = []
    for i, (doc, meta) in enumerate(zip(
        context_chunks['documents'][0],
        context_chunks['metadatas'][0]
    ), 1):
        context_parts.append(f"[{i}] {doc}\nSource: {meta['source']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Context:
{context}

Question: {query}

Answer the question and cite sources using [1], [2], etc."""
    
    # ... generate answer ...
    
    return answer, sources
```

### 4. Add Progress Indicators

**Current:** Silent processing  
**Better:** Progress bars

```bash
pip install tqdm
```

```python
from tqdm import tqdm

def store_in_chromadb(chunks, metadata, ids):
    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding chunks"):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    
    collection.add(...)
```

---

## Intermediate Enhancements

### 1. Multi-Query Retrieval

**Concept:** Generate multiple versions of user query

```python
def expand_query(query):
    """Generate query variations"""
    prompt = f"""Generate 3 different ways to ask this question:
    "{query}"
    
    Return only the questions, one per line."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    variations = response.choices[0].message.content.split('\n')
    return [query] + variations

def ask_question_multiquery(query):
    # Get query variations
    queries = expand_query(query)
    
    # Retrieve for each variation
    all_chunks = []
    for q in queries:
        results = search_documents(q, n_results=2)
        all_chunks.extend(results['documents'][0])
    
    # Deduplicate and use
    unique_chunks = list(set(all_chunks))
    # ... generate answer ...
```

**Benefits:** Better coverage, handles ambiguity

### 2. Re-ranking Retrieved Chunks

**Concept:** Two-stage retrieval

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

def ask_question_with_reranking(query):
    # Stage 1: Fast retrieval (get 20)
    results = search_documents(query, n_results=20)
    chunks = results['documents'][0]
    
    # Stage 2: Re-rank with cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Sort by score, take top 3
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in ranked[:3]]
    
    # Generate answer with best chunks
    # ...
```

**Benefits:** Better precision, higher quality results

### 3. Conversation Memory

**Concept:** Multi-turn dialogue

```python
class ConversationRAG:
    def __init__(self):
        self.history = []
    
    def ask(self, query):
        # Retrieve chunks
        results = search_documents(query)
        context = results['documents'][0]
        
        # Build prompt with history
        messages = [
            {"role": "system", "content": "You are a helpful assistant..."}
        ]
        
        # Add conversation history
        for user_msg, assistant_msg in self.history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current query with context
        current_msg = f"Context: {context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": current_msg})
        
        # Generate answer
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        
        # Save to history
        self.history.append((query, answer))
        
        return answer

# Usage
conv = ConversationRAG()
conv.ask("What is the remote work policy?")
conv.ask("What about expenses?")  # Remembers context
```

**Benefits:** Natural dialogue, context retention

### 4. Add Metadata Filtering

**Concept:** Filter by document attributes

```python
def ingest_with_metadata(documents_folder):
    for pdf_file in Path(documents_folder).glob("*.pdf"):
        text = load_pdf(pdf_file)
        chunks = chunk_text(text)
        
        # Rich metadata
        metadata = [{
            "source": pdf_file.name,
            "chunk_index": i,
            "document_type": detect_type(pdf_file),  # policy, manual, etc.
            "date": extract_date(text),
            "category": categorize(text)
        } for i, chunk in enumerate(chunks)]
        
        # Store with metadata
        store_in_chromadb(chunks, metadata, ids)

def search_with_filters(query, doc_type=None, after_date=None):
    # Build filter
    where_clause = {}
    if doc_type:
        where_clause["document_type"] = doc_type
    if after_date:
        where_clause["date"] = {"$gte": after_date}
    
    # Search with filters
    results = collection.query(
        query_embeddings=[get_embedding(query)],
        where=where_clause,
        n_results=3
    )
    return results

# Usage
results = search_with_filters(
    "vacation policy",
    doc_type="policy",
    after_date="2024-01-01"
)
```

**Benefits:** More precise retrieval, domain filtering

---

## Advanced Features

### 1. Hybrid Search

**Concept:** Combine semantic + keyword search

```bash
pip install rank-bm25
```

```python
from rank_bm25 import BM25Okapi

class HybridSearch:
    def __init__(self, collection):
        self.collection = collection
        # Get all documents for BM25
        all_docs = collection.get()
        self.documents = all_docs['documents']
        tokenized = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query, n_results=5, alpha=0.5):
        # Semantic search
        semantic_results = collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=n_results * 2
        )
        
        # Keyword search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores
        combined_scores = {}
        for doc, score in zip(semantic_results['documents'][0],
                             semantic_results['distances'][0]):
            combined_scores[doc] = alpha * (1 - score)  # Semantic
        
        for i, doc in enumerate(self.documents):
            if doc in combined_scores:
                combined_scores[doc] += (1 - alpha) * bm25_scores[i]
            else:
                combined_scores[doc] = (1 - alpha) * bm25_scores[i]
        
        # Sort and return top N
        ranked = sorted(combined_scores.items(),
                       key=lambda x: x[1],
                       reverse=True)[:n_results]
        
        return [doc for doc, score in ranked]
```

**Benefits:** Best of both worlds, handles edge cases

### 2. Agentic RAG

**Concept:** LLM decides when/what to retrieve

```python
def agentic_rag(query):
    messages = [
        {"role": "system", "content": """You are a helpful assistant.
        You can search documents using the search_documents function.
        Only search when you need information you don't have."""},
        {"role": "user", "content": query}
    ]
    
    # LLM with function calling
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=[{
            "name": "search_documents",
            "description": "Search the document database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }],
        function_call="auto"
    )
    
    # Check if LLM wants to search
    if response.choices[0].message.function_call:
        search_query = json.loads(
            response.choices[0].message.function_call.arguments
        )['query']
        
        # Execute search
        results = search_documents(search_query)
        
        # Send results back to LLM
        messages.append(response.choices[0].message)
        messages.append({
            "role": "function",
            "name": "search_documents",
            "content": json.dumps(results)
        })
        
        # LLM generates final answer
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    else:
        return response.choices[0].message.content
```

**Benefits:** Adaptive, multi-step reasoning

### 3. Build Web Interface

**Concept:** Streamlit dashboard

```bash
pip install streamlit
```

Create `app.py`:
```python
import streamlit as st
from rag_pipeline import ask_question, setup_rag

st.title("RAG Document Q&A")

# Sidebar for setup
with st.sidebar:
    st.header("Setup")
    if st.button("Process Documents"):
        with st.spinner("Processing..."):
            setup_rag("./documents")
        st.success("Documents processed!")

# Main chat interface
st.header("Ask Questions")

query = st.text_input("Your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Searching..."):
            answer = ask_question(query)
        st.write("**Answer:**")
        st.write(answer)
    else:
        st.warning("Please enter a question")

# History
if 'history' not in st.session_state:
    st.session_state.history = []

if query:
    st.session_state.history.append(query)

if st.session_state.history:
    st.header("Recent Questions")
    for q in st.session_state.history[-5:]:
        st.text(f"• {q}")
```

Run it:
```bash
streamlit run app.py
```

**Benefits:** User-friendly, shareable, interactive

---

## Production Deployment

### 1. Containerization

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rag_pipeline.py .
COPY documents/ ./documents/

CMD ["python", "app.py"]
```

```bash
# Build
docker build -t rag-mvp .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-mvp
```

### 2. API Backend

```bash
pip install fastapi uvicorn
```

Create `api.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import ask_question

app = FastAPI()

class Query(BaseModel):
    question: str
    n_results: int = 3

@app.post("/ask")
async def ask(query: Query):
    try:
        answer = ask_question(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

Run it:
```bash
uvicorn api:app --reload
```

### 3. Replace ChromaDB

**For production scale:**

```bash
# Qdrant (self-hosted)
docker run -p 6333:6333 qdrant/qdrant

pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Insert
client.upsert(
    collection_name="documents",
    points=[{
        "id": i,
        "vector": embedding,
        "payload": {"text": chunk, "source": source}
    } for i, (embedding, chunk, source) in enumerate(data)]
)

# Search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=3
)
```

**Benefits:** Better performance, production-ready

### 4. Add Monitoring

```bash
pip install prometheus-client
```

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
query_count = Counter('rag_queries_total', 'Total queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')

@query_duration.time()
def ask_question_monitored(query):
    query_count.inc()
    return ask_question(query)

# Start metrics server
start_http_server(8001)
```

**Track:**
- Query volume
- Response times
- Error rates
- Cost per query

---

## Evaluation Framework

### 1. Automated Testing

```python
# Create test cases
test_cases = [
    {
        "query": "What is the remote work policy?",
        "expected_keywords": ["remote", "work", "days", "week"],
        "expected_source": "policy.pdf"
    },
    # ... more test cases
]

def evaluate_retrieval():
    results = []
    for test in test_cases:
        retrieved = search_documents(test['query'])
        
        # Check if expected keywords present
        text = ' '.join(retrieved['documents'][0])
        keyword_match = all(k in text.lower() 
                           for k in test['expected_keywords'])
        
        # Check if expected source retrieved
        sources = [m['source'] for m in retrieved['metadatas'][0]]
        source_match = test['expected_source'] in sources
        
        results.append({
            'query': test['query'],
            'keyword_match': keyword_match,
            'source_match': source_match
        })
    
    return results
```

### 2. Human Evaluation

```python
def collect_feedback():
    """Let users rate answers"""
    answer = ask_question(query)
    
    rating = input("Rate this answer (1-5): ")
    feedback = input("Comments: ")
    
    # Log to file
    with open('feedback.jsonl', 'a') as f:
        f.write(json.dumps({
            'query': query,
            'answer': answer,
            'rating': rating,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }) + '\n')
```

---

## Learning Resources

### Books
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Natural Language Processing with Transformers" by Tunstall et al.

### Courses
- DeepLearning.AI: "Building Applications with Vector Databases"
- "LangChain for LLM Application Development"

### Documentation
- OpenAI API: https://platform.openai.com/docs
- ChromaDB: https://docs.trychroma.com
- LangChain: https://python.langchain.com

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Original RAG paper)
- "Dense Passage Retrieval for Open-Domain Question Answering"

---

## Recommended Path Forward

### Week 1-2: Solidify Basics
- [ ] Run MVP successfully
- [ ] Test with your own documents
- [ ] Experiment with parameters
- [ ] Understand all components

### Week 3-4: Add Improvements
- [ ] Implement error handling
- [ ] Add progress indicators
- [ ] Improve chunking strategy
- [ ] Add source citations

### Month 2: Intermediate Features
- [ ] Multi-query retrieval
- [ ] Re-ranking
- [ ] Conversation memory
- [ ] Metadata filtering

### Month 3: Advanced Topics
- [ ] Hybrid search
- [ ] Build web interface
- [ ] Add monitoring
- [ ] Evaluation framework

### Month 4+: Production
- [ ] Containerization
- [ ] API backend
- [ ] Production vector DB
- [ ] Deployment to cloud

---

## Key Principles

**1. Iterate incrementally**
- Make one change at a time
- Test thoroughly
- Measure impact

**2. Focus on user value**
- Better answers > fancy features
- Speed matters
- Cost matters

**3. Monitor and measure**
- Track what works
- Identify failure modes
- Optimize based on data

**4. Learn by building**
- Theory only gets you so far
- Break things and fix them
- Build real projects

---

## Your Advantage

As a data engineer, you're well-positioned to:
- Build robust data pipelines for document ingestion
- Implement effective monitoring and logging
- Optimize for cost and performance
- Scale to production workloads

**Focus on your strengths:**
- Data quality and preprocessing
- Pipeline orchestration
- Performance optimization
- Production deployment

---

**Ready to improve?** Pick one item from "Immediate Improvements" and implement it today.

**Questions?** → [FAQ.md](FAQ.md)

**Need help?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
