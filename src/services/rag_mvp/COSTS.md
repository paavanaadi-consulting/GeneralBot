# Cost Guide

Detailed breakdown of costs and optimization strategies.

---

## Cost Components

### OpenAI API Costs

**Embeddings (text-embedding-3-small):**
- **Price:** $0.02 per 1M tokens
- **Usage:** Every document chunk + every query

**Chat Completions (gpt-3.5-turbo):**
- **Input:** $0.50 per 1M tokens
- **Output:** $1.50 per 1M tokens  
- **Usage:** Every query response

### Infrastructure (This MVP)

**Local:**
- ChromaDB: FREE (runs on your machine)
- Storage: FREE (local disk)
- Compute: FREE (your MacBook)

**Total infrastructure cost: $0**

---

## One-Time Setup Costs

### Embedding Documents

**Scenario:** 10 PDF files, ~1000 chunks

```
Text to embed:     ~300K tokens
Model:             text-embedding-3-small
Cost per 1M:       $0.02
────────────────────────────────
Total:             ~$0.60
```

### Initial Testing

**Scenario:** 20 test queries

```
Query embeddings:  ~20 * 10 tokens = 200 tokens
LLM input:         ~20 * 2000 tokens = 40K tokens
LLM output:        ~20 * 200 tokens = 4K tokens
────────────────────────────────
Embedding cost:    $0.00004
LLM input cost:    $0.02
LLM output cost:   $0.006
────────────────────────────────
Total:             ~$0.50
```

### Total Setup Cost

```
Embeddings:        $0.60
Initial testing:   $0.50
────────────────────────────────
TOTAL:             ~$1-2
```

---

## Per-Query Costs

### Single Query Breakdown

**Query:** "What is the main topic?"

```
1. Embed question:        10 tokens × $0.02/1M = $0.0002
2. Search (ChromaDB):     FREE
3. LLM input (context):   2000 tokens × $0.50/1M = $0.001
4. LLM output (answer):   200 tokens × $1.50/1M = $0.0003
────────────────────────────────────────────────────────
Total per query:          ~$0.002 ($0.02 worst case)
```

### Query Volume Estimates

| Queries | Cost |
|---------|------|
| 10 | $0.02-0.20 |
| 50 | $0.10-1.00 |
| 100 | $0.20-2.00 |
| 500 | $1.00-10.00 |

**Note:** Actual costs depend on retrieved chunks and answer length.

---

## Daily Usage Estimates

### Light Use (Personal)

**Activity:**
- Add 5 new documents
- Run 10 queries

```
Embeddings (new docs):     ~$0.30
Queries:                   ~$0.20
────────────────────────────────
Daily total:               ~$0.50
```

### Moderate Use (Small Team)

**Activity:**
- Add 20 documents
- Run 50 queries

```
Embeddings (new docs):     ~$1.20
Queries:                   ~$1.00
────────────────────────────────
Daily total:               ~$2-3
```

### Heavy Use (Active Development)

**Activity:**
- Add 50 documents
- Run 200 queries

```
Embeddings (new docs):     ~$3.00
Queries:                   ~$4.00
────────────────────────────────
Daily total:               ~$7-10
```

---

## Monthly Estimates

### Personal Use

```
Documents:         50 PDFs (one-time)
Queries per day:   10
Days active:       20
────────────────────────────────
Setup:             $3
Ongoing:           $4-10
────────────────────────────────
Monthly total:     $7-13
```

### Small Team (5 users)

```
Documents:         200 PDFs (one-time)
Queries per day:   50
Days active:       22
────────────────────────────────
Setup:             $12
Ongoing:           $22-44
────────────────────────────────
Monthly total:     $34-56
```

### Medium Team (20 users)

```
Documents:         500 PDFs (one-time)
Queries per day:   200
Days active:       22
────────────────────────────────
Setup:             $30
Ongoing:           $88-176
────────────────────────────────
Monthly total:     $118-206
```

---

## Cost Comparison by Model

### Embedding Models

| Model | Cost/1M | Use Case |
|-------|---------|----------|
| text-embedding-3-small | $0.02 | Default (recommended) |
| text-embedding-3-large | $0.13 | Higher quality needed |
| text-embedding-ada-002 | $0.10 | Legacy |
| Sentence Transformers | $0 | Free (local) |

**For this MVP:** text-embedding-3-small (best value)

### LLM Models

| Model | Input/1M | Output/1M | Speed | Quality |
|-------|----------|-----------|-------|---------|
| gpt-3.5-turbo | $0.50 | $1.50 | Fast | Good |
| gpt-3.5-turbo-16k | $3.00 | $4.00 | Fast | Good |
| gpt-4 | $30 | $60 | Slow | Excellent |
| gpt-4-turbo | $10 | $30 | Medium | Excellent |
| Ollama (local) | $0 | $0 | Medium | Fair |

**For this MVP:** gpt-3.5-turbo (best value for Q&A)

---

## Cost Optimization Strategies

### 1. Cache Frequent Queries

**Problem:** Repeated questions cost money

**Solution:**
```python
query_cache = {}

def ask_question_cached(query):
    if query in query_cache:
        return query_cache[query]  # $0 cost
    
    answer = ask_question(query)
    query_cache[query] = answer
    return answer
```

**Savings:** ~$0.02 per cached query

### 2. Use Cheaper Models for Simple Queries

**Problem:** Using GPT-4 for everything is expensive

**Solution:**
```python
def ask_question_smart(query):
    # Simple factual query
    if is_simple_query(query):
        model = "gpt-3.5-turbo"  # $0.50/$1.50
    # Complex reasoning
    else:
        model = "gpt-4"  # $30/$60
    
    return generate_answer(query, model)
```

**Savings:** ~60-80% on simple queries

### 3. Retrieve Fewer Chunks

**Problem:** More chunks = more tokens = higher cost

**Solution:**
```python
# Default: 3 chunks
results = search_documents(query, n_results=3)

# For simple queries: 1-2 chunks
results = search_documents(query, n_results=1)
```

**Savings:** ~50-66% on token costs

### 4. Reduce Chunk Overlap

**Problem:** Overlap duplicates content

**Solution:**
```python
# Default: 50-word overlap
chunks = chunk_text(text, chunk_size=500, overlap=50)

# Reduced: 25-word overlap
chunks = chunk_text(text, chunk_size=500, overlap=25)
```

**Savings:** ~5-10% on embedding costs

### 5. Batch Embed Documents

**Problem:** Individual API calls have overhead

**Solution:**
```python
# Instead of 1000 individual calls
embeddings = []
for chunk in chunks:
    emb = get_embedding(chunk)
    embeddings.append(emb)

# Batch request (up to 2048 inputs)
response = client.embeddings.create(
    input=chunks[:2048],
    model="text-embedding-3-small"
)
embeddings = [d.embedding for d in response.data]
```

**Savings:** Faster, same cost but less API overhead

### 6. Don't Re-embed Unchanged Documents

**Problem:** Deleting ChromaDB and re-running costs money

**Solution:**
```python
# Only embed new documents
existing_docs = set(get_existing_doc_names())
new_docs = [d for d in all_docs if d not in existing_docs]

# Only embed new ones
for doc in new_docs:
    # process...
```

**Savings:** Avoid unnecessary re-embedding

### 7. Use Local Models for Development

**Problem:** Testing costs money

**Solution:**
```bash
# Install Ollama
brew install ollama

# Run local LLaMA
ollama pull llama2
```

```python
import ollama

# Free embeddings
embedding = ollama.embeddings(model='llama2', prompt=text)

# Free completions
response = ollama.chat(model='llama2', messages=[...])
```

**Savings:** $0 for development/testing

### 8. Set Token Limits

**Problem:** Very long answers cost more

**Solution:**
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    max_tokens=200  # Limit answer length
)
```

**Savings:** Prevents runaway costs

---

## Cost Monitoring

### Track Usage in Code

```python
import time

class CostTracker:
    def __init__(self):
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
    
    def log_embedding(self, text):
        tokens = len(text.split()) * 1.3  # Rough estimate
        self.embedding_tokens += tokens
    
    def log_llm(self, input_tokens, output_tokens):
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
    
    def get_costs(self):
        emb_cost = self.embedding_tokens / 1_000_000 * 0.02
        llm_in_cost = self.llm_input_tokens / 1_000_000 * 0.50
        llm_out_cost = self.llm_output_tokens / 1_000_000 * 1.50
        return {
            'embedding': emb_cost,
            'llm_input': llm_in_cost,
            'llm_output': llm_out_cost,
            'total': emb_cost + llm_in_cost + llm_out_cost
        }

tracker = CostTracker()
```

### OpenAI Dashboard

**Check usage:**
1. Go to https://platform.openai.com/usage
2. View breakdown by:
   - Date
   - Model
   - Token count

**Set limits:**
1. Go to https://platform.openai.com/account/billing/limits
2. Set monthly budget
3. Get alerts at thresholds

---

## Free Alternatives

### Zero-Cost Setup

**For learning/testing:**

```bash
# 1. Install local models
brew install ollama
ollama pull llama2

# 2. Install sentence transformers
pip install sentence-transformers
```

```python
# 3. Use in code
from sentence_transformers import SentenceTransformer
import ollama

# Free embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)

# Free LLM
response = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': prompt}]
)
```

**Trade-offs:**
- ✅ $0 cost
- ✅ No API limits
- ✅ Works offline
- ❌ Slower
- ❌ Lower quality
- ❌ Requires more RAM

**When to use:**
- Learning RAG concepts
- Development/testing
- Very limited budget
- Offline requirements

---

## Cost vs Quality Trade-offs

### Retrieval Quality

```
More chunks = Better context + Higher cost
Fewer chunks = Lower cost + Risk missing info
```

**Recommendation:** Start with 3, increase if answers poor

### LLM Quality

```
GPT-4 = Best answers + 20x cost
GPT-3.5 = Good answers + Low cost
```

**Recommendation:** Use GPT-3.5 for Q&A, GPT-4 for complex reasoning

### Embedding Quality

```
Large model = Better retrieval + 6.5x cost
Small model = Good retrieval + Low cost
```

**Recommendation:** text-embedding-3-small is sufficient

---

## Budget Planning

### Monthly Budget Recommendations

**Learning/Personal:**
- Budget: $10-20/month
- Covers: 500-1000 queries
- Use case: Learning, personal docs

**Small Team:**
- Budget: $50-100/month
- Covers: 2000-5000 queries
- Use case: Team knowledge base

**Production:**
- Budget: $200-500/month
- Covers: 10K-25K queries
- Use case: Customer support, large team

---

## Cost Reduction Checklist

- [ ] Cache common queries
- [ ] Use GPT-3.5 instead of GPT-4
- [ ] Retrieve minimum necessary chunks
- [ ] Set max_tokens limits
- [ ] Don't re-embed unchanged documents
- [ ] Use local models for testing
- [ ] Monitor usage dashboard
- [ ] Set budget alerts

---

## Real-World Examples

### Example 1: Personal Knowledge Base

**Setup:**
- 50 PDFs (research papers, books)
- 10 queries per day
- 20 active days per month

**Costs:**
```
Setup (one-time):      $3
Daily queries:         $0.20
Monthly:               $7
```

**Optimization:**
- Cache common queries → Save $1-2/month
- Total: **$5-7/month**

### Example 2: Company Documentation

**Setup:**
- 200 PDFs (policies, procedures)
- 50 queries per day (5 users)
- 22 working days

**Costs:**
```
Setup (one-time):      $12
Daily queries:         $1
Monthly:               $34
```

**Optimization:**
- Cache FAQs → Save $10/month
- Use GPT-3.5 only → Already doing
- Total: **$24/month**

### Example 3: Customer Support

**Setup:**
- 500 PDFs (product docs, FAQs)
- 200 queries per day
- 30 days

**Costs:**
```
Setup (one-time):      $30
Daily queries:         $4
Monthly:               $150
```

**Optimization:**
- Aggressive caching → Save $50/month
- Smart model selection → Save $30/month
- Fewer chunks for simple queries → Save $20/month
- Total: **$50/month** (67% reduction!)

---

## Next Steps

- **Understand architecture:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
- **Learn to optimize:** [NEXT_STEPS.md](NEXT_STEPS.md)
- **Start building:** [USAGE.md](USAGE.md)

---

**Questions about costs?** → [FAQ.md](FAQ.md)

**Ready to optimize?** → [NEXT_STEPS.md](NEXT_STEPS.md)
