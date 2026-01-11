# How RAG Works

Complete explanation of RAG architecture and concepts.

---

## The Problem RAG Solves

### LLM Limitations

**Knowledge Cutoff**
- LLMs only know information up to their training date
- Can't answer questions about recent events
- Don't know about your specific documents

**No Private Data Access**
- Can't read your company documents
- Don't know your database contents
- Can't access your files

### The RAG Solution

**Retrieval-Augmented Generation** = Retrieve relevant information + Augment LLM prompt + Generate answer

Instead of asking the LLM to answer from memory, we:
1. Find relevant information from YOUR documents
2. Give that information to the LLM
3. LLM answers based on provided context

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                           │
└─────────────────────────────────────────────────────────────┘

OFFLINE PROCESSING (Done Once)
═══════════════════════════════
PDF Documents
    ↓
Text Extraction
    ↓
Chunking (500 words)
    ↓
Generate Embeddings (OpenAI API)
    ↓
Store in ChromaDB
    

ONLINE PROCESSING (Per Query)
══════════════════════════════
User Question
    ↓
Generate Query Embedding
    ↓
Search Vector DB (Semantic Similarity)
    ↓
Retrieve Top 3 Chunks
    ↓
Build Prompt (Context + Question)
    ↓
LLM Generates Answer
    ↓
Return Answer with Citations
```

---

## Step-by-Step Example

### Scenario

You have company policy documents. User asks: **"Can I get reimbursed for my home WiFi?"**

### Step 1: Document Preparation (Offline)

**Original document chunk:**
```
Home internet costs can be reimbursed up to $50/month 
with proper receipts. Office supplies purchased for 
home office are eligible for reimbursement up to 
$200/quarter.
```

**Convert to embedding (vector):**
```
Text → OpenAI API → [0.67, 0.21, -0.33, 0.12, ..., 0.45]
                     ↑
                     1536 numbers (dimensions)
```

**Store in ChromaDB:**
```
{
  id: "doc2_chunk15",
  text: "Home internet costs can be...",
  embedding: [0.67, 0.21, -0.33, ...],
  metadata: {source: "policy.pdf", chunk: 15}
}
```

### Step 2: Query Processing (Real-time)

**User question:**
```
"Can I get reimbursed for my home WiFi?"
```

**Convert to embedding:**
```
Question → OpenAI API → [0.64, 0.19, -0.31, 0.09, ..., 0.43]
```

### Step 3: Semantic Search

**Calculate similarity with all stored chunks:**

```
Chunk 1 (scheduling policy):   Similarity: 0.45 ← Low
Chunk 2 (equipment policy):     Similarity: 0.38 ← Low  
Chunk 3 (reimbursement):        Similarity: 0.91 ← HIGH!
```

**Key insight:** "WiFi" matched "internet" semantically, not just by exact word match!

**Retrieve top 3:**
```
1. Chunk 3 (reimbursement) - 0.91
2. Chunk 8 (home office) - 0.67
3. Chunk 5 (expenses) - 0.62
```

### Step 4: Build Prompt

**Combine context + question:**
```
System: You are a helpful assistant...

User:
Context from documents:
[Chunk 3]: "Home internet costs can be reimbursed up to 
$50/month with proper receipts..."

[Chunk 8]: "Home office setup is eligible for..."

[Chunk 5]: "All expense claims require..."

Question: Can I get reimbursed for my home WiFi?

Please answer based on the context provided.
```

### Step 5: LLM Response

**GPT-3.5-turbo generates:**
```
Yes, you can get reimbursed for your home WiFi. 
According to company policy, home internet costs 
are reimbursable up to $50 per month. You'll need 
to submit proper receipts to claim this reimbursement.
```

---

## Key Concepts Explained

### Embeddings

**What are they?**
- Numerical representations of text
- Arrays of 1536 numbers (for OpenAI's model)
- Similar meanings = similar vectors

**Example:**
```
"dog" → [0.23, -0.15, 0.67, ...]
"puppy" → [0.24, -0.14, 0.68, ...]  ← Very similar!
"car" → [-0.45, 0.82, -0.12, ...]   ← Very different!
```

**Why useful?**
- Computers can calculate similarity between numbers
- Enables semantic search (meaning-based, not keyword-based)

### Chunking

**Why chunk documents?**
- Can't send entire 100-page document to LLM (token limits)
- Better precision when retrieving
- Faster search

**How it works:**
```
Original Document (5000 words)
    ↓
Split into chunks
    ↓
Chunk 1: words 1-500
Chunk 2: words 450-950  ← 50-word overlap
Chunk 3: words 900-1400 ← 50-word overlap
...
```

**Why overlap?**
- Maintains context at boundaries
- Important information might span chunk breaks

### Semantic Search

**Keyword search (traditional):**
```
Query: "WiFi reimbursement"
Finds: Documents containing exact words "WiFi" AND "reimbursement"
Misses: Documents about "internet expense" (different words, same meaning)
```

**Semantic search (RAG):**
```
Query: "WiFi reimbursement" → Embedding
Finds: Similar embeddings
Matches: "internet costs", "home network expenses", "WiFi bills"
```

### Context Window

**What is it?**
- Everything sent to the LLM in one request
- Has size limits (e.g., 4K, 16K, 128K tokens)

**What's included:**
```
[System Instructions]      ~500 tokens
[Retrieved Chunks]         ~2000 tokens
[Conversation History]     ~1000 tokens
[Current Question]         ~20 tokens
────────────────────────────────────
Total:                     ~3520 tokens
```

**Trade-off:**
- More retrieved chunks = better context
- But less room for conversation history
- Must balance based on use case

---

## Pipeline Components

### 1. Document Ingestion

**Code location:** `load_pdf()`, `chunk_text()`, `ingest_documents()`

**What it does:**
- Reads PDF files
- Extracts text from each page
- Splits into chunks
- Adds metadata (source file, chunk index)

**Parameters:**
- `chunk_size=500` - Words per chunk
- `overlap=50` - Overlapping words

### 2. Embedding Generation

**Code location:** `get_embedding()`

**What it does:**
- Sends text to OpenAI API
- Receives 1536-dimensional vector
- Returns embedding

**Model:** `text-embedding-3-small`
- Cost: $0.02 per 1M tokens
- Fast: ~100ms per request
- Quality: Good for most use cases

### 3. Vector Storage

**Code location:** `store_in_chromadb()`

**What it does:**
- Stores text + embeddings in ChromaDB
- Creates searchable index
- Persists to disk

**ChromaDB features:**
- Local storage (no external database)
- Automatic indexing
- Fast similarity search

### 4. Semantic Search

**Code location:** `search_documents()`

**What it does:**
1. Embed user's question
2. Calculate similarity with all stored embeddings
3. Return top N most similar chunks

**Similarity metric:** Cosine similarity
- Measures angle between vectors
- Range: -1 (opposite) to 1 (identical)
- Higher = more similar

### 5. Answer Generation

**Code location:** `generate_answer()`

**What it does:**
1. Combine retrieved chunks into context
2. Build prompt (system + context + question)
3. Call LLM
4. Return generated answer

**Model:** GPT-3.5-turbo
- Cost: $0.50/$1.50 per 1M input/output tokens
- Speed: 1-3 seconds
- Quality: Good for Q&A

---

## Why This Works

### Traditional Approach (Fails)

```
User: "Can I expense my WiFi?"
    ↓
LLM (no context): "I don't know your company's policy"
```

### RAG Approach (Works)

```
User: "Can I expense my WiFi?"
    ↓
Retrieve: [Policy document chunks]
    ↓
LLM (with context): "Yes, up to $50/month with receipts"
```

**Key difference:** LLM receives relevant information BEFORE answering

---

## Advantages of RAG

✅ **Always up-to-date**
- Update documents, not the model
- No retraining required

✅ **Works with private data**
- Your documents stay local
- No need to expose to training

✅ **Explainable**
- Can cite sources
- Show which documents informed answer

✅ **Cost-effective**
- Cheaper than fine-tuning
- Pay only for what you use

✅ **Flexible**
- Easy to add/remove documents
- Works with any LLM

---

## Limitations

❌ **Retrieval quality matters**
- Wrong chunks = wrong answer
- No magic fix for bad retrieval

❌ **Can't reason beyond context**
- LLM limited to provided chunks
- Can't synthesize from many sources easily

❌ **Latency**
- Embedding + search + LLM = 3-6 seconds
- Slower than cached responses

❌ **Token costs**
- Every query requires API calls
- Costs scale with usage

---

## When to Use RAG

✅ **Good use cases:**
- Q&A over documents (policies, manuals, research)
- Customer support (FAQs, knowledge base)
- Research assistance (papers, reports)
- Code documentation search

❌ **Not ideal for:**
- Simple classification tasks
- When all data fits in context window
- Latency-critical applications (<100ms)
- Tasks requiring complex multi-step reasoning

---

## RAG vs Alternatives

### RAG vs Fine-tuning

| RAG | Fine-tuning |
|-----|-------------|
| Update documents instantly | Requires retraining |
| Works with any LLM | Model-specific |
| Cheaper ($0.02/query) | Expensive upfront |
| Explainable (citations) | Black box |
| Limited to retrieved context | Full model knowledge |

### RAG vs Long Context Models

| RAG | Long Context (e.g., Claude 200K) |
|-----|----------------------------------|
| Fast search across millions of docs | Limited to ~200K tokens |
| Cheap (only relevant chunks) | Expensive (all tokens charged) |
| Scalable | Hits limits with large datasets |
| Retrieval overhead | Instant access |

### RAG vs Vector Search Only

| RAG | Vector Search |
|-----|---------------|
| Natural language answers | Returns document chunks |
| Synthesizes information | Just finds relevant docs |
| More expensive (LLM costs) | Cheaper (no LLM) |
| Better UX | Requires user to read |

---

## Real-World Example

### Company Knowledge Base

**Documents:**
- Employee handbook (200 pages)
- IT policies (50 pages)
- Benefits guide (100 pages)
- Remote work policy (30 pages)

**Traditional search:**
```
User searches: "work from home internet"
Results: 47 PDF pages mentioning these words
User must: Read through all to find answer
```

**RAG approach:**
```
User asks: "Can I expense my home internet?"
RAG:
  1. Searches 1,500 chunks in 0.5 seconds
  2. Finds 3 most relevant chunks
  3. LLM reads chunks and answers: "Yes, up to $50/month..."
User gets: Direct answer in 5 seconds
```

---

## Next Steps

- **Try it yourself:** [USAGE.md](USAGE.md)
- **Test thoroughly:** [TESTING.md](TESTING.md)
- **Understand costs:** [COSTS.md](COSTS.md)
- **Learn more:** [FAQ.md](FAQ.md)

---

**Questions?** → [FAQ.md](FAQ.md)

**Ready to run?** → [USAGE.md](USAGE.md)
