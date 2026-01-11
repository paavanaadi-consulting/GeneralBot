# GeneralBot Architecture

## System Overview

GeneralBot is an AI-powered conversational assistant that combines Large Language Models (LLM) with Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses based on organization-specific knowledge.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interfaces                          │
├─────────────────────┬───────────────────┬──────────────────────┤
│   REST API Client   │   CLI Interface   │   Web Application    │
│    (HTTP/JSON)      │    (Terminal)     │     (Future)         │
└──────────┬──────────┴──────────┬────────┴──────────────────────┘
           │                     │
           ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                         │
├─────────────────────────────────────────────────────────────────┤
│  Routes: /chat, /ingest, /upload, /session, /health             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Conversational Assistant                       │
├─────────────────────────────────────────────────────────────────┤
│  • Orchestrates RAG pipeline                                     │
│  • Manages conversation sessions                                 │
│  • Coordinates component interactions                            │
└───────┬────────────────┬─────────────────┬──────────────────────┘
        │                │                 │
        ▼                ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
│   Document   │ │  Vector Store│ │    LLM Manager       │
│  Processor   │ │   Manager    │ │                      │
├──────────────┤ ├──────────────┤ ├──────────────────────┤
│ • Load docs  │ │ • Embeddings │ │ • OpenAI/Azure       │
│ • Chunk text │ │ • ChromaDB   │ │ • Prompt management  │
│ • Extract    │ │ • Similarity │ │ • Response generation│
│   metadata   │ │   search     │ │ • Context handling   │
└──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘
       │                │                    │
       ▼                ▼                    ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
│  Documents   │ │ Vector Store │ │   OpenAI API         │
│  (PDF, Word, │ │  (ChromaDB)  │ │                      │
│   Markdown,  │ │              │ │ • GPT-4              │
│   Text)      │ │ • Embeddings │ │ • Azure OpenAI       │
└──────────────┘ └──────────────┘ └──────────────────────┘
```

## Component Details

### 1. User Interfaces

**REST API Client**
- HTTP/JSON communication
- Supports all CRUD operations
- Session management
- File uploads

**CLI Interface**
- Interactive chat mode
- Single query mode
- Document ingestion commands
- Server management

### 2. API Layer (FastAPI)

**Endpoints:**
- `GET /health` - Health check
- `POST /chat` - Send query and receive response
- `POST /ingest` - Ingest documents from path
- `POST /upload` - Upload document file
- `POST /rebuild` - Rebuild knowledge base
- `GET /session/{id}` - Get conversation history
- `DELETE /session/{id}` - Clear conversation history

**Features:**
- CORS support
- Request/response validation (Pydantic)
- Async processing
- Error handling
- Logging

### 3. Conversational Assistant

**Core Responsibilities:**
- Orchestrate RAG pipeline
- Manage conversation sessions
- Track conversation history
- Coordinate retrieval and generation

**RAG Pipeline:**
1. Receive user query
2. Retrieve relevant documents from vector store
3. Build context from retrieved documents
4. Generate response using LLM with context
5. Return response with metadata

### 4. Document Processor

**Capabilities:**
- Support multiple formats (PDF, Word, Markdown, Text)
- Intelligent text chunking
- Metadata extraction
- Batch processing

**Chunking Strategy:**
- Recursive character splitting
- Configurable chunk size (default: 1000)
- Overlap for context preservation (default: 200)
- Separators: paragraphs, sentences, words

### 5. Vector Store Manager

**Technology:** ChromaDB with HuggingFace Embeddings

**Features:**
- Create and load vector stores
- Add documents incrementally
- Similarity search
- Persistence to disk

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional embeddings
- Fast inference
- Good balance of speed and accuracy

### 6. LLM Manager

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5-turbo)
- Azure OpenAI

**Features:**
- Dynamic system prompts
- Conversation history management
- Temperature control
- Token limit management
- Error handling and retries

**RAG Enhancement:**
- Context injection from retrieved documents
- Source citation
- Grounded response generation

## Data Flow

### Query Processing Flow

```
1. User Query
   ↓
2. API Request (/chat endpoint)
   ↓
3. Conversational Assistant
   ↓
4. Vector Store Search (if RAG enabled)
   ├─ Generate query embedding
   ├─ Similarity search in ChromaDB
   └─ Return top K documents
   ↓
5. LLM Manager
   ├─ Build context from retrieved docs
   ├─ Format prompt with conversation history
   └─ Call OpenAI/Azure API
   ↓
6. Response Generation
   ├─ LLM generates response
   ├─ Add metadata (sources, timing)
   └─ Update conversation history
   ↓
7. API Response
   └─ Return to user
```

### Document Ingestion Flow

```
1. Document Upload/Path
   ↓
2. API Request (/ingest or /upload)
   ↓
3. Document Processor
   ├─ Load document(s)
   ├─ Extract text
   └─ Split into chunks
   ↓
4. Vector Store Manager
   ├─ Generate embeddings for chunks
   ├─ Store in ChromaDB
   └─ Persist to disk
   ↓
5. Success Response
```

## Configuration

**Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_MODEL` - Model name (default: gpt-4)
- `EMBEDDING_MODEL` - Embedding model name
- `VECTOR_DB_PATH` - Vector database storage path
- `CHUNK_SIZE` - Document chunk size
- `CHUNK_OVERLAP` - Chunk overlap size
- `TOP_K_RESULTS` - Number of documents to retrieve
- `API_HOST` - API server host
- `API_PORT` - API server port
- `LOG_LEVEL` - Logging level

## Security Considerations

1. **API Key Management:** Environment variables, never hardcoded
2. **Input Validation:** Pydantic models for request validation
3. **File Upload Security:** Temporary storage, cleanup after processing
4. **Session Isolation:** Per-session conversation history
5. **Error Handling:** Sensitive information not exposed in errors
6. **Logging:** Structured logging without sensitive data

## Scalability

**Current Design:**
- Single server deployment
- In-memory session storage
- File-based vector store

**Future Enhancements:**
- Distributed vector store (Pinecone, Weaviate)
- Redis for session management
- Load balancing
- Caching layer
- Database for conversation history
- Kubernetes deployment

## Performance Optimization

1. **Embedding Generation:** GPU acceleration for production
2. **Vector Search:** Index optimization, approximate search
3. **LLM Calls:** Response caching, streaming responses
4. **Chunking:** Optimal chunk size based on content type
5. **Batch Processing:** Parallel document processing

## Monitoring & Observability

**Metrics:**
- Query processing time
- Vector search latency
- LLM response time
- Error rates
- Document ingestion throughput
- API endpoint performance

**Logging:**
- Structured JSON logs
- Request/response tracking
- Error tracking with stack traces
- Performance metrics

## Deployment Options

1. **Docker:** Single container deployment
2. **Docker Compose:** Multi-container with volumes
3. **Kubernetes:** Scalable production deployment
4. **Cloud Platforms:** AWS, Azure, GCP

## Technology Stack

- **Language:** Python 3.8+
- **Web Framework:** FastAPI
- **LLM Integration:** LangChain, OpenAI SDK
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace Transformers
- **Document Processing:** PyPDF, python-docx, UnstructuredIO
- **API Server:** Uvicorn (ASGI)
- **Configuration:** Pydantic Settings
- **Logging:** Python JSON Logger

## Future Enhancements

1. Multi-language support
2. Advanced analytics dashboard
3. Fine-tuning capabilities
4. Integration with Slack, Teams, etc.
5. Voice interface
6. Custom model support
7. Enhanced security (OAuth, RBAC)
8. Real-time streaming responses
9. Multi-modal support (images, audio)
10. A/B testing framework
