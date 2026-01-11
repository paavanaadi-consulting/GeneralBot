# Technical Documentation

## Architecture Overview

GeneralBot uses a modern microservices architecture with the following components:

### Core Components

1. **Document Processor**
   - Handles ingestion of various document formats
   - Splits documents into optimally-sized chunks
   - Extracts metadata and maintains document lineage

2. **Vector Store Manager**
   - Manages document embeddings using state-of-the-art models
   - Implements efficient similarity search using ChromaDB
   - Handles incremental updates to the knowledge base

3. **LLM Manager**
   - Interfaces with OpenAI or Azure OpenAI
   - Manages conversation context and history
   - Implements response generation strategies

4. **Conversational Assistant**
   - Orchestrates the RAG pipeline
   - Manages multi-turn conversations
   - Tracks session state and history

## API Documentation

### Authentication

Currently, the API is open for development. Production deployments should implement:
- API key authentication
- OAuth 2.0 for user-based access
- Rate limiting per client

### Endpoints

#### POST /chat
Send a query and receive an AI-generated response.

**Request:**
```json
{
  "query": "What are your business hours?",
  "session_id": "optional-session-id",
  "use_rag": true
}
```

**Response:**
```json
{
  "query": "What are your business hours?",
  "response": "Our business hours are Monday-Friday 9AM-6PM EST...",
  "session_id": "session_123",
  "processing_time": 1.23,
  "sources_used": 2,
  "timestamp": "2026-01-11T18:00:00"
}
```

#### POST /ingest
Ingest documents from a local path.

**Request:**
```json
{
  "source_path": "/path/to/documents"
}
```

#### POST /upload
Upload a document file.

**Request:** Multipart form data with file

#### GET /health
Health check endpoint.

## Deployment Guide

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t generalbot:latest .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key generalbot:latest
```

### Kubernetes Deployment

Use the provided Kubernetes manifests in the `k8s/` directory.

### Environment Variables

See `.env.example` for all configuration options.

## Performance Tuning

### Vector Store Optimization

- Adjust `CHUNK_SIZE` based on document complexity
- Increase `TOP_K_RESULTS` for better context but slower response
- Use GPU acceleration for embedding generation in production

### LLM Optimization

- Adjust temperature (0.7 default) for response creativity
- Set appropriate max_tokens to control response length
- Use streaming for real-time response display

## Monitoring & Logging

All operations are logged to:
- Console output (structured)
- Log file in JSON format (`./logs/app.log`)

Key metrics to monitor:
- Query processing time
- Vector search latency
- LLM response time
- Error rates
- Document ingestion throughput

## Troubleshooting

### Common Issues

**Issue**: "Vector store not found"
**Solution**: Ensure documents have been ingested using `/ingest` endpoint

**Issue**: "OpenAI API error"
**Solution**: Verify API key is correct and has sufficient credits

**Issue**: "Out of memory"
**Solution**: Reduce batch size for document processing or upgrade server resources

## Maintenance

### Regular Tasks

1. **Knowledge Base Updates**: Regularly update documents to keep information current
2. **Log Rotation**: Implement log rotation to manage disk space
3. **Performance Monitoring**: Track response times and optimize as needed
4. **Security Updates**: Keep dependencies updated

### Backup & Recovery

- Backup vector database directory regularly
- Store document source files separately
- Implement disaster recovery procedures
