# GeneralBot - AI-Powered Conversational Assistant

An intelligent conversational assistant leveraging **Large Language Models (LLM)** and **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware responses by combining the reasoning capabilities of modern LLMs with organization-specific knowledge retrieved from internal documents and databases.

## 🌟 Features

- **🤖 Advanced LLM Integration**: Powered by OpenAI GPT-4 or Azure OpenAI
- **📚 RAG System**: Retrieval-Augmented Generation for grounded, accurate responses
- **🔍 Intelligent Document Processing**: Supports PDF, Word, Markdown, and Text files
- **💾 Vector Database**: Efficient similarity search using ChromaDB with semantic embeddings
- **💬 Context-Aware Conversations**: Maintains conversation history for coherent multi-turn dialogues
- **🌐 REST API**: FastAPI-based API for easy integration
- **⚡ 24/7 Availability**: Automated customer support and information retrieval
- **🐳 Docker Support**: Easy deployment with Docker and Docker Compose
- **📊 Comprehensive Logging**: JSON-structured logging for monitoring and debugging

## 🏗️ Architecture

GeneralBot uses a modern architecture with the following components:

1. **Document Processor**: Ingests and chunks documents for optimal retrieval
2. **Vector Store Manager**: Manages document embeddings and similarity search
3. **LLM Manager**: Interfaces with OpenAI/Azure OpenAI for response generation
4. **Conversational Assistant**: Orchestrates the RAG pipeline and manages conversations
5. **FastAPI Server**: Provides REST API endpoints for integration

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (or Azure OpenAI credentials)
- 8GB RAM minimum (16GB recommended)
- 10GB storage space

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/paavanaadi-consulting/GeneralBot.git
cd GeneralBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Ingest Documents

Add documents to the knowledge base:

```bash
python cli.py ingest ./data/documents
```

### 5. Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Or Use Interactive Chat

```bash
python cli.py chat
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f
```

### Using Docker

```bash
# Build the image
docker build -t generalbot:latest .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key generalbot:latest
```

## 📖 Usage

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Chat with the Assistant
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are your business hours?",
    "use_rag": true
  }'
```

#### Upload a Document
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```

#### Ingest Documents from Path
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/path/to/documents"}'
```

#### Clear Session History
```bash
curl -X DELETE http://localhost:8000/session/session_id
```

### Command-Line Interface

#### Interactive Chat
```bash
python cli.py chat
```

#### Single Query
```bash
python cli.py query "What services do you offer?"
```

#### Ingest Documents
```bash
python cli.py ingest ./data/documents
```

#### Rebuild Knowledge Base
```bash
python cli.py rebuild ./data/documents
```

## 📁 Project Structure

```
GeneralBot/
├── assistant.py              # Main conversational assistant
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
├── document_processor.py     # Document ingestion and processing
├── llm_manager.py           # LLM integration
├── logger.py                 # Logging configuration
├── main.py                   # FastAPI application
├── vector_store.py          # Vector database management
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── .env.example             # Example environment variables
├── .gitignore              # Git ignore file
├── data/
│   ├── documents/          # Sample knowledge base documents
│   └── vector_db/          # Vector database (auto-created)
└── logs/                    # Application logs (auto-created)
```

## ⚙️ Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4)
- `EMBEDDING_MODEL`: Embedding model for vector search
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap size (default: 200)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 3)

## 🔧 Advanced Features

### Custom System Prompts

Modify the system prompt in `llm_manager.py` to customize the assistant's behavior.

### Supported Document Formats

- PDF (`.pdf`)
- Microsoft Word (`.docx`, `.doc`)
- Text (`.txt`)
- Markdown (`.md`)

### Session Management

The assistant maintains conversation history per session, enabling context-aware multi-turn conversations.

### Performance Tuning

- Adjust `CHUNK_SIZE` for optimal retrieval granularity
- Increase `TOP_K_RESULTS` for more context (slower responses)
- Use GPU for faster embedding generation in production

## 🔐 Security Features

- Environment-based configuration (no hardcoded secrets)
- API key authentication support (extend for production)
- Secure document handling
- JSON-structured logging for audit trails

## 📊 Monitoring

Logs are written to:
- Console (human-readable format)
- `./logs/app.log` (JSON format for analysis)

Key metrics tracked:
- Query processing time
- Number of sources used
- Vector search performance
- LLM response time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: support@generalbot.com

## 🎯 Use Cases

- **Customer Support**: 24/7 automated responses based on company knowledge
- **Internal Knowledge Management**: Quick access to company documentation
- **Decision Support**: Context-aware recommendations based on historical data
- **Information Retrieval**: Fast, accurate answers from large document collections
- **Employee Onboarding**: Interactive assistant for new employee questions

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Integration with popular platforms (Slack, Teams, etc.)
- [ ] Advanced analytics dashboard
- [ ] Fine-tuning capabilities
- [ ] Custom model support
- [ ] Enhanced security features (OAuth, RBAC)

---

Built with ❤️ using LangChain, OpenAI, and FastAPI