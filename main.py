"""FastAPI application for the conversational assistant."""
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import aiofiles
from pathlib import Path
import tempfile
import os

from assistant import ConversationalAssistant
from config import settings
from logger import setup_logging

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="GeneralBot - AI Conversational Assistant",
    description="AI-powered conversational assistant with LLM and RAG capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the conversational assistant
assistant = ConversationalAssistant()


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model."""
    query: str
    session_id: Optional[str] = None
    use_rag: bool = True


class ChatResponse(BaseModel):
    """Chat response model."""
    query: str
    response: str
    session_id: str
    processing_time: float
    sources_used: int
    timestamp: str


class IngestRequest(BaseModel):
    """Document ingestion request model."""
    source_path: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for conversational queries.
    
    Args:
        request: Chat request containing query and optional session_id
        
    Returns:
        Chat response with generated answer
    """
    try:
        logger.info(f"Received chat request: {request.query[:50]}...")
        result = assistant.chat(
            query=request.query,
            session_id=request.session_id,
            use_rag=request.use_rag
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the knowledge base.
    
    Args:
        request: Ingestion request with source path
        
    Returns:
        Success message
    """
    try:
        logger.info(f"Ingesting documents from {request.source_path}")
        assistant.ingest_documents(request.source_path)
        return {
            "status": "success",
            "message": f"Documents ingested from {request.source_path}"
        }
    except Exception as e:
        logger.error(f"Error ingesting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document.
    
    Args:
        file: Uploaded file
        
    Returns:
        Success message
    """
    try:
        # Create temporary directory for uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = Path(temp_dir) / file.filename
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Uploaded file: {file.filename}")
        
        # Ingest the document
        assistant.ingest_documents(str(file_path))
        
        # Clean up
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        return {
            "status": "success",
            "message": f"Document {file.filename} uploaded and ingested successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild_knowledge_base(request: IngestRequest):
    """Rebuild the knowledge base from scratch.
    
    Args:
        request: Ingestion request with source path
        
    Returns:
        Success message
    """
    try:
        logger.info("Rebuilding knowledge base")
        assistant.rebuild_knowledge_base(request.source_path)
        return {
            "status": "success",
            "message": "Knowledge base rebuilt successfully"
        }
    except Exception as e:
        logger.error(f"Error rebuilding knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    try:
        assistant.clear_conversation_history(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation history
    """
    try:
        history = assistant.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the FastAPI application."""
    logger.info(f"Starting GeneralBot API on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
