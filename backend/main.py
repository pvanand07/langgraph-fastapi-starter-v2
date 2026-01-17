"""FastAPI entrypoint for the chatbot backend."""

import os
import uuid
import json
from datetime import datetime
from typing import Optional, List
import dotenv
import logging

# Load environment variables FIRST before importing config
dotenv.load_dotenv("secrets.env")

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from config import HOST, PORT
from models import (
    ChatInput, ChatChunk, ThreadListResponse, ThreadResponse,
    MessageListResponse, MessageResponse, DocumentListResponse,
    DocumentResponse, UploadResponse, HealthResponse
)
from server import ChatServer
from memory_store import initialize_database
import document_store



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LangGraph Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global server instance
chat_server: Optional[ChatServer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global chat_server
    
    # Initialize document store tables
    document_store.create_tables()
    logger.info("✅ Document store initialized")
    
    # Initialize memory store (matching IResearcher-v5 pattern)
    await initialize_database()
    
    # Initialize chat server
    chat_server = ChatServer()
    logger.info("✅ Chat server initialized")


@app.post("/api/v1/chat")
async def chat(input_data: ChatInput):
    """
    Main chat endpoint with SSE streaming.
    
    Supports:
    - Text messages
    - Conversation threading
    - Model selection
    - Tool calling
    """
    global chat_server
    
    if not chat_server:
        raise HTTPException(status_code=503, detail="Chat server not initialized")
    
    # Generate thread_id if not provided
    thread_id = input_data.thread_id or str(uuid.uuid4())
    
    # Get document context if user has documents
    context = ""
    try:
        user_docs = document_store.list_documents(input_data.user_id)
        if user_docs:
            doc_ids = [doc["doc_id"] for doc in user_docs]
            context = document_store.get_document_context(
                user_id=input_data.user_id,
                doc_ids=doc_ids,
                max_tokens=1000
            )
    except Exception as e:
        logger.warning(f"Error loading document context: {e}")
    
    async def generate():
        """Generate streaming response."""
        try:
            async for chunk in chat_server.process_message(
                message=input_data.query,
                thread_id=thread_id,
                user_id=input_data.user_id,
                model_id=input_data.model_id,
                context=context
            ):
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield json.dumps({
                "type": "error",
                "content": f"Error: {str(e)}"
            }) + "\n"
    
    return EventSourceResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a PDF file and store it as a document.
    
    Returns document ID and page count.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file bytes
        pdf_bytes = await file.read()
        
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Generate doc_id from filename
        doc_id = document_store.generate_doc_id(file.filename)
        
        # Extract pages
        pages = document_store.extract_pages_text_from_bytes(
            pdf_bytes=pdf_bytes,
            doc_id=doc_id,
            user_id=user_id,
            filename=file.filename
        )
        
        if not pages:
            raise HTTPException(
                status_code=500,
                detail="Could not extract text from PDF. The document may be empty or contain only images."
            )
        
        # Store pages
        document_store.store_pages(
            pages=pages,
            user_id=user_id,
            doc_id=doc_id,
            doc_name=file.filename
        )
        
        return UploadResponse(
            message=f"Successfully stored {len(pages)} pages",
            doc_id=doc_id,
            page_count=len(pages)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        await file.close()


@app.get("/api/v1/threads", response_model=ThreadListResponse)
async def list_threads(
    user_id: str = Query(..., description="User ID"),  # noqa: ARG001
    limit: int = Query(20, ge=1, le=100, description="Number of threads to return"),  # noqa: ARG001
    after: Optional[str] = Query(None, description="Cursor for pagination"),  # noqa: ARG001
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")  # noqa: ARG001
):
    """List conversation threads for a user with pagination."""
    # Note: Thread listing functionality skipped per user request
    # In production, you would query the checkpoints table directly
    return ThreadListResponse(
        threads=[],
        has_more=False,
        after=None
    )


@app.get("/api/v1/threads/{thread_id}/messages", response_model=MessageListResponse)
async def get_thread_messages(
    thread_id: str,  # noqa: ARG001
    user_id: str = Query(..., description="User ID"),  # noqa: ARG001
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),  # noqa: ARG001
    after: Optional[str] = Query(None, description="Cursor for pagination"),  # noqa: ARG001
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")  # noqa: ARG001
):
    """
    Get messages from a conversation thread with pagination.
    
    Note: This is a simplified implementation. In production, you'd want
    to extract messages from LangGraph checkpoints or maintain a separate
    messages table for better querying.
    """
    # For now, return empty list as LangGraph checkpoints don't expose
    # a simple message history API. In production, you'd want to:
    # 1. Store messages separately in a messages table, or
    # 2. Extract messages from checkpoint state
    
    return MessageListResponse(
        messages=[],
        has_more=False,
        after=None
    )


@app.delete("/api/v1/threads/{thread_id}")
async def delete_thread(thread_id: str, user_id: str = Query(..., description="User ID")):  # noqa: ARG001
    """Delete a conversation thread."""
    # Note: Thread deletion functionality skipped per user request
    # In production, you would delete checkpoints for the thread_id
    return {"message": "Thread deletion not implemented"}


@app.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents(user_id: str = Query(..., description="User ID")):
    """List all documents for a user."""
    try:
        docs = document_store.list_documents(user_id)
        documents = [
            DocumentResponse(
                doc_id=doc["doc_id"],
                doc_name=doc["doc_name"],
                user_id=doc["user_id"],
                page_count=doc["page_count"],
                created_at=doc["created_at"]
            )
            for doc in docs
        ]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="LangGraph Chatbot",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "message": "LangGraph Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /api/v1/chat - Chat with streaming responses",
            "upload": "POST /api/v1/upload - Upload PDF documents",
            "threads": "GET /api/v1/threads - List conversation threads",
            "messages": "GET /api/v1/threads/{thread_id}/messages - Get thread messages",
            "delete_thread": "DELETE /api/v1/threads/{thread_id} - Delete thread",
            "documents": "GET /api/v1/documents - List user documents",
            "health": "GET /api/v1/health - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

