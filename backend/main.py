"""FastAPI entrypoint for the chatbot backend."""

import uuid
import json
from datetime import datetime
from typing import Optional
import dotenv
import logging

# Load environment variables FIRST before importing config
dotenv.load_dotenv("secrets.env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from config import HOST, PORT, DEBUG
from models import (
    ChatInput, ThreadListResponse, ThreadResponse,
    MessageListResponse, MessageResponse, DocumentListResponse,
    DocumentResponse, UploadResponse, HealthResponse, ExcelUploadResponse,
    HITLResumeInput
)
from server import ChatServer
from memory_store import initialize_database
import document_store
import frontend_store
import data_loader



# Configure logging
log_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Global server instance
chat_server: Optional[ChatServer] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global chat_server
    
    # Startup
    # Initialize document store tables
    document_store.create_tables()
    logger.info("✅ Document store initialized")
    
    # Initialize memory store (matching IResearcher-v5 pattern)
    await initialize_database()
    
    # Initialize frontend store
    await frontend_store.initialize_database()
    logger.info("✅ Frontend store initialized")
    
    # Initialize data loader (DuckDB)
    data_loader.initialize_database()
    logger.info("✅ Data loader initialized")
    
    # Initialize chat server
    chat_server = ChatServer()
    logger.info("✅ Chat server initialized")
    
    yield
    
    # Shutdown (if needed in the future)


# Initialize FastAPI app
app = FastAPI(title="LangGraph Chatbot API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    is_new_thread = input_data.thread_id is None
    
    # Create or update user in frontend store
    try:
        await frontend_store.create_or_update_user(input_data.user_id)
    except Exception as e:
        logger.warning(f"Error creating/updating user: {e}")
    
    # Create thread if new
    if is_new_thread:
        try:
            # Generate title from user query
            title = frontend_store.generate_title_from_query(input_data.query)
            await frontend_store.create_thread(
                thread_id=thread_id,
                user_id=input_data.user_id,
                title=title
            )
        except Exception as e:
            logger.warning(f"Error creating thread: {e}")
    
    # Save user message to frontend store
    try:
        await frontend_store.add_message(
            thread_id=thread_id,
            role="user",
            content=input_data.query,
            user_id=input_data.user_id
        )
    except Exception as e:
        logger.warning(f"Error saving user message: {e}")
    
    # Build context from multiple sources
    context_parts = []
    
    # Get document context if user has documents
    try:
        user_docs = document_store.list_documents(input_data.user_id)
        if user_docs:
            doc_ids = [doc["doc_id"] for doc in user_docs]
            doc_context = document_store.get_document_context(
                user_id=input_data.user_id,
                doc_ids=doc_ids,
                max_tokens=1000
            )
            if doc_context:
                context_parts.append("=== DOCUMENT CONTEXT ===")
                context_parts.append(doc_context)
    except Exception as e:
        logger.warning(f"Error loading document context: {e}")
    
    # Get data loader context (metadata and schema)
    try:
        # Get metadata CSV
        metadata_csv = data_loader.get_metadata_csv(user_id=input_data.user_id, session_id=thread_id)
        if metadata_csv:
            context_parts.append("\n=== AVAILABLE DATA TABLES (METADATA) ===")
            context_parts.append(metadata_csv)
        
        # Get tables grouped by schema
        schema_markdown = data_loader.get_tables_by_schema_csv(user_id=input_data.user_id)
        if schema_markdown:
            context_parts.append("\n=== DATA TABLE SCHEMAS ===")
            context_parts.append(schema_markdown)
    except Exception as e:
        logger.warning(f"Error loading data loader context: {e}")
    
    # Combine all context parts
    context = "\n".join(context_parts) if context_parts else ""
    
    async def generate():
        """Generate streaming response."""
        # Collect tool events during streaming
        tool_events = []
        
        try:
            async for chunk in chat_server.process_message(
                message=input_data.query,
                thread_id=thread_id,
                user_id=input_data.user_id,
                model_id=input_data.model_id,
                context=context
            ):
                # Stream chunk to frontend
                yield json.dumps(chunk) + "\n"
                
                # Collect tool events
                if chunk.get("type") == "tool_start":
                    tool_events.append({
                        "type": "tool_start",
                        "name": chunk.get("name", "unknown"),
                        "input": chunk.get("input", {})
                    })
                    
                elif chunk.get("type") == "tool_end":
                    tool_events.append({
                        "type": "tool_end",
                        "name": chunk.get("name", "unknown"),
                        "output": chunk.get("output", "")
                    })
                    
                elif chunk.get("type") == "full_response":
                    # Save assistant message with all tool events at once
                    try:
                        await frontend_store.add_message(
                            thread_id=thread_id,
                            role="assistant",
                            content=chunk.get("content", ""),
                            user_id=input_data.user_id,
                            tool_events=tool_events if tool_events else None
                        )
                    except Exception as e:
                        logger.warning(f"Error saving assistant message: {e}")
                    
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


@app.post("/api/v1/chat/resume")
async def resume_chat(input_data: HITLResumeInput):
    """
    Resume chat execution after a human-in-the-loop interrupt.
    
    This endpoint is called when the user provides a decision for an interrupt
    (e.g., answering a question from the ask_question tool).
    """
    global chat_server
    
    if not chat_server:
        raise HTTPException(status_code=503, detail="Chat server not initialized")
    
    # Build context (same as chat endpoint)
    context_parts = []
    
    # Get document context if user has documents
    try:
        user_docs = document_store.list_documents(input_data.user_id)
        if user_docs:
            doc_ids = [doc["doc_id"] for doc in user_docs]
            doc_context = document_store.get_document_context(
                user_id=input_data.user_id,
                doc_ids=doc_ids,
                max_tokens=1000
            )
            if doc_context:
                context_parts.append("=== DOCUMENT CONTEXT ===")
                context_parts.append(doc_context)
    except Exception as e:
        logger.warning(f"Error loading document context: {e}")
    
    # Get data loader context
    try:
        metadata_csv = data_loader.get_metadata_csv(user_id=input_data.user_id, session_id=input_data.thread_id)
        if metadata_csv:
            context_parts.append("\n=== AVAILABLE DATA TABLES (METADATA) ===")
            context_parts.append(metadata_csv)
        
        schema_markdown = data_loader.get_tables_by_schema_csv(user_id=input_data.user_id)
        if schema_markdown:
            context_parts.append("\n=== DATA TABLE SCHEMAS ===")
            context_parts.append(schema_markdown)
    except Exception as e:
        logger.warning(f"Error loading data loader context: {e}")
    
    context = "\n".join(context_parts) if context_parts else ""
    
    async def generate():
        """Generate streaming response after resume."""
        tool_events = []
        
        try:
            # Convert decisions to dict format
            decisions = [decision.dict() for decision in input_data.decisions]
            
            async for chunk in chat_server.resume_after_interrupt(
                thread_id=input_data.thread_id,
                user_id=input_data.user_id,
                decisions=decisions,
                context=context
            ):
                # Stream chunk to frontend
                yield json.dumps(chunk) + "\n"
                
                # Collect tool events
                if chunk.get("type") == "tool_start":
                    tool_events.append({
                        "type": "tool_start",
                        "name": chunk.get("name", "unknown"),
                        "input": chunk.get("input", {})
                    })
                    
                elif chunk.get("type") == "tool_end":
                    tool_events.append({
                        "type": "tool_end",
                        "name": chunk.get("name", "unknown"),
                        "output": chunk.get("output", "")
                    })
                    
                elif chunk.get("type") == "full_response":
                    # Save assistant message
                    try:
                        await frontend_store.add_message(
                            thread_id=input_data.thread_id,
                            role="assistant",
                            content=chunk.get("content", ""),
                            user_id=input_data.user_id,
                            tool_events=tool_events if tool_events else None
                        )
                    except Exception as e:
                        logger.warning(f"Error saving assistant message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in resume stream: {e}")
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


@app.post("/api/v1/upload-excel", response_model=ExcelUploadResponse)
async def upload_excel_file(
    user_id: str = Form(...),
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload an Excel (.xlsx) file and store it in DuckDB.
    
    Accepts:
    - user_id: User ID
    - session_id: Session ID
    - file: Excel file (.xlsx only)
    
    Returns table name, row count, column count, and metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file type - only .xlsx files
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")
    
    try:
        # Read file bytes
        excel_bytes = await file.read()
        
        if not excel_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Load Excel file into DuckDB
        result = data_loader.load_excel_file(
            excel_bytes=excel_bytes,
            filename=file.filename,
            user_id=user_id,
            session_id=session_id
        )
        
        return ExcelUploadResponse(
            message=f"Successfully loaded {result['row_count']} rows into table {result['table_name']}",
            table_name=result['table_name'],
            filename=result['filename'],
            row_count=result['row_count'],
            column_count=result['column_count'],
            columns=result['columns'],
            metadata=result['metadata']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading Excel file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")
    finally:
        await file.close()


@app.get("/api/v1/metadata-csv")
async def get_metadata_csv(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID")
):
    """
    Get metadata table as CSV string, filtered by user_id and/or session_id.
    
    Returns CSV string with all metadata columns.
    """
    try:
        csv_data = data_loader.get_metadata_csv(user_id=user_id, session_id=session_id)
        
        if not csv_data:
            raise HTTPException(status_code=404, detail="No metadata found for the specified filters")
        
        from fastapi.responses import Response
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=metadata.csv"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metadata: {str(e)}")


@app.get("/api/v1/tables-by-schema-csv")
async def get_tables_by_schema_csv(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID")
):
    """
    Get table names grouped by unique table schemas (column name and column type only).
    
    Returns markdown with embedded CSV sections. Each section shows:
    - Tables with the same schema grouped together
    - Schema information (column_name, column_type) as CSV
    
    Format:
    ---
    ## tables: [table1, table2]
    ## schema:
    csv_data
    ---
    """
    try:
        markdown_data = data_loader.get_tables_by_schema_csv(user_id=user_id, session_id=session_id)
        
        if not markdown_data:
            raise HTTPException(status_code=404, detail="No tables found for the specified filters")
        
        from fastapi.responses import Response
        return Response(
            content=markdown_data,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=tables_by_schema.md"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tables by schema CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting tables by schema: {str(e)}")


@app.get("/api/v1/threads", response_model=ThreadListResponse)
async def list_threads(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of threads to return"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order")
):
    """List conversation threads for a user with pagination."""
    try:
        result = await frontend_store.list_threads(
            user_id=user_id,
            limit=limit,
            after=after,
            order=order
        )
        
        # Convert to response model
        threads = [
            ThreadResponse(
                id=thread["id"],
                user_id=thread["user_id"],
                title=thread.get("title"),
                created_at=datetime.fromisoformat(thread["created_at"]),
                updated_at=datetime.fromisoformat(thread["updated_at"])
            )
            for thread in result["threads"]
        ]
        
        return ThreadListResponse(
            threads=threads,
            has_more=result["has_more"],
            after=result["after"]
        )
    except Exception as e:
        logger.error(f"Error listing threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/threads/{thread_id}/messages", response_model=MessageListResponse)
async def get_thread_messages(
    thread_id: str,
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order")
):
    """
    Get messages from a conversation thread with pagination.
    """
    try:
        result = await frontend_store.get_messages(
            thread_id=thread_id,
            user_id=user_id,
            limit=limit,
            after=after,
            order=order
        )
        
        # Convert to response model
        messages = [
            MessageResponse(
                id=msg["id"],
                thread_id=msg["thread_id"],
                role=msg["role"],
                content=msg["content"],
                created_at=datetime.fromisoformat(msg["created_at"]),
                tool_events=msg.get("tool_events")
            )
            for msg in result["messages"]
        ]
        
        return MessageListResponse(
            messages=messages,
            has_more=result["has_more"],
            after=result["after"]
        )
    except frontend_store.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting thread messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/threads/{thread_id}")
async def delete_thread(thread_id: str, user_id: str = Query(..., description="User ID")):
    """Delete a conversation thread from both stores."""
    try:
        # Delete from frontend store
        await frontend_store.delete_thread(thread_id, user_id)
        
        # Note: Also delete from LangGraph checkpoints if needed
        # This would require implementing checkpoint deletion in memory_store
        
        return {"message": f"Thread {thread_id} deleted successfully"}
    except frontend_store.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            "chat_resume": "POST /api/v1/chat/resume - Resume chat after human-in-the-loop interrupt",
            "upload": "POST /api/v1/upload - Upload PDF documents",
            "upload_excel": "POST /api/v1/upload-excel - Upload Excel (.xlsx) files to DuckDB",
            "metadata_csv": "GET /api/v1/metadata-csv - Get metadata table as CSV (filtered by user_id, session_id)",
            "tables_by_schema_csv": "GET /api/v1/tables-by-schema-csv - Get tables grouped by schema as CSV",
            "threads": "GET /api/v1/threads - List conversation threads",
            "messages": "GET /api/v1/threads/{thread_id}/messages - Get thread messages",
            "delete_thread": "DELETE /api/v1/threads/{thread_id} - Delete thread",
            "documents": "GET /api/v1/documents - List user documents",
            "health": "GET /api/v1/health - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    log_level = "debug" if DEBUG else "info"
    
    # When reload is enabled, use import string format
    if DEBUG:
        uvicorn.run(
            "main:app",  # Import string format for reload
            host=HOST, 
            port=PORT,
            log_level=log_level,
            reload=True  # Auto-reload on code changes in debug mode
        )
    else:
        uvicorn.run(
            app,  # Direct app object when reload is disabled
            host=HOST, 
            port=PORT,
            log_level=log_level
        )

