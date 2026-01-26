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
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from config import HOST, PORT, DEBUG
from models import (
    ChatInput, ThreadListResponse, ThreadResponse,
    MessageListResponse, MessageResponse, DocumentListResponse,
    DocumentResponse, HealthResponse, UnifiedUploadResponse, MultiUploadResponse,
    RenameThreadRequest, UnifiedDocumentListResponse, UnifiedDocumentItem
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

# Get the frontend directory path
# Try Docker path first (/app/frontend), then local development path
frontend_dir = None
docker_path = Path("/app/frontend")
local_path = Path(__file__).parent.parent / "frontend"

if docker_path.exists():
    frontend_dir = docker_path
elif local_path.exists():
    frontend_dir = local_path

# Mount static files from frontend directory
if frontend_dir:
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"✅ Frontend static files mounted from {frontend_dir}")
else:
    logger.warning(f"⚠️ Frontend directory not found (checked {docker_path} and {local_path})")


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
                    tool_event = {
                        "type": "tool_end",
                        "name": chunk.get("name", "unknown"),
                        "output": chunk.get("output", "")
                    }
                    # Include artifacts_data if present
                    if chunk.get("artifacts_data") is not None:
                        tool_event["artifacts_data"] = chunk.get("artifacts_data")
                    tool_events.append(tool_event)
                    
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


async def process_single_file(
    file: UploadFile,
    user_id: str,
    session_id: Optional[str] = None
) -> UnifiedUploadResponse:
    """Process a single file upload."""
    if not file.filename:
        raise HTTPException(status_code=400, detail=f"No filename provided for file")
    
    filename_lower = file.filename.lower()
    
    # Read file bytes
    file_bytes = await file.read()
    
    if not file_bytes:
        raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")
    
    # Route based on file type
    if filename_lower.endswith(('.pdf', '.docx')):
        # Calculate file hash
        file_hash = document_store.calculate_file_hash(file_bytes)
        
        # Check for duplicate file (same hash for same user)
        duplicate = document_store.check_duplicate_file(user_id, file_hash)
        if duplicate:
            # Return success message for duplicate files without updating the database
            return UnifiedUploadResponse(
                file_type="document",
                message=f"File '{file.filename}' already exists in the database (doc_id: {duplicate['doc_id']})",
                filename=file.filename,
                doc_id=duplicate['doc_id'],
                page_count=duplicate.get('page_count')
            )
        
        # Handle PDF or DOCX documents
        # Generate doc_id from filename
        doc_id = document_store.generate_doc_id(file.filename)
        
        # Extract pages based on file type
        if filename_lower.endswith('.pdf'):
            pages = document_store.extract_pages_text_from_bytes(
                pdf_bytes=file_bytes,
                doc_id=doc_id,
                user_id=user_id,
                filename=file.filename
            )
        else:  # .docx
            pages = document_store.extract_pages_text_from_docx_bytes(
                docx_bytes=file_bytes,
                doc_id=doc_id,
                user_id=user_id,
                filename=file.filename
            )
        
        if not pages:
            raise HTTPException(
                status_code=500,
                detail=f"Could not extract text from {file.filename}. The document may be empty or contain only images."
            )
        
        # Store pages with file hash
        document_store.store_pages(
            pages=pages,
            user_id=user_id,
            doc_id=doc_id,
            doc_name=file.filename,
            file_hash=file_hash
        )
        
        return UnifiedUploadResponse(
            file_type="document",
            message=f"Successfully stored {len(pages)} pages",
            filename=file.filename,
            doc_id=doc_id,
            page_count=len(pages)
        )
    
    elif filename_lower.endswith(('.xlsx', '.xls')):
        # Handle Excel files
        # Calculate file hash and check for duplicates (by user_id and file_hash only, regardless of session_id)
        file_hash = document_store.calculate_file_hash(file_bytes)
        duplicate = data_loader.check_duplicate_excel_file(user_id, file_hash, session_id=None)
        if duplicate:
            # Return success message for duplicate files without updating the database
            return UnifiedUploadResponse(
                file_type="excel",
                message=f"File '{file.filename}' already exists in the database (table: {duplicate['table_name']})",
                filename=file.filename,
                doc_id=duplicate['table_name'],  # Use existing table_name as doc_id
                table_name=duplicate['table_name']
            )
        
        # Generate default session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Load Excel file into DuckDB (session_id is now optional)
        result = data_loader.load_excel_file(
            excel_bytes=file_bytes,
            filename=file.filename,
            user_id=user_id,
            session_id=session_id
        )
        
        return UnifiedUploadResponse(
            file_type="excel",
            message=f"Successfully loaded {result['row_count']} rows into table {result['table_name']}",
            filename=result['filename'],
            doc_id=result['table_name'],  # Use table_name as doc_id for Excel files
            table_name=result['table_name'],
            row_count=result['row_count'],
            column_count=result['column_count'],
            columns=result['columns'],
            metadata=result['metadata']
        )
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF, DOCX, and Excel (.xlsx, .xls) files are supported"
        )


@app.post("/api/v1/upload", response_model=MultiUploadResponse)
async def upload_file(
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None, description="Session ID (optional for all file types)"),
    files: List[UploadFile] = File(...)
):
    """
    Unified file upload endpoint supporting multiple PDF, DOCX, and Excel files.
    
    Accepts:
    - user_id: User ID (required)
    - session_id: Session ID (optional for all file types)
    - files: One or more files to upload (.pdf, .docx, .xlsx, .xls)
    
    Returns:
    - List of upload results with success/failure status for each file
    - Summary of total, successful, and failed uploads
    
    Note: For duplicate files (same user_id and file_hash), returns success message without updating the database.
    This applies to all file types (PDF, DOCX, and Excel files).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    errors = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            result = await process_single_file(file, user_id, session_id)
            results.append(result)
            successful += 1
        except HTTPException as e:
            failed += 1
            errors.append({
                "filename": file.filename or "unknown",
                "error": e.detail,
                "status_code": e.status_code
            })
            logger.error(f"Error uploading file {file.filename}: {e.detail}")
        except Exception as e:
            failed += 1
            error_msg = str(e)
            errors.append({
                "filename": file.filename or "unknown",
                "error": error_msg,
                "status_code": 500
            })
            logger.error(f"Error uploading file {file.filename}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await file.close()
    
    return MultiUploadResponse(
        results=results,
        total_files=len(files),
        successful=successful,
        failed=failed,
        errors=errors if errors else None
    )


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


@app.patch("/api/v1/threads/{thread_id}/rename", response_model=ThreadResponse)
async def rename_thread(
    thread_id: str,
    request: RenameThreadRequest,
    user_id: str = Query(..., description="User ID")
):
    """Rename a conversation thread by updating its title."""
    try:
        result = await frontend_store.update_thread(
            thread_id=thread_id,
            user_id=user_id,
            title=request.title
        )
        
        return ThreadResponse(
            id=result["id"],
            user_id=result["user_id"],
            title=result["title"],
            created_at=datetime.fromisoformat(result["created_at"]),
            updated_at=datetime.fromisoformat(result["updated_at"])
        )
    except frontend_store.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error renaming thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents", response_model=UnifiedDocumentListResponse)
async def list_documents(
    user_id: str = Query(..., description="User ID"),
    session_id: Optional[str] = Query(None, description="Optional session ID filter for Excel files")
):
    """
    List all documents and Excel files for a user in a unified format.
    
    Returns both:
    - Documents (PDF/DOCX) with document metadata
    - Excel files with table metadata and Excel-specific metadata
    """
    try:
        items = []
        
        # Get documents (PDF/DOCX)
        try:
            docs = document_store.list_documents(user_id)
            for doc in docs:
                items.append(UnifiedDocumentItem(
                    file_type="document",
                    filename=doc["doc_name"],
                    created_at=doc["created_at"],
                    doc_id=doc["doc_id"],
                    page_count=doc["page_count"],
                    content_preview=None
                ))
        except Exception as e:
            logger.warning(f"Error loading documents: {e}")
        
        # Get Excel files metadata
        try:
            excel_tables = data_loader.list_tables(user_id=user_id, session_id=session_id)
            for excel_data in excel_tables:
                # Format content_preview as markdown from Excel metadata
                content_preview_parts = []
                if excel_data.get("name"):
                    content_preview_parts.append(f"**Name:** {excel_data.get('name')}")
                if excel_data.get("address"):
                    content_preview_parts.append(f"**Address:** {excel_data.get('address')}")
                if excel_data.get("report_title"):
                    content_preview_parts.append(f"**Report Title:** {excel_data.get('report_title')}")
                if excel_data.get("client"):
                    content_preview_parts.append(f"**Client:** {excel_data.get('client')}")
                
                content_preview = "\n".join(content_preview_parts) if content_preview_parts else None
                
                # Use table_name as doc_id for Excel files
                table_name = excel_data.get("table_name")
                if not table_name:
                    # Fallback: use filename without extension as doc_id
                    filename = excel_data.get("filename", excel_data.get("source_file", "unknown"))
                    table_name = Path(filename).stem if filename != "unknown" else f"excel_{uuid.uuid4().hex[:8]}"
                
                items.append(UnifiedDocumentItem(
                    file_type="excel",
                    filename=excel_data.get("filename", excel_data.get("source_file", "unknown")),
                    created_at=str(excel_data.get("created_at", "")),
                    doc_id=table_name,
                    content_preview=content_preview
                ))
        except Exception as e:
            logger.warning(f"Error loading Excel metadata: {e}")
        
        # Sort by created_at (most recent first)
        items.sort(key=lambda x: x.created_at, reverse=True)
        
        document_count = sum(1 for item in items if item.file_type == "document")
        excel_count = sum(1 for item in items if item.file_type == "excel")
        
        return UnifiedDocumentListResponse(
            items=items,
            total_count=len(items),
            document_count=document_count,
            excel_count=excel_count
        )
    except Exception as e:
        logger.error(f"Error listing unified documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Delete a document or Excel file by doc_id.
    
    Works for both:
    - Documents (PDF/DOCX): doc_id is the document ID
    - Excel files: doc_id is the table name
    
    Returns success message or 404 if not found.
    """
    try:
        # Try to delete as a document first
        deleted = document_store.delete_document(user_id, doc_id)
        if deleted:
            return {"message": f"Document {doc_id} deleted successfully", "file_type": "document"}
        
        # If not found as document, try as Excel table
        deleted = data_loader.delete_excel_table(user_id, doc_id)
        if deleted:
            return {"message": f"Excel file {doc_id} deleted successfully", "file_type": "excel"}
        
        # Not found in either store
        raise HTTPException(
            status_code=404,
            detail=f"Document or Excel file with doc_id '{doc_id}' not found for user '{user_id}'"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
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
    """Serve the frontend index.html file."""
    if frontend_dir:
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
    
    # Fallback to API info if frontend not found
    return {
        "message": "LangGraph Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /api/v1/chat - Chat with streaming responses",
            "upload": "POST /api/v1/upload - Upload files (PDF, DOCX, Excel)",
            "metadata_csv": "GET /api/v1/metadata-csv - Get metadata table as CSV (filtered by user_id, session_id)",
            "tables_by_schema_csv": "GET /api/v1/tables-by-schema-csv - Get tables grouped by schema as CSV",
            "threads": "GET /api/v1/threads - List conversation threads",
            "messages": "GET /api/v1/threads/{thread_id}/messages - Get thread messages",
            "delete_thread": "DELETE /api/v1/threads/{thread_id} - Delete thread",
            "documents": "GET /api/v1/documents - List user documents (unified: documents + Excel)",
            "delete_document": "DELETE /api/v1/documents/{doc_id} - Delete document or Excel file by doc_id",
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

