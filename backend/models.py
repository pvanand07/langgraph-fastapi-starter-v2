"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


class ChatInput(BaseModel):
    """Input model for chat endpoint."""
    query: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    user_id: str = Field(..., description="User ID")
    model_id: Optional[str] = Field(None, description="Optional model override")


class ChatChunk(BaseModel):
    """Streaming chat response chunk."""
    type: Literal["chunk", "tool_start", "tool_end", "full_response", "error"]
    content: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    output: Optional[str] = None


class ThreadResponse(BaseModel):
    """Thread metadata response."""
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None


class ThreadListResponse(BaseModel):
    """Paginated thread list response."""
    threads: List[ThreadResponse]
    has_more: bool
    after: Optional[str] = None


class RenameThreadRequest(BaseModel):
    """Request model for renaming a thread."""
    title: str = Field(..., description="New thread title", min_length=1)


class MessageResponse(BaseModel):
    """Message response model."""
    id: str
    thread_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    tool_events: Optional[List[dict]] = None


class MessageListResponse(BaseModel):
    """Paginated message list response."""
    messages: List[MessageResponse]
    has_more: bool
    after: Optional[str] = None


class DocumentResponse(BaseModel):
    """Document metadata response."""
    doc_id: str
    doc_name: str
    user_id: str
    page_count: int
    file_hash: Optional[str] = None
    created_at: str


class DocumentListResponse(BaseModel):
    """Document list response."""
    documents: List[DocumentResponse]


class UploadResponse(BaseModel):
    """File upload response."""
    message: str
    doc_id: str
    page_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    timestamp: str


class ExcelUploadResponse(BaseModel):
    """Excel file upload response."""
    message: str
    table_name: str
    filename: str
    row_count: int
    column_count: int
    columns: List[str]
    metadata: dict


class UnifiedUploadResponse(BaseModel):
    """Unified file upload response for documents (PDF/DOCX) and Excel files."""
    file_type: Literal["document", "excel"]
    message: str
    filename: str
    
    # Document fields (for PDF/DOCX)
    doc_id: Optional[str] = None
    page_count: Optional[int] = None
    
    # Excel fields
    table_name: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: Optional[List[str]] = None
    metadata: Optional[dict] = None


class CSVResponse(BaseModel):
    """CSV data response."""
    csv_data: str
    content_type: str = "text/csv"


class MultiUploadResponse(BaseModel):
    """Response for multiple file uploads."""
    results: List[UnifiedUploadResponse]
    total_files: int
    successful: int
    failed: int
    errors: Optional[List[dict]] = None

