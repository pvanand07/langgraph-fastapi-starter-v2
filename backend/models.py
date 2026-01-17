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


class MessageResponse(BaseModel):
    """Message response model."""
    id: str
    thread_id: str
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: datetime
    tool_name: Optional[str] = None


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


