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
    type: Literal["chunk", "tool_start", "tool_end", "full_response", "error", "interrupt"]
    content: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    output: Optional[str] = None
    interrupt_data: Optional[dict] = None  # For interrupt events


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


class CSVResponse(BaseModel):
    """CSV data response."""
    csv_data: str
    content_type: str = "text/csv"


class HITLDecision(BaseModel):
    """Decision for a single human-in-the-loop action."""
    type: Literal["approve", "edit", "reject"] = Field(..., description="Decision type")
    edited_action: Optional[dict] = Field(None, description="Edited action (only for 'edit' type)")
    message: Optional[str] = Field(None, description="Rejection message (only for 'reject' type)")


class HITLResumeInput(BaseModel):
    """Input model for resuming after human-in-the-loop interrupt."""
    thread_id: str = Field(..., description="Thread ID to resume")
    user_id: str = Field(..., description="User ID")
    decisions: List[HITLDecision] = Field(..., description="List of decisions for each action")


class HITLInterruptData(BaseModel):
    """Interrupt data sent to frontend when human input is needed."""
    action_requests: List[dict] = Field(..., description="List of actions requiring review")
    review_configs: List[dict] = Field(..., description="Configuration for each action")


