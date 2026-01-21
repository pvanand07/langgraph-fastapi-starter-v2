"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict
from datetime import datetime


class ResumeData(BaseModel):
    """Resume data for human-in-the-loop responses."""
    answers: Dict[str, List[str]] = Field(..., description="Maps question_id to list of selected options")


class ChatInput(BaseModel):
    """Input model for chat endpoint."""
    query: Optional[str] = Field(None, description="User message")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    user_id: str = Field(..., description="User ID")
    model_id: Optional[str] = Field(None, description="Optional model override")
    resume_data: Optional[ResumeData] = Field(None, description="Resume data for human-in-the-loop")


class QuestionData(BaseModel):
    """Individual question data for human-in-the-loop."""
    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of available options")


class ChatChunk(BaseModel):
    """Streaming chat response chunk."""
    type: Literal["chunk", "tool_start", "tool_end", "full_response", "error", "questions_pending"]
    content: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    output: Optional[str] = None
    questions: Optional[List[QuestionData]] = Field(None, description="Questions for human-in-the-loop")
    thread_id: Optional[str] = Field(None, description="Thread ID for resuming")


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

