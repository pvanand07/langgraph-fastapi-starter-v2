# LangGraph Chatbot API Documentation

This document provides comprehensive API documentation for the LangGraph Chatbot backend, designed to be easily parsed and understood by AI agents.

## Base Information

- **Base URL**: `http://localhost:8000` (development) or configured via environment variables
- **API Version**: v1
- **API Prefix**: `/api/v1`
- **Content Type**: `application/json` (except where specified)
- **Streaming**: Server-Sent Events (SSE) for chat endpoint

## Authentication

Currently, the API uses `user_id` as a query parameter or form field for user identification. No additional authentication headers are required.

## Endpoints

### 1. Chat Endpoint

**POST** `/api/v1/chat`

Main chat endpoint with Server-Sent Events (SSE) streaming support.

#### Request Body

```json
{
  "query": "string (required) - User message",
  "thread_id": "string (optional) - Conversation thread ID. If not provided, a new thread will be created",
  "user_id": "string (required) - User identifier",
  "model_id": "string (optional) - Model override"
}
```

#### Response

**Content-Type**: `text/event-stream`

The response is a stream of Server-Sent Events (SSE). Each event is a JSON object with one of the following types:

1. **Chunk Event** (type: "chunk")
```json
{
  "type": "chunk",
  "content": "partial text content"
}
```

2. **Tool Start Event** (type: "tool_start")
```json
{
  "type": "tool_start",
  "name": "tool_name",
  "input": { /* tool input parameters */ }
}
```

3. **Tool End Event** (type: "tool_end")
```json
{
  "type": "tool_end",
  "name": "tool_name",
  "output": "tool output result"
}
```

4. **Full Response Event** (type: "full_response")
```json
{
  "type": "full_response",
  "content": "complete assistant response"
}
```

5. **Error Event** (type: "error")
```json
{
  "type": "error",
  "content": "error message"
}
```

#### Behavior

- If `thread_id` is not provided, a new conversation thread is automatically created
- The thread title is auto-generated from the user's query
- User messages are automatically saved to the frontend store
- Assistant responses are saved with associated tool events
- The endpoint builds context from:
  - User's uploaded documents (PDF/DOCX)
  - Available data tables (Excel files)
  - Table schemas and metadata

#### Example Request

```javascript
const response = await fetch('/api/v1/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: "What documents do I have?",
    user_id: "user123",
    thread_id: "thread456" // optional
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.trim()) {
      const event = JSON.parse(line);
      console.log(event);
    }
  }
}
```

#### Error Responses

- **503 Service Unavailable**: Chat server not initialized
- **500 Internal Server Error**: Processing error

---

### 2. File Upload Endpoint

**POST** `/api/v1/upload`

Unified file upload endpoint supporting multiple files of different types.

#### Request

**Content-Type**: `multipart/form-data`

**Form Fields**:
- `user_id` (string, required): User identifier
- `session_id` (string, optional): Session ID (required for Excel files)
- `files` (File[], required): One or more files to upload

#### Supported File Types

1. **Documents**: `.pdf`, `.docx`
   - Extracted and stored as searchable text
   - Duplicate detection based on file hash
   - Returns page count and document ID

2. **Excel Files**: `.xlsx`, `.xls`
   - Loaded into DuckDB database
   - Requires `session_id` parameter
   - Returns table name, row count, column count, and metadata

#### Response

```json
{
  "results": [
    {
      "file_type": "document" | "excel",
      "message": "Success message",
      "filename": "file.pdf",
      "doc_id": "doc_123", // for all docs as well as excel
      "page_count": 10, // for documents only
      "table_name": "table_123", // for excel (also available as doc_id)
      "row_count": 100, // for excel
      "column_count": 5, // for excel
      "columns": ["col1", "col2"], // for excel
      "metadata": {} // for excel
    }
  ],
  "total_files": 2,
  "successful": 1,
  "failed": 1,
  "errors": [
    {
      "filename": "file.pdf",
      "error": "Error message",
      "status_code": 409
    }
  ]
}
```

**Note**: For Excel files, `doc_id` is set to the `table_name` for consistency with the unified documents endpoint.

#### Error Responses

- **400 Bad Request**: No files provided, empty file, unsupported file type, or missing session_id for Excel files
- **409 Conflict**: Duplicate file (same hash already exists)
- **500 Internal Server Error**: File processing error

#### Example Request

```javascript
const formData = new FormData();
formData.append('user_id', 'user123');
formData.append('session_id', 'session456'); // required for Excel
formData.append('files', file1);
formData.append('files', file2);

const response = await fetch('/api/v1/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

---


### 5. List Threads

**GET** `/api/v1/threads`

Lists conversation threads for a user with pagination.

#### Query Parameters

- `user_id` (string, required): User ID
- `limit` (integer, optional, default: 20, min: 1, max: 100): Number of threads to return
- `after` (string, optional): Cursor for pagination (from previous response)
- `order` (string, optional, default: "desc", enum: ["asc", "desc"]): Sort order

#### Response

```json
{
  "threads": [
    {
      "id": "thread_123",
      "user_id": "user123",
      "title": "Thread title",
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00"
    }
  ],
  "has_more": true,
  "after": "cursor_string"
}
```

#### Error Responses

- **500 Internal Server Error**: Error listing threads

#### Example Request

```javascript
const response = await fetch('/api/v1/threads?user_id=user123&limit=10&order=desc');
const result = await response.json();
console.log(result.threads);
```

---

### 6. Get Thread Messages

**GET** `/api/v1/threads/{thread_id}/messages`

Retrieves messages from a conversation thread with pagination.

#### Path Parameters

- `thread_id` (string, required): Thread ID

#### Query Parameters

- `user_id` (string, required): User ID
- `limit` (integer, optional, default: 50, min: 1, max: 100): Number of messages to return
- `after` (string, optional): Cursor for pagination
- `order` (string, optional, default: "desc", enum: ["asc", "desc"]): Sort order

#### Response

```json
{
  "messages": [
    {
      "id": "msg_123",
      "thread_id": "thread_456",
      "role": "user" | "assistant",
      "content": "Message content",
      "created_at": "2024-01-01T00:00:00",
      "tool_events": [
        {
          "type": "tool_start",
          "name": "tool_name",
          "input": {}
        },
        {
          "type": "tool_end",
          "name": "tool_name",
          "output": "result"
        }
      ]
    }
  ],
  "has_more": false,
  "after": null
}
```

#### Error Responses

- **404 Not Found**: Thread not found
- **500 Internal Server Error**: Error retrieving messages

#### Example Request

```javascript
const response = await fetch('/api/v1/threads/thread123/messages?user_id=user123&limit=20');
const result = await response.json();
console.log(result.messages);
```

---

### 7. Rename Thread

**PATCH** `/api/v1/threads/{thread_id}/rename`

Renames a conversation thread by updating its title.

#### Path Parameters

- `thread_id` (string, required): Thread ID

#### Query Parameters

- `user_id` (string, required): User ID

#### Request Body

```json
{
  "title": "string (required, min length: 1) - New thread title"
}
```

#### Response

```json
{
  "id": "thread_123",
  "user_id": "user123",
  "title": "New Thread Title",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

#### Error Responses

- **404 Not Found**: Thread not found for user
- **500 Internal Server Error**: Error renaming thread

#### Example Request

```javascript
const response = await fetch('/api/v1/threads/thread123/rename?user_id=user123', {
  method: 'PATCH',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    title: "New Thread Title"
  })
});
const result = await response.json();
console.log(result);
```

---

### 8. Delete Thread

**DELETE** `/api/v1/threads/{thread_id}`

Deletes a conversation thread.

#### Path Parameters

- `thread_id` (string, required): Thread ID

#### Query Parameters

- `user_id` (string, required): User ID

#### Response

```json
{
  "message": "Thread thread_123 deleted successfully"
}
```

#### Error Responses

- **404 Not Found**: Thread not found
- **500 Internal Server Error**: Error deleting thread

#### Example Request

```javascript
const response = await fetch('/api/v1/threads/thread123?user_id=user123', {
  method: 'DELETE'
});
const result = await response.json();
console.log(result);
```

---

### 9. List Documents (Unified)

**GET** `/api/v1/documents`

Lists all documents (PDF/DOCX) and Excel files for a user in a unified format.

#### Query Parameters

- `user_id` (string, required): User ID
- `session_id` (string, optional): Optional session ID filter for Excel files

#### Response

```json
{
  "items": [
    {
      "file_type": "document",
      "filename": "example.pdf",
      "created_at": "2024-01-10T14:20:30",
      "doc_id": "doc_abc123",
      "content_preview": null,
      "page_count": 10
    },
    {
      "file_type": "excel",
      "filename": "sales_report.xlsx",
      "created_at": "2024-01-15T10:30:45.123456",
      "doc_id": "sales_report_abc12",
      "content_preview": "**Name:** MARWA ENTERPRISES\n**Address:** ALUVA: 683101\n**Report Title:** All Product Purchase Report\n**Client:** MONDELEZ INDIA FOODS LTD",
      "page_count": null
    }
  ],
  "total_count": 2,
  "document_count": 1,
  "excel_count": 1
}
```

#### Response Fields

- `items`: Array of unified document items
  - `file_type`: `"document"` (PDF/DOCX) or `"excel"`
  - `filename`: Original filename
  - `created_at`: Timestamp string
  - `doc_id`: Document ID (for documents) or table name (for Excel files)
  - `content_preview`: Markdown string for Excel metadata (name, address, report_title, client), or `null` for documents
  - `page_count`: Number of pages (documents only), `null` for Excel
- `total_count`: Total number of items
- `document_count`: Number of document files
- `excel_count`: Number of Excel files

**Note**: Items are sorted by `created_at` (most recent first).

#### Error Responses

- **500 Internal Server Error**: Error listing documents

#### Example Request

```javascript
const response = await fetch('/api/v1/documents?user_id=user123');
const result = await response.json();
console.log(result.items);
```

---

### 10. Delete Document

**DELETE** `/api/v1/documents/{doc_id}`

Deletes a document (PDF/DOCX) or Excel file by doc_id.

#### Path Parameters

- `doc_id` (string, required): Document ID (for documents) or table name (for Excel files)

#### Query Parameters

- `user_id` (string, required): User ID

#### Response

```json
{
  "message": "Document doc_123 deleted successfully",
  "file_type": "document"
}
```

or for Excel files:

```json
{
  "message": "Excel file sales_report_abc12 deleted successfully",
  "file_type": "excel"
}
```

#### Behavior

- The endpoint automatically detects whether the `doc_id` is a document or an Excel table
- Tries to delete from document store first, then from Excel tables
- Returns 404 if not found in either store

#### Error Responses

- **404 Not Found**: Document or Excel file not found for user
- **500 Internal Server Error**: Error deleting document

#### Example Request

```javascript
// Delete a document
const response = await fetch('/api/v1/documents/doc_123?user_id=user123', {
  method: 'DELETE'
});
const result = await response.json();
console.log(result);

// Delete an Excel file
const response = await fetch('/api/v1/documents/sales_report_abc12?user_id=user123', {
  method: 'DELETE'
});
const result = await response.json();
console.log(result);
```

---

## Data Models

### ChatInput
```typescript
{
  query: string;           // Required: User message
  thread_id?: string;      // Optional: Conversation thread ID
  user_id: string;         // Required: User identifier
  model_id?: string;       // Optional: Model override
}
```

### ThreadResponse
```typescript
{
  id: string;
  user_id: string;
  title?: string;
  created_at: string;      // ISO 8601 datetime
  updated_at: string;      // ISO 8601 datetime
}
```

### MessageResponse
```typescript
{
  id: string;
  thread_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;      // ISO 8601 datetime
  tool_events?: Array<{
    type: "tool_start" | "tool_end";
    name: string;
    input?: object;
    output?: string;
  }>;
}
```

### DocumentResponse
```typescript
{
  doc_id: string;
  doc_name: string;
  user_id: string;
  page_count: number;
  file_hash?: string;
  created_at: string;      // ISO 8601 datetime
}
```

### UnifiedUploadResponse
```typescript
{
  file_type: "document" | "excel";
  message: string;
  filename: string;
  // Document fields
  doc_id?: string;  // Document ID for documents
  page_count?: number;  // For documents only
  // Excel fields
  doc_id?: string;  // Table name (same as table_name) for Excel files
  table_name?: string;  // Table name (also available as doc_id)
  row_count?: number;
  column_count?: number;
  columns?: string[];
  metadata?: object;
}
```

**Note**: For Excel files, `doc_id` is set to the `table_name` for consistency with the unified documents endpoint.

### UnifiedDocumentItem
```typescript
{
  file_type: "document" | "excel";
  filename: string;
  created_at: string;  // ISO 8601 datetime
  doc_id: string;  // Document ID or table name
  content_preview?: string;  // Markdown string for Excel metadata, null for documents
  page_count?: number;  // For documents only, null for Excel
}
```

### UnifiedDocumentListResponse
```typescript
{
  items: UnifiedDocumentItem[];
  total_count: number;
  document_count: number;
  excel_count: number;
}
```

### RenameThreadRequest
```typescript
{
  title: string;  // Required: New thread title (min length: 1)
}
```

---

## Error Handling

All endpoints follow consistent error handling:

1. **400 Bad Request**: Invalid request parameters or missing required fields
2. **404 Not Found**: Resource not found (thread, document, etc.)
3. **409 Conflict**: Duplicate resource (e.g., duplicate file upload)
4. **500 Internal Server Error**: Server-side processing error
5. **503 Service Unavailable**: Service not initialized or unavailable

Error response format:
```json
{
  "detail": "Error message description"
}
```

---

## Context Building

The chat endpoint automatically builds context from multiple sources:

1. **Document Context**: Extracts relevant text from user's uploaded PDF/DOCX documents
2. **Data Table Metadata**: Includes metadata CSV from uploaded Excel files
3. **Table Schemas**: Includes schema information (column names and types) for all tables

This context is automatically included in the chat request to provide the AI with relevant information about the user's data.

---

## Streaming Protocol

The chat endpoint uses Server-Sent Events (SSE) for streaming responses:

1. **Connection**: Establish a connection to `/api/v1/chat` with POST request
2. **Streaming**: Read events as they arrive
3. **Event Format**: Each line is a JSON object (except empty lines)
4. **Event Types**: `chunk`, `tool_start`, `tool_end`, `full_response`, `error`
5. **Completion**: Stream ends when connection closes or error occurs

---

## Best Practices

1. **User ID**: Always provide a consistent `user_id` across requests for the same user
2. **Thread Management**: Use `thread_id` to maintain conversation context
3. **File Uploads**: 
   - Use `session_id` for Excel files to group related data
   - Check for duplicate files before uploading (409 error indicates duplicate)
4. **Pagination**: Use `after` cursor from responses for paginated requests
5. **Error Handling**: Always check response status and handle errors appropriately
6. **Streaming**: Properly handle SSE streams and parse JSON events

---

## CORS

The API is configured with CORS middleware allowing:
- **Origins**: All origins (`*`)
- **Credentials**: Allowed
- **Methods**: All methods
- **Headers**: All headers

---

## Static Files

Frontend static files are served from:
- Docker: `/app/frontend`
- Local: `../frontend` (relative to backend directory)

Static files are mounted at `/static` endpoint.

---

## Notes for AI Agents

1. **Thread Creation**: Threads are automatically created when `thread_id` is omitted from chat requests
2. **Context Integration**: The chat endpoint automatically integrates document and data table context
3. **Tool Events**: Tool usage is tracked and included in message responses
4. **File Deduplication**: Files are deduplicated by hash to prevent duplicate storage
5. **Session Grouping**: Excel files should use the same `session_id` to group related data tables
6. **Pagination**: All list endpoints support cursor-based pagination using the `after` parameter
7. **Unified Documents**: The `/api/v1/documents` endpoint returns both documents and Excel files in a unified format, sorted by creation date (most recent first)
8. **Document IDs**: Excel files use their `table_name` as `doc_id` for consistency across endpoints
9. **Unified Delete**: The delete endpoint automatically detects whether a `doc_id` is a document or Excel table and deletes accordingly
10. **Content Preview**: Excel files include a markdown-formatted `content_preview` with metadata (name, address, report_title, client) when available

