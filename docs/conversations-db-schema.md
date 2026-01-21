# LangGraph Conversations Database Schema

## Overview

The `conversations.db` database uses LangGraph's `AsyncSqliteSaver` checkpoint system for conversation state persistence. This database stores conversation checkpoints and state writes in a SQLite format.

**Database Location**: `backend/data/chatbot/conversations.db`

## Database Structure

The database contains two main tables:

1. **`checkpoints`** - Stores conversation state checkpoints
2. **`writes`** - Stores incremental state writes/changes

---

## Table: `checkpoints`

Stores conversation state checkpoints that represent snapshots of the conversation state at different points in time.

### Schema

| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| `thread_id` | TEXT | NOT NULL | PK (1) | Conversation/thread identifier (UUID format) |
| `checkpoint_ns` | TEXT | NOT NULL | PK (2) | Checkpoint namespace (usually empty string `''`) |
| `checkpoint_id` | TEXT | NOT NULL | PK (3) | Unique checkpoint identifier (UUID format) |
| `parent_checkpoint_id` | TEXT | NULL | - | Parent checkpoint ID for state lineage tracking |
| `type` | TEXT | NULL | - | Checkpoint serialization type (e.g., `"msgpack"`) |
| `checkpoint` | BLOB | NULL | - | Serialized checkpoint state in msgpack format |
| `metadata` | BLOB | NULL | - | JSON metadata containing execution context |

### Primary Key

Composite primary key: `(thread_id, checkpoint_ns, checkpoint_id)`

### Indexes

- `sqlite_autoindex_checkpoints_1` - Auto-created index on the composite primary key

### Metadata Structure

The `metadata` column contains JSON with the following structure:

```json
{
  "source": "input" | "loop",
  "step": <integer>,
  "parents": {},
  "user_id": "<user_id_string>"
}
```

- **`source`**: Indicates where the checkpoint was created (`"input"` for initial state, `"loop"` for execution steps)
- **`step`**: Execution step number (-1 for input, 0+ for loop steps)
- **`parents`**: Parent checkpoint references (usually empty object)
- **`user_id`**: User identifier associated with the conversation

### Checkpoint Data Structure

The `checkpoint` BLOB contains msgpack-serialized data with the following structure:

- **`v`**: Version number
- **`ts`**: Timestamp (ISO format)
- **`id`**: Checkpoint ID
- **`channel_values`**: State channel values including:
  - `__start__`: Initial state
  - `messages`: Array of conversation messages
  - `context`: Additional context data
  - `branch:to:agent`: Graph execution branches
- **`channel_versions`**: Version tracking for each channel
- **`versions_seen`**: Version tracking information
- **`updated_channels`**: List of channels that were updated

---

## Table: `writes`

Stores incremental state writes/changes for each checkpoint. This table tracks individual channel updates.

### Schema

| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| `thread_id` | TEXT | NOT NULL | PK (1) | Conversation/thread identifier |
| `checkpoint_ns` | TEXT | NOT NULL | PK (2) | Checkpoint namespace |
| `checkpoint_id` | TEXT | NOT NULL | PK (3) | Associated checkpoint ID |
| `task_id` | TEXT | NOT NULL | PK (4) | Task identifier (UUID format) |
| `idx` | INTEGER | NOT NULL | PK (5) | Write index/sequence number |
| `channel` | TEXT | NOT NULL | - | State channel name (e.g., `"messages"`, `"context"`, `"branch:to:agent"`) |
| `type` | TEXT | NULL | - | Write type (e.g., `"msgpack"` or `null`) |
| `value` | BLOB | NULL | - | Serialized channel value in msgpack format |

### Primary Key

Composite primary key: `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`

### Indexes

- `sqlite_autoindex_writes_1` - Auto-created index on the composite primary key

### Common Channels

- **`messages`**: Conversation messages (HumanMessage, AIMessage, etc.)
- **`context`**: Additional context data
- **`branch:to:agent`**: Graph execution branch information

---

## Usage in Codebase

### Initialization

The database is initialized using LangGraph's `AsyncSqliteSaver`:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

conn = await aiosqlite.connect(CONVERSATIONS_DB_PATH)
memory = AsyncSqliteSaver(conn)
await memory.setup()
```

**Location**: `backend/memory_store.py`

### Configuration

**Database Path**: `backend/data/chatbot/conversations.db`

Defined in: `backend/config.py`

```python
CONVERSATIONS_DB_PATH = str(DATA_DIR / "conversations.db")
```

---

## Key Concepts

### Checkpoint System

LangGraph uses a checkpoint system to:
- **Persist state**: Save conversation state at various points
- **Enable recovery**: Restore conversation state from any checkpoint
- **Track lineage**: Maintain parent-child relationships between checkpoints
- **Support branching**: Allow multiple execution paths from a single checkpoint

### State Channels

State is organized into channels:
- Each channel represents a different aspect of the conversation state
- Channels can be updated independently
- The `writes` table tracks changes to individual channels

### Message Serialization

Messages are stored as msgpack-serialized BLOBs:
- **HumanMessage**: User input messages
- **AIMessage**: Assistant response messages
- **ToolMessage**: Tool execution results
- **SystemMessage**: System instructions

### Thread Management

- Each conversation is identified by a unique `thread_id`
- Multiple checkpoints can exist for the same thread (representing different states)
- Checkpoints form a tree structure via `parent_checkpoint_id`

---

## Example Data

### Sample Checkpoint Metadata

```json
{
  "source": "loop",
  "step": 1,
  "parents": {},
  "user_id": "1"
}
```

### Sample Channel Values

- **messages**: Array of LangChain message objects
- **context**: String or object containing additional context
- **branch:to:agent**: Execution branch information

---

## Notes

- The database uses SQLite format 3
- Data is serialized using msgpack format for efficient storage
- The schema is automatically created by LangGraph's `AsyncSqliteSaver.setup()` method
- Checkpoints maintain a tree structure via parent-child relationships
- The `writes` table provides granular tracking of state changes per channel

---

## Related Files

- `backend/memory_store.py` - Database initialization and memory management
- `backend/config.py` - Database path configuration
- `backend/server.py` - Usage of checkpointer in chat server








