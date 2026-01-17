"""Frontend SQLite store for threads and messages."""

import os
import json
import uuid
import aiosqlite
from typing import Optional, Dict, Any
from datetime import datetime
from config import FRONTEND_STORE_DB_PATH

# Global database connection
_db_conn: Optional[aiosqlite.Connection] = None


class NotFoundError(Exception):
    """Exception raised when a resource is not found."""


async def initialize_database():
    """Initialize the async SQLite database and create tables."""
    global _db_conn
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(FRONTEND_STORE_DB_PATH), exist_ok=True)
        
        conn = await aiosqlite.connect(FRONTEND_STORE_DB_PATH)
        _db_conn = conn
        
        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Create threads table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create index on threads
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_user_updated 
            ON threads(user_id, updated_at DESC)
        """)
        
        # Create messages table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                role TEXT,
                content TEXT NOT NULL,
                tool_events TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(id)
            )
        """)
        
        # Create index on messages
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread_created 
            ON messages(thread_id, created_at ASC)
        """)
        
        await conn.commit()
        print(f"✅ Frontend store initialized at: {FRONTEND_STORE_DB_PATH}")
    except Exception as e:
        print(f"❌ Frontend store database error: {e}")
        raise


async def get_db_connection() -> aiosqlite.Connection:
    """Get the database connection."""
    if _db_conn is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _db_conn


# User Management

async def create_or_update_user(
    user_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None
) -> Dict[str, Any]:
    """Create or update a user."""
    conn = await get_db_connection()
    now = datetime.utcnow().isoformat()
    
    # Check if user exists
    cursor = await conn.execute(
        "SELECT id, created_at FROM users WHERE id = ?",
        (user_id,)
    )
    existing = await cursor.fetchone()
    
    if existing:
        # Update existing user
        await conn.execute("""
            UPDATE users 
            SET name = COALESCE(?, name),
                email = COALESCE(?, email),
                updated_at = ?
            WHERE id = ?
        """, (name, email, now, user_id))
        created_at = existing[1]
    else:
        # Create new user
        created_at = now
        await conn.execute("""
            INSERT INTO users (id, name, email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, name, email, created_at, now))
    
    await conn.commit()
    
    return {
        "id": user_id,
        "name": name,
        "email": email,
        "created_at": created_at,
        "updated_at": now
    }


async def get_user(user_id: str) -> Dict[str, Any]:
    """Get user by ID."""
    conn = await get_db_connection()
    
    cursor = await conn.execute(
        "SELECT id, name, email, created_at, updated_at FROM users WHERE id = ?",
        (user_id,)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"User {user_id} not found")
    
    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "created_at": row[3],
        "updated_at": row[4]
    }


# Thread Management

async def create_thread(
    thread_id: str,
    user_id: str,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new thread."""
    conn = await get_db_connection()
    now = datetime.utcnow().isoformat()
    
    await conn.execute("""
        INSERT INTO threads (id, user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (thread_id, user_id, title, now, now))
    
    await conn.commit()
    
    return {
        "id": thread_id,
        "user_id": user_id,
        "title": title,
        "created_at": now,
        "updated_at": now
    }


async def update_thread(thread_id: str, title: Optional[str] = None) -> Dict[str, Any]:
    """Update thread metadata."""
    conn = await get_db_connection()
    now = datetime.utcnow().isoformat()
    
    await conn.execute("""
        UPDATE threads 
        SET title = COALESCE(?, title),
            updated_at = ?
        WHERE id = ?
    """, (title, now, thread_id))
    
    await conn.commit()
    
    # Get updated thread
    cursor = await conn.execute(
        "SELECT id, user_id, title, created_at, updated_at FROM threads WHERE id = ?",
        (thread_id,)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Thread {thread_id} not found")
    
    return {
        "id": row[0],
        "user_id": row[1],
        "title": row[2],
        "created_at": row[3],
        "updated_at": row[4]
    }


async def get_thread(thread_id: str, user_id: str) -> Dict[str, Any]:
    """Get thread by ID with user verification."""
    conn = await get_db_connection()
    
    cursor = await conn.execute(
        "SELECT id, user_id, title, created_at, updated_at FROM threads WHERE id = ? AND user_id = ?",
        (thread_id, user_id)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Thread {thread_id} not found for user {user_id}")
    
    return {
        "id": row[0],
        "user_id": row[1],
        "title": row[2],
        "created_at": row[3],
        "updated_at": row[4]
    }


async def list_threads(
    user_id: str,
    limit: int = 20,
    after: Optional[str] = None,
    order: str = "desc"
) -> Dict[str, Any]:
    """List threads with pagination."""
    conn = await get_db_connection()
    
    # Build query
    order_clause = "DESC" if order == "desc" else "ASC"
    
    if after:
        # Get the updated_at of the cursor thread
        cursor = await conn.execute(
            "SELECT updated_at FROM threads WHERE id = ? AND user_id = ?",
            (after, user_id)
        )
        after_row = await cursor.fetchone()
        
        if after_row:
            after_timestamp = after_row[0]
            if order == "desc":
                query = """
                    SELECT id, user_id, title, created_at, updated_at 
                    FROM threads 
                    WHERE user_id = ? AND updated_at < ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """
            else:
                query = """
                    SELECT id, user_id, title, created_at, updated_at 
                    FROM threads 
                    WHERE user_id = ? AND updated_at > ?
                    ORDER BY updated_at ASC
                    LIMIT ?
                """
            cursor = await conn.execute(query, (user_id, after_timestamp, limit + 1))
        else:
            # Invalid cursor, start from beginning
            query = f"""
                SELECT id, user_id, title, created_at, updated_at 
                FROM threads 
                WHERE user_id = ?
                ORDER BY updated_at {order_clause}
                LIMIT ?
            """
            cursor = await conn.execute(query, (user_id, limit + 1))
    else:
        query = f"""
            SELECT id, user_id, title, created_at, updated_at 
            FROM threads 
            WHERE user_id = ?
            ORDER BY updated_at {order_clause}
            LIMIT ?
        """
        cursor = await conn.execute(query, (user_id, limit + 1))
    
    rows = await cursor.fetchall()
    
    # Check if there are more results
    has_more = len(rows) > limit
    threads = rows[:limit]
    
    # Build thread list
    thread_list = [
        {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "created_at": row[3],
            "updated_at": row[4]
        }
        for row in threads
    ]
    
    # Get next cursor
    next_after = thread_list[-1]["id"] if has_more and thread_list else None
    
    return {
        "threads": thread_list,
        "has_more": has_more,
        "after": next_after
    }


async def delete_thread(thread_id: str, user_id: str) -> None:
    """Delete thread and all its messages."""
    conn = await get_db_connection()
    
    # Verify thread belongs to user
    cursor = await conn.execute(
        "SELECT id FROM threads WHERE id = ? AND user_id = ?",
        (thread_id, user_id)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Thread {thread_id} not found for user {user_id}")
    
    # Delete messages first
    await conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    
    # Delete thread
    await conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
    
    await conn.commit()


# Message Management

async def add_message(
    thread_id: str,
    message_type: str,
    content: str,
    role: Optional[str] = None,
    tool_events: Optional[list] = None,
    message_id: Optional[str] = None
) -> Dict[str, Any]:
    """Add or update message to thread."""
    conn = await get_db_connection()
    now = datetime.utcnow().isoformat()
    
    if message_id is None:
        message_id = str(uuid.uuid4())
    
    # Serialize tool_events to JSON if provided
    tool_events_json = json.dumps(tool_events) if tool_events else None
    
    # Check if message exists
    cursor = await conn.execute(
        "SELECT id, created_at FROM messages WHERE id = ?",
        (message_id,)
    )
    existing = await cursor.fetchone()
    
    if existing:
        # Update existing message
        await conn.execute("""
            UPDATE messages 
            SET content = COALESCE(?, content),
                tool_events = COALESCE(?, tool_events),
                updated_at = ?
            WHERE id = ?
        """, (content if content else None, tool_events_json, now, message_id))
        created_at = existing[1]
    else:
        # Insert new message
        created_at = now
        await conn.execute("""
            INSERT INTO messages (
                id, thread_id, message_type, role, content,
                tool_events, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id, thread_id, message_type, role, content,
            tool_events_json, created_at, now
        ))
    
    # Update thread's updated_at
    await conn.execute(
        "UPDATE threads SET updated_at = ? WHERE id = ?",
        (now, thread_id)
    )
    
    await conn.commit()
    
    return {
        "id": message_id,
        "thread_id": thread_id,
        "message_type": message_type,
        "role": role,
        "content": content,
        "tool_events": tool_events,
        "created_at": created_at,
        "updated_at": now
    }


async def get_messages(
    thread_id: str,
    user_id: str,
    limit: int = 50,
    after: Optional[str] = None,
    order: str = "desc"
) -> Dict[str, Any]:
    """Get messages with pagination."""
    conn = await get_db_connection()
    
    # Verify thread belongs to user
    cursor = await conn.execute(
        "SELECT id FROM threads WHERE id = ? AND user_id = ?",
        (thread_id, user_id)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Thread {thread_id} not found for user {user_id}")
    
    # Build query
    order_clause = "DESC" if order == "desc" else "ASC"
    
    if after:
        # Get the created_at of the cursor message
        cursor = await conn.execute(
            "SELECT created_at FROM messages WHERE id = ? AND thread_id = ?",
            (after, thread_id)
        )
        after_row = await cursor.fetchone()
        
        if after_row:
            after_timestamp = after_row[0]
            if order == "desc":
                query = """
                    SELECT id, thread_id, message_type, role, content,
                           tool_events, created_at, updated_at
                    FROM messages 
                    WHERE thread_id = ? AND created_at < ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """
            else:
                query = """
                    SELECT id, thread_id, message_type, role, content,
                           tool_events, created_at, updated_at
                    FROM messages 
                    WHERE thread_id = ? AND created_at > ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """
            cursor = await conn.execute(query, (thread_id, after_timestamp, limit + 1))
        else:
            # Invalid cursor, start from beginning
            query = f"""
                SELECT id, thread_id, message_type, role, content,
                       tool_events, created_at, updated_at
                FROM messages 
                WHERE thread_id = ?
                ORDER BY created_at {order_clause}
                LIMIT ?
            """
            cursor = await conn.execute(query, (thread_id, limit + 1))
    else:
        query = f"""
            SELECT id, thread_id, message_type, role, content,
                   tool_events, created_at, updated_at
            FROM messages 
            WHERE thread_id = ?
            ORDER BY created_at {order_clause}
            LIMIT ?
        """
        cursor = await conn.execute(query, (thread_id, limit + 1))
    
    rows = await cursor.fetchall()
    
    # Check if there are more results
    has_more = len(rows) > limit
    messages = rows[:limit]
    
    # Build message list
    message_list = []
    for row in messages:
        msg = {
            "id": row[0],
            "thread_id": row[1],
            "message_type": row[2],
            "role": row[3],
            "content": row[4],
            "created_at": row[6],
            "updated_at": row[7] if len(row) > 7 else row[6]
        }
        
        # Parse tool_events if present
        if row[5]:  # tool_events
            try:
                msg["tool_events"] = json.loads(row[5])
            except json.JSONDecodeError:
                msg["tool_events"] = None
        else:
            msg["tool_events"] = None
        
        message_list.append(msg)
    
    # Get next cursor
    next_after = message_list[-1]["id"] if has_more and message_list else None
    
    return {
        "messages": message_list,
        "has_more": has_more,
        "after": next_after
    }


async def delete_message(message_id: str, thread_id: str) -> None:
    """Delete specific message."""
    conn = await get_db_connection()
    
    # Verify message belongs to thread
    cursor = await conn.execute(
        "SELECT id FROM messages WHERE id = ? AND thread_id = ?",
        (message_id, thread_id)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Message {message_id} not found in thread {thread_id}")
    
    # Delete message
    await conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    
    await conn.commit()

