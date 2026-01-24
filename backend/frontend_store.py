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


def generate_title_from_query(query: str, max_length: int = 60) -> str:
    """Generate a title from user query by truncating to max_length."""
    if not query:
        return "New Conversation"
    
    # Strip whitespace and newlines
    title = query.strip().replace("\n", " ").replace("\r", " ")
    
    # Remove extra spaces
    title = " ".join(title.split())
    
    # Truncate if needed
    if len(title) > max_length:
        title = title[:max_length].rstrip() + "..."
    
    return title


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
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_events TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create index on messages
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread_created 
            ON messages(thread_id, created_at ASC)
        """)
        
        # Create index on messages for user_id
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_thread 
            ON messages(user_id, thread_id, created_at ASC)
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


async def update_thread(thread_id: str, user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
    """Update thread metadata with user verification."""
    conn = await get_db_connection()
    now = datetime.utcnow().isoformat()
    
    # Verify thread belongs to user
    cursor = await conn.execute(
        "SELECT id FROM threads WHERE id = ? AND user_id = ?",
        (thread_id, user_id)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Thread {thread_id} not found for user {user_id}")
    
    # Update thread title
    await conn.execute("""
        UPDATE threads 
        SET title = ?,
            updated_at = ?
        WHERE id = ? AND user_id = ?
    """, (title, now, thread_id, user_id))
    
    await conn.commit()
    
    # Get updated thread
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
    """List threads with pagination, derived from messages table."""
    conn = await get_db_connection()
    
    # Build query
    order_clause = "DESC" if order == "desc" else "ASC"
    
    if after:
        # Get the latest updated_at from messages for the cursor thread
        cursor = await conn.execute("""
            SELECT MAX(updated_at) 
            FROM messages 
            WHERE thread_id = ? AND thread_id IN (
                SELECT id FROM threads WHERE user_id = ?
            )
        """, (after, user_id))
        after_row = await cursor.fetchone()
        
        if after_row and after_row[0]:
            after_timestamp = after_row[0]
            if order == "desc":
                query = """
                    SELECT 
                        m.thread_id as id,
                        t.user_id,
                        t.title,
                        MIN(m.created_at) as created_at,
                        MAX(m.updated_at) as updated_at
                    FROM messages m
                    INNER JOIN threads t ON m.thread_id = t.id
                    WHERE t.user_id = ?
                    GROUP BY m.thread_id, t.user_id, t.title
                    HAVING MAX(m.updated_at) < ?
                    ORDER BY MAX(m.updated_at) DESC, m.thread_id DESC
                    LIMIT ?
                """
                cursor = await conn.execute(query, (user_id, after_timestamp, limit + 1))
            else:
                query = """
                    SELECT 
                        m.thread_id as id,
                        t.user_id,
                        t.title,
                        MIN(m.created_at) as created_at,
                        MAX(m.updated_at) as updated_at
                    FROM messages m
                    INNER JOIN threads t ON m.thread_id = t.id
                    WHERE t.user_id = ?
                    GROUP BY m.thread_id, t.user_id, t.title
                    HAVING MAX(m.updated_at) > ?
                    ORDER BY MAX(m.updated_at) ASC, m.thread_id ASC
                    LIMIT ?
                """
                cursor = await conn.execute(query, (user_id, after_timestamp, limit + 1))
        else:
            # Invalid cursor, start from beginning
            query = f"""
                SELECT 
                    m.thread_id as id,
                    t.user_id,
                    t.title,
                    MIN(m.created_at) as created_at,
                    MAX(m.updated_at) as updated_at
                FROM messages m
                INNER JOIN threads t ON m.thread_id = t.id
                WHERE t.user_id = ?
                GROUP BY m.thread_id, t.user_id, t.title
                ORDER BY MAX(m.updated_at) {order_clause}, m.thread_id {order_clause}
                LIMIT ?
            """
            cursor = await conn.execute(query, (user_id, limit + 1))
    else:
        # Query threads from messages, joining with threads table for user_id and title
        query = f"""
            SELECT 
                m.thread_id as id,
                t.user_id,
                t.title,
                MIN(m.created_at) as created_at,
                MAX(m.updated_at) as updated_at
            FROM messages m
            INNER JOIN threads t ON m.thread_id = t.id
            WHERE t.user_id = ?
            GROUP BY m.thread_id, t.user_id, t.title
            ORDER BY MAX(m.updated_at) {order_clause}, m.thread_id {order_clause}
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
    role: str,
    content: str,
    user_id: str,
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
                user_id = COALESCE(?, user_id),
                updated_at = ?
            WHERE id = ?
        """, (content if content else None, tool_events_json, user_id, now, message_id))
        created_at = existing[1]
    else:
        # Insert new message
        created_at = now
        await conn.execute("""
            INSERT INTO messages (
                id, thread_id, user_id, role, content,
                tool_events, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id, thread_id, user_id, role, content,
            tool_events_json, created_at, now
        ))
    
    # Check if thread exists and get its title
    cursor = await conn.execute(
        "SELECT id, title FROM threads WHERE id = ?",
        (thread_id,)
    )
    thread_row = await cursor.fetchone()
    
    if not thread_row:
        # Create thread if it doesn't exist
        # Generate title from content if it's a user message, otherwise use None
        title = generate_title_from_query(content) if role == "user" else None
        await conn.execute("""
            INSERT INTO threads (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (thread_id, user_id, title, now, now))
    else:
        # Thread exists - update updated_at and set title if missing and this is a user message
        existing_title = thread_row[1]
        if role == "user" and not existing_title:
            # Update title from user message
            title = generate_title_from_query(content)
            await conn.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, thread_id)
            )
        else:
            # Just update updated_at
            await conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (now, thread_id)
            )
    
    await conn.commit()
    
    return {
        "id": message_id,
        "thread_id": thread_id,
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
    
    # Build query
    order_clause = "DESC" if order == "desc" else "ASC"
    
    if after:
        # Get the created_at of the cursor message
        cursor = await conn.execute(
            "SELECT created_at FROM messages WHERE id = ? AND thread_id = ? AND user_id = ?",
            (after, thread_id, user_id)
        )
        after_row = await cursor.fetchone()
        
        if after_row:
            after_timestamp = after_row[0]
            if order == "desc":
                query = """
                    SELECT id, thread_id, role, content,
                           tool_events, created_at, updated_at
                    FROM messages 
                    WHERE thread_id = ? AND user_id = ? AND created_at < ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """
            else:
                query = """
                    SELECT id, thread_id, role, content,
                           tool_events, created_at, updated_at
                    FROM messages 
                    WHERE thread_id = ? AND user_id = ? AND created_at > ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """
            cursor = await conn.execute(query, (thread_id, user_id, after_timestamp, limit + 1))
        else:
            # Invalid cursor, start from beginning
            query = f"""
                SELECT id, thread_id, role, content,
                       tool_events, created_at, updated_at
                FROM messages 
                WHERE thread_id = ? AND user_id = ?
                ORDER BY created_at {order_clause}
                LIMIT ?
            """
            cursor = await conn.execute(query, (thread_id, user_id, limit + 1))
    else:
        query = f"""
            SELECT id, thread_id, role, content,
                   tool_events, created_at, updated_at
            FROM messages 
            WHERE thread_id = ? AND user_id = ?
            ORDER BY created_at {order_clause}
            LIMIT ?
        """
        cursor = await conn.execute(query, (thread_id, user_id, limit + 1))
    
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
            "role": row[2],
            "content": row[3],
            "created_at": row[5],
            "updated_at": row[6] if len(row) > 6 else row[5]
        }
        
        # Parse tool_events if present
        if row[4]:  # tool_events
            try:
                msg["tool_events"] = json.loads(row[4])
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


async def delete_message(message_id: str, thread_id: str, user_id: str) -> None:
    """Delete specific message."""
    conn = await get_db_connection()
    
    # Verify message belongs to thread and user
    cursor = await conn.execute(
        "SELECT id FROM messages WHERE id = ? AND thread_id = ? AND user_id = ?",
        (message_id, thread_id, user_id)
    )
    row = await cursor.fetchone()
    
    if not row:
        raise NotFoundError(f"Message {message_id} not found in thread {thread_id} for user {user_id}")
    
    # Delete message
    await conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    
    await conn.commit()

