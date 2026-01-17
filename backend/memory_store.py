"""SQLite-based conversation memory using LangGraph's AsyncSqliteSaver."""

import os
import aiosqlite
from typing import Optional
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config import CONVERSATIONS_DB_PATH

# Global memory instance (matching IResearcher-v5 pattern)
memory: Optional[AsyncSqliteSaver] = None


async def initialize_database():
    """Initialize the async SQLite database (matching IResearcher-v5 pattern)."""
    global memory
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(CONVERSATIONS_DB_PATH), exist_ok=True)
        
        conn = await aiosqlite.connect(CONVERSATIONS_DB_PATH)
        
        # Patch connection to add is_alive() method if it doesn't exist
        # This is needed for compatibility with AsyncSqliteSaver
        if not hasattr(conn, 'is_alive'):
            def is_alive():
                return True
            conn.is_alive = is_alive
        
        memory = AsyncSqliteSaver(conn)
        await memory.setup()
        print(f"✅ Memory store initialized at: {CONVERSATIONS_DB_PATH}")
    except Exception as e:
        print(f"❌ Database error: {e}")
        conn = await aiosqlite.connect(":memory:")
        
        # Patch connection for in-memory database too
        if not hasattr(conn, 'is_alive'):
            def is_alive():
                return True
            conn.is_alive = is_alive
        
        memory = AsyncSqliteSaver(conn)
        await memory.setup()
        print("⚠️ Using in-memory database")


def get_memory() -> AsyncSqliteSaver:
    """Get the memory checkpointer instance."""
    if memory is None:
        raise RuntimeError("Memory not initialized. Call initialize_database() first.")
    return memory

