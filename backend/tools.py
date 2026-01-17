"""Minimal tool definitions for the chatbot agent."""

import operator
from datetime import datetime
from typing import Optional, List
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import document_store


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "10 * 5", "100 / 4")
    
    Returns:
        The result of the calculation as a string.
    """
    try:
        # Use operator module for safe evaluation
        # Only allow basic math operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /) are allowed."
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def get_current_time(timezone: Optional[str] = None) -> str:  # noqa: ARG001
    """
    Get the current date and time.
    
    Args:
        timezone: Optional timezone (not implemented, uses UTC)
    
    Returns:
        Current date and time as a formatted string.
    """
    now = datetime.utcnow()
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def search_documents(query: str, user_id: str, doc_ids: Optional[List[str]] = None, limit: int = 5, config: Optional[RunnableConfig] = None) -> str:  # noqa: ARG001
    """
    Search through uploaded documents for the user.
    
    Args:
        query: Search query string
        user_id: User ID to scope the search
        doc_ids: Optional list of document IDs to search within
        limit: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted string with search results including document ID, page number, and content snippet.
    """
    try:
        results = document_store.search_documents(
            user_id=user_id,
            query=query,
            doc_ids=doc_ids,
            limit=limit
        )
        
        if not results:
            return f"No documents found matching '{query}'."
        
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"Document: {result['doc_id']}, Page {result['page_number']}\n"
                f"Snippet: {result.get('snippet', result['content'][:200])}\n"
            )
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error searching documents: {str(e)}"


def get_tools() -> List:
    """Get all available tools."""
    return [calculator, get_current_time, search_documents]

