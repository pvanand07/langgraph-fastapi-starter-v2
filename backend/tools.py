"""Minimal tool definitions for the chatbot agent."""

import logging
import re
import uuid
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from langchain.tools import tool, ToolRuntime
import pandas as pd
import plotly.express as px
import document_store
import data_loader

logger = logging.getLogger(__name__)


@dataclass
class ChatContext:
    """Context passed to tools containing user and session information."""
    user_id: str
    thread_id: str  # session_id is same as thread_id


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


@tool(response_format="content_and_artifact")
async def search_documents(
    query: str,
    runtime: ToolRuntime[ChatContext],
    doc_ids: Optional[List[str]] = None,
    limit: int = 5
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Search through uploaded documents for the user.
    
    Args:
        query: Search query string
        runtime: ToolRuntime containing user context
        doc_ids: Optional list of document IDs to search within
        limit: Maximum number of results to return (default: 5)
    
    Returns:
        Tuple of (content_string, artifacts_dict) with search results
    """
    writer = runtime.stream_writer
    user_id = runtime.context.user_id
    session_id = runtime.context.thread_id
    
    writer(f"Searching documents for: {query[:100]}...")
    
    try:
        writer(f"Searching documents for user: {user_id}...")
        
        results = document_store.search_documents(
            user_id=user_id,
            query=query,
            doc_ids=doc_ids,
            limit=limit
        )
        
        if not results:
            writer(f"No documents found matching '{query}'.")
            return (f"No documents found matching '{query}'.", None)
        
        formatted_results = []
        artifacts = {
            "results": [],
            "user_id": user_id,
            "session_id": session_id,
            "query": query
        }
        
        for result in results:
            formatted_results.append(
                f"Document: {result['doc_id']}, Page {result['page_number']}\n"
                f"Snippet: {result.get('snippet', result['content'][:200])}\n"
            )
            
            artifacts["results"].append({
                "doc_id": result['doc_id'],
                "page_number": result['page_number'],
                "snippet": result.get('snippet', result['content'][:200])
            })
        
        content = "\n---\n".join(formatted_results)
        
        writer(f"Found {len(artifacts['results'])} relevant document(s).")
        
        return (content, artifacts)
    except Exception as e:
        error_msg = f"Error searching documents: {str(e)}"
        logger.error(error_msg)
        writer(error_msg)
        return (error_msg, None)


@tool(response_format="content_and_artifact")
async def query_duckdb(
    sql_query: str,
    runtime: ToolRuntime[ChatContext],
    limit: int = 100
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Execute a SQL query against DuckDB data tables.
    
    This tool allows you to query the user's data tables stored in DuckDB. The query will automatically
    be scoped to tables accessible by the user_id (and optionally session_id from thread_id).
    
    Args:
        sql_query: SQL query string to execute (e.g., "SELECT * FROM table_name LIMIT 10")
        runtime: ToolRuntime containing user context
        limit: Maximum number of rows to return (default: 100, applied as safety limit)
    
    Returns:
        Tuple of (content_string, artifacts_dict) with query results.
        Results are returned as CSV format for easy reading.
    
    Examples:
        - "SELECT * FROM metadata WHERE user_id = 'user123'"
        - "SELECT table_name, row_count FROM metadata LIMIT 5"
        - "SELECT column_name, column_type FROM information_schema.columns WHERE table_name = 'my_table'"
    """
    writer = runtime.stream_writer
    user_id = runtime.context.user_id
    session_id = runtime.context.thread_id
    
    writer(f"Executing SQL query: {sql_query[:100]}...")
    
    try:
        conn = data_loader.get_connection()
        
        # Get list of tables accessible by this user
        metadata_query = "SELECT table_name FROM metadata WHERE user_id = ?"
        metadata_params = [user_id]
        
        if session_id:
            metadata_query += " AND session_id = ?"
            metadata_params.append(session_id)
        
        try:
            accessible_tables_df = conn.execute(metadata_query, metadata_params).fetchdf()
            accessible_tables = set(accessible_tables_df['table_name'].tolist()) if not accessible_tables_df.empty else set()
            # Also include the metadata table
            accessible_tables.add('metadata')
            
            # Get accessible views for this user/session
            accessible_views = data_loader.get_accessible_views(user_id, session_id)
            accessible_tables.update(accessible_views)
        except Exception:
            accessible_tables = {'metadata'}  # Fallback to at least metadata table
        
        # Basic SQL injection prevention - check for dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                error_msg = f"Error: SQL query contains forbidden keyword '{keyword}'. Only SELECT queries are allowed."
                writer(error_msg)
                return (error_msg, None)
        
        # Ensure it's a SELECT query
        if not sql_query.strip().upper().startswith('SELECT'):
            error_msg = "Error: Only SELECT queries are allowed."
            writer(error_msg)
            return (error_msg, None)
        
        writer("Executing query...")
        
        # Execute the query
        try:
            result_df = conn.execute(sql_query).fetchdf()
        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            logger.error(error_msg)
            writer(error_msg)
            return (error_msg, None)
        
        if result_df.empty:
            msg = "Query executed successfully but returned no results."
            writer(msg)
            return (msg, None)
        
        # Apply limit if result is too large
        truncated = False
        if len(result_df) > limit:
            result_df = result_df.head(limit)
            truncated = True
        
        # Convert to CSV string
        csv_result = result_df.to_csv(index=False)
        
        # Format response
        response = f"Query executed successfully. Returned {len(result_df)} row(s).\n\n"
        if truncated:
            response += f"Note: Results truncated to {limit} rows.\n\n"
        response += "Results (CSV format):\n"
        response += csv_result
        
        # Create artifacts
        artifacts = {
            "query": sql_query,
            "user_id": user_id,
            "session_id": session_id,
            "row_count": len(result_df),
            "truncated": truncated,
            "accessible_tables": list(accessible_tables)
        }
        
        writer(f"Query completed. Returned {len(result_df)} row(s).")
        
        return (response, artifacts)
    except Exception as e:
        error_msg = f"Error querying DuckDB: {str(e)}"
        logger.error(error_msg)
        writer(error_msg)
        return (error_msg, None)


@tool(response_format="content_and_artifact")
async def create_view(
    view_name: str,
    view_definition: str,
    runtime: ToolRuntime[ChatContext]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Create a persistent view in DuckDB scoped to the current user and session.
    
    Views are stored in the database and can be queried like tables. They are automatically
    scoped to the user_id and session_id, so users can only see and use their own views.
    
    Args:
        view_name: Name for the view (alphanumeric and underscores only)
        view_definition: SQL SELECT statement defining the view (e.g., "SELECT * FROM table_name WHERE column = 'value'")
        runtime: ToolRuntime containing user context
    
    Returns:
        Tuple of (content_string, artifacts_dict) with creation result
    
    Examples:
        - view_name: "my_filtered_data", view_definition: "SELECT * FROM sales_data WHERE amount > 1000"
        - view_name: "monthly_summary", view_definition: "SELECT month, SUM(revenue) FROM transactions GROUP BY month"
    """
    writer = runtime.stream_writer
    user_id = runtime.context.user_id
    session_id = runtime.context.thread_id
    
    writer(f"Creating view '{view_name}'...")
    
    try:
        data_loader.create_view(
            view_name=view_name,
            view_definition=view_definition,
            user_id=user_id,
            session_id=session_id
        )
        
        content = f"View '{view_name}' created successfully. You can now query it like a table using: SELECT * FROM {view_name}"
        writer(content)
        
        artifacts = {
            "view_name": view_name,
            "user_id": user_id,
            "session_id": session_id,
            "view_definition": view_definition,
            "success": True
        }
        
        return (content, artifacts)
    except ValueError as e:
        error_msg = f"Error creating view: {str(e)}"
        logger.error(error_msg)
        writer(error_msg)
        return (error_msg, None)
    except Exception as e:
        error_msg = f"Error creating view: {str(e)}"
        logger.error(error_msg)
        writer(error_msg)
        return (error_msg, None)


@tool(response_format="content_and_artifact")
async def create_visualization(
    sql_query: str,
    plotly_code: str,
    runtime: ToolRuntime[ChatContext]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Create a visualization from SQL query results using custom plotly code.
    
    Args:
        sql_query: SQL query to create the DataFrame
        plotly_code: Python code using created df and plotly to create the visualization
        runtime: ToolRuntime containing user context
    
    Returns:
        Tuple of (content_string, artifacts_dict) with visualization result.
        The plotly_code should expect a DataFrame named 'df' and create a figure named 'fig'.
        Use mapbox for map visualizations and use mapbox_style='mapbox://styles/mapbox/streets-v11'.
        IMPORTANT: The df (from SQL query), px (plotly.express), pd (pandas) variables are already 
        imported and available in the namespace, do not create them again.
    """
    writer = runtime.stream_writer
    user_id = runtime.context.user_id
    session_id = runtime.context.thread_id
    
    MAPBOX_ACCESS_TOKEN = "pk.eyJ1Ijoiam9obi1qb2huMTIzNDIzNCIsImEiOiJjbTRza3hpencwMXdrMnJzY3lyYnR2ZjJ6In0.LxOF4CYHZ2r99hlnW4Ai8w"
    
    writer(f"Creating visualization from SQL query...")
    
    try:
        # Get DuckDB connection
        conn = data_loader.get_connection()
        
        # Get list of tables accessible by this user (similar to query_duckdb)
        metadata_query = "SELECT table_name FROM metadata WHERE user_id = ?"
        metadata_params = [user_id]
        
        if session_id:
            metadata_query += " AND session_id = ?"
            metadata_params.append(session_id)
        
        try:
            accessible_tables_df = conn.execute(metadata_query, metadata_params).fetchdf()
            accessible_tables = set(accessible_tables_df['table_name'].tolist()) if not accessible_tables_df.empty else set()
            accessible_tables.add('metadata')
            accessible_views = data_loader.get_accessible_views(user_id, session_id)
            accessible_tables.update(accessible_views)
        except Exception:
            accessible_tables = {'metadata'}
        
        # Basic SQL injection prevention
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                error_msg = f"Error: SQL query contains forbidden keyword '{keyword}'. Only SELECT queries are allowed."
                writer(error_msg)
                return (error_msg, None)
        
        # Ensure it's a SELECT query
        if not sql_query.strip().upper().startswith('SELECT'):
            error_msg = "Error: Only SELECT queries are allowed."
            writer(error_msg)
            return (error_msg, None)
        
        writer("Executing SQL query to get DataFrame...")
        
        # Execute SQL query to get DataFrame
        try:
            df = conn.execute(sql_query).fetchdf()
        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            logger.error(error_msg)
            writer(error_msg)
            return (error_msg, None)
        
        if df.empty:
            error_msg = "SQL query returned no results. Cannot create visualization from empty DataFrame."
            writer(error_msg)
            return (error_msg, None)
        
        writer(f"DataFrame created with {len(df)} rows. Executing plotly code...")
        
        # Set mapbox access token
        px.set_mapbox_access_token(MAPBOX_ACCESS_TOKEN)
        
        # Create a namespace for code execution
        namespace = {
            'df': df,
            'sql_query': sql_query,
            'conn': conn,
            'px': px,
            'pd': pd,
            'fig': None,
            'mapbox_access_token': MAPBOX_ACCESS_TOKEN
        }
        
        # Use regex to remove the full line containing "import plotly.express"
        cleaned_plotly_code = re.sub(r'^import\s+plotly\.express.*(\n|$)', '', plotly_code, flags=re.MULTILINE)
        # Also remove other common plotly imports
        cleaned_plotly_code = re.sub(r'^import\s+plotly.*(\n|$)', '', cleaned_plotly_code, flags=re.MULTILINE)
        cleaned_plotly_code = re.sub(r'^from\s+plotly.*(\n|$)', '', cleaned_plotly_code, flags=re.MULTILINE)
        
        # Execute the plotly code in the namespace
        try:
            exec(cleaned_plotly_code, namespace)
            fig = namespace.get('fig')
            
            if fig is None:
                error_msg = "Plotly code did not create a 'fig' object. Make sure your code creates a figure and assigns it to 'fig'."
                writer(error_msg)
                return (error_msg, None)
                
        except Exception as e:
            error_msg = f"Error in plotly code execution: {str(e)}"
            logger.error(error_msg)
            writer(error_msg)
            return (error_msg, None)
        
        # Convert plotly figure to JSON
        try:
            fig_json = fig.to_json()
        except Exception as e:
            error_msg = f"Error serializing plotly figure to JSON: {str(e)}"
            logger.error(error_msg)
            writer(error_msg)
            return (error_msg, None)
        
        # Generate artifact ID
        artifact_id = str(uuid.uuid4())
        
        # Create artifacts dictionary
        artifacts = {
            "artifact_id": artifact_id,
            "artifact_type": "application/vnd.plotly.v1+json",
            "plotly_fig_json": fig_json,
            "user_id": user_id,
            "session_id": session_id,
            "sql_query": sql_query,
            "row_count": len(df)
        }
        
        content = f"Visualization created successfully from plotly code. DataFrame had {len(df)} rows. artifact id is <artifact_id={artifact_id}>"
        writer(content)
        
        return (content, artifacts)
        
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        writer(error_msg)
        return (error_msg, None)


def get_tools() -> List:
    """Get all available tools."""
    return [calculator, get_current_time, search_documents, query_duckdb, create_view, create_visualization]

 