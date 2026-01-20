"""Data loader for Excel files using DuckDB."""

import pandas as pd
import duckdb
import hashlib
import re
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from io import BytesIO

from config import DUCKDB_PATH

logger = logging.getLogger(__name__)

# Global DuckDB connection (persistent)
_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get or create persistent DuckDB connection."""
    global _conn
    if _conn is None:
        _conn = duckdb.connect(DUCKDB_PATH)
        logger.info(f"Connected to DuckDB at {DUCKDB_PATH}")
    return _conn


def generate_table_name(file_path: str) -> str:
    """
    Generate a table name from filename + file hash.
    Format: filename_lowercase_letters_only_hash5
    Only keeps letters (a-z), removes numbers, special characters, and stop words.
    """
    # Common stop words to remove
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'wise', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'could', 'should', 'would', 'may', 'might', 'must', 'shall'
    }
    
    # Get filename without extension
    path_obj = Path(file_path)
    filename = path_obj.stem  # filename without extension
    
    # Generate file hash from filename (since we don't have file bytes yet)
    # We'll use filename + timestamp for uniqueness
    import time
    file_hash = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()[:5]
    
    # Convert to lowercase and keep ONLY letters (a-z), replace everything else with underscore
    # This removes numbers, spaces, hyphens, and all special characters
    name_part = re.sub(r'[^a-z]', '_', filename.lower())
    
    # Split by underscores and filter out stop words
    words = [word for word in name_part.split('_') if word and word not in stop_words]
    
    # Join words back with underscores
    name_part = '_'.join(words)
    
    # Remove multiple consecutive underscores and strip underscores from ends
    name_part = re.sub(r'_+', '_', name_part).strip('_')
    
    # If name_part is empty after removing all non-letters and stop words, use a default
    if not name_part:
        name_part = 'table'
    
    # Combine: filename_hash
    table_name = f"{name_part}_{file_hash}"
    
    return table_name


def load_excel_to_dataframe(excel_bytes: bytes) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load an Excel file from bytes and extract both metadata and data table.
    Returns a tuple: (metadata_dict, cleaned_dataframe)
    """
    # Read the raw Excel file without header to see all rows
    raw_df = pd.read_excel(BytesIO(excel_bytes), header=None, engine='openpyxl')
    
    # Extract metadata from rows 0-5, first column (index 0)
    metadata = {}
    if len(raw_df) > 0:
        first_col = raw_df.iloc[:, 0]
        
        # Row 0: Company name
        val_0 = first_col.iloc[0]
        if pd.notna(val_0):
            metadata['name'] = str(val_0).strip()
        
        # Row 1: Address
        val_1 = first_col.iloc[1]
        if pd.notna(val_1):
            metadata['address'] = str(val_1).strip()
        
        # Row 3: Report title
        val_3 = first_col.iloc[3]
        if pd.notna(val_3):
            metadata['report_title'] = str(val_3).strip()
        
        # Row 4: Company/Client
        val_4 = first_col.iloc[4]
        if pd.notna(val_4):
            company_str = str(val_4).strip()
            if 'Company' in company_str:
                metadata['client'] = company_str.replace('Company', '').replace(':', '').strip()
            else:
                metadata['client'] = company_str
        
        # Row 5: Date range
        val_5 = first_col.iloc[5]
        if pd.notna(val_5):
            date_str = str(val_5).strip()
            metadata['date_range'] = date_str
            if 'From' in date_str and 'To' in date_str:
                try:
                    if 'From:' in date_str:
                        from_date = date_str.split('From:')[1].split('To:')[0].strip()
                        to_date = date_str.split('To:')[1].strip()
                    elif 'From Date' in date_str:
                        from_date = date_str.split('From Date')[1].split('To')[0].replace(':', '').strip()
                        to_date = date_str.split('To Dt:')[1].strip()
                    else:
                        from_date = date_str.split('From')[1].split('To')[0].strip()
                        to_date = date_str.split('To')[1].strip()
                    metadata['from_date'] = from_date
                    metadata['to_date'] = to_date
                except (ValueError, IndexError) as e:
                    logger.warning("Error parsing dates: %s", e)
    
    # Load the table starting from row 6 (which contains column headers)
    df = pd.read_excel(BytesIO(excel_bytes), header=6, engine='openpyxl')
    
    # Use the first row as column names
    if len(df) > 0:
        df.columns = df.iloc[0].values
        # Drop the first row (it was the header row)
        df = df.drop(df.index[0])
    
    # Clean up the DataFrame
    df = df.dropna(how='all')
    df = df.reset_index(drop=True)
    
    # Remove any rows where the first column contains the header names (duplicate headers)
    if len(df) > 0:
        first_col_name = df.columns[0]
        # Remove rows where first column equals the column name itself
        df = df[df[first_col_name] != first_col_name]
        df = df.reset_index(drop=True)
    
    # Convert date columns to DATE type if they exist
    date_columns = ['Date', 'from_date', 'to_date']  # Common date column names
    for col in date_columns:
        if col in df.columns:
            try:
                # Try to parse dates in DD/MM/YYYY format
                df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
            except (ValueError, TypeError):
                try:
                    # Fallback to auto-detect format
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    pass
    
    return metadata, df


def load_excel_file(
    excel_bytes: bytes,
    filename: str,
    user_id: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Load an Excel file and store it in DuckDB.
    
    Args:
        excel_bytes: Excel file content as bytes
        filename: Original filename
        user_id: User ID
        session_id: Session ID
        
    Returns:
        Dictionary with table_name, row_count, column_count, and metadata
    """
    conn = get_connection()
    
    try:
        # Generate table name automatically
        table_name = generate_table_name(filename)
        
        logger.info("Loading Excel file: %s -> table: %s", filename, table_name)
        
        # Load metadata and dataframe
        metadata, df = load_excel_to_dataframe(excel_bytes)
        
        # Add user_id and session_id to metadata
        metadata['user_id'] = user_id
        metadata['session_id'] = session_id
        metadata['filename'] = filename
        
        logger.info("Loaded DataFrame shape: %s", df.shape)
        logger.info("Metadata: %s", metadata)
        
        # Create persistent DuckDB table directly from DataFrame
        # First register as a temporary view, then create persistent table
        temp_view = f"temp_{table_name}"
        conn.register(temp_view, df)
        
        # Create persistent table from the temporary view
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {temp_view}")
        
        # Unregister the temporary view
        conn.unregister(temp_view)
        
        logger.info("Created DuckDB table: %s", table_name)
        
        # Update metadata table
        update_metadata_table(table_name, filename, metadata, len(df), len(df.columns))
        
        return {
            'table_name': table_name,
            'filename': filename,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error("Error loading Excel file %s: %s", filename, e)
        raise


def update_metadata_table(
    table_name: str,
    source_file: str,
    metadata: Dict[str, Any],
    row_count: int,
    column_count: int
):
    """Create or update the metadata table in DuckDB."""
    conn = get_connection()
    
    # Check if metadata table exists
    try:
        result = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'metadata'").fetchone()
        table_exists = result[0] > 0
    except:
        table_exists = False
    
    if not table_exists:
        # Create metadata table
        conn.execute("""
            CREATE TABLE metadata (
                table_name VARCHAR,
                source_file VARCHAR,
                name VARCHAR,
                address VARCHAR,
                report_title VARCHAR,
                client VARCHAR,
                date_range VARCHAR,
                from_date VARCHAR,
                to_date VARCHAR,
                user_id VARCHAR,
                session_id VARCHAR,
                filename VARCHAR,
                row_count INTEGER,
                column_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created metadata table")
    
    # Insert or replace metadata row
    conn.execute(
        """
        INSERT INTO metadata (
            table_name, source_file, name, address, report_title, client,
            date_range, from_date, to_date, user_id, session_id, filename,
            row_count, column_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            table_name,
            source_file,
            metadata.get('name'),
            metadata.get('address'),
            metadata.get('report_title'),
            metadata.get('client'),
            metadata.get('date_range'),
            metadata.get('from_date'),
            metadata.get('to_date'),
            metadata.get('user_id'),
            metadata.get('session_id'),
            metadata.get('filename'),
            row_count,
            column_count
        ]
    )
    
    logger.info("Updated metadata table for %s", table_name)


def list_tables(user_id: Optional[str] = None, session_id: Optional[str] = None) -> list:
    """List all tables in DuckDB, optionally filtered by user_id and/or session_id."""
    conn = get_connection()
    
    try:
        query = "SELECT * FROM metadata WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY created_at DESC"
        
        result = conn.execute(query, params).fetchdf()
        return result.to_dict('records')
    except Exception as e:
        logger.error("Error listing tables: %s", e)
        return []


def get_table_data(table_name: str, limit: int = 100) -> Dict[str, Any]:
    """Get data from a specific table."""
    conn = get_connection()
    
    try:
        # Get table data
        df = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
        
        # Get metadata
        metadata = conn.execute(
            "SELECT * FROM metadata WHERE table_name = ?",
            [table_name]
        ).fetchdf()
        
        return {
            'table_name': table_name,
            'data': df.to_dict('records'),
            'metadata': metadata.to_dict('records')[0] if len(metadata) > 0 else None,
            'row_count': len(df)
        }
    except Exception as e:
        logger.error("Error getting table data for %s: %s", table_name, e)
        raise


def get_metadata_csv(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Get metadata table as CSV string, filtered by user_id and/or session_id.
    
    Args:
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        
    Returns:
        CSV string of metadata table
    """
    conn = get_connection()
    
    try:
        query = "SELECT * FROM metadata WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY created_at DESC"
        
        df = conn.execute(query, params).fetchdf()
        
        if df.empty:
            return ""
        
        # Convert to CSV string
        return df.to_csv(index=False)
    except Exception as e:
        logger.error("Error getting metadata CSV: %s", e)
        raise


def get_tables_by_schema_csv(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Get table names grouped by unique table schemas (column name and column type only).
    Returns markdown with embedded CSV sections.
    
    Format:
    ---
    ## tables: [table1, table2]
    ## schema:
    csv_data
    ---
    
    Args:
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        
    Returns:
        Markdown string with embedded CSV sections
    """
    conn = get_connection()
    
    try:
        # First, get list of tables filtered by user_id and session_id
        metadata_query = "SELECT table_name FROM metadata WHERE 1=1"
        metadata_params = []
        
        if user_id:
            metadata_query += " AND user_id = ?"
            metadata_params.append(user_id)
        
        if session_id:
            metadata_query += " AND session_id = ?"
            metadata_params.append(session_id)
        
        # Get table names
        tables_df = conn.execute(metadata_query, metadata_params).fetchdf()
        
        if tables_df.empty:
            return ""
        
        table_names = tables_df['table_name'].tolist()
        
        # Get schema information for each table
        # Map schema_hash to (schema_data, table_list)
        schema_map: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
        
        for table_name in table_names:
            try:
                # Use DESCRIBE to get column information in DuckDB
                schema_info = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                
                # Create a schema signature (sorted list of (column_name, column_type) tuples)
                schema_signature = tuple(
                    sorted(
                        [(row['column_name'], row['column_type']) for _, row in schema_info.iterrows()],
                        key=lambda x: x[0]  # Sort by column name
                    )
                )
                
                # Create a hash of the schema for grouping
                schema_hash = hashlib.md5(str(schema_signature).encode()).hexdigest()[:8]
                
                # Store schema information
                if schema_hash not in schema_map:
                    # Create a DataFrame with just column_name and column_type
                    schema_df = pd.DataFrame({
                        'column_name': schema_info['column_name'],
                        'column_type': schema_info['column_type']
                    })
                    schema_map[schema_hash] = (schema_df, [table_name])
                else:
                    # Add table to existing schema group
                    schema_map[schema_hash][1].append(table_name)
                    
            except Exception as e:
                logger.warning("Error getting schema for table %s: %s", table_name, e)
                continue
        
        if not schema_map:
            return ""
        
        # Build markdown output
        markdown_parts = []
        
        for schema_hash, (schema_df, tables) in schema_map.items():
            # Sort tables alphabetically
            tables_sorted = sorted(set(tables))
            
            # Format tables list
            tables_str = ', '.join(tables_sorted)
            
            # Get CSV for this schema (just column_name and column_type)
            schema_csv = schema_df.to_csv(index=False)
            
            # Build markdown section
            markdown_parts.append("---")
            markdown_parts.append(f"## tables: [{tables_str}]")
            markdown_parts.append("## schema:")
            markdown_parts.append(schema_csv)
        
        # Join all parts with newlines
        return "\n".join(markdown_parts)
    except Exception as e:
        logger.error("Error getting tables by schema CSV: %s", e)
        raise


def initialize_database():
    """Initialize the DuckDB database and metadata table."""
    conn = get_connection()
    
    # Create metadata table if it doesn't exist
    try:
        result = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'metadata'").fetchone()
        table_exists = result[0] > 0
    except Exception:
        table_exists = False
    
    if not table_exists:
        conn.execute("""
            CREATE TABLE metadata (
                table_name VARCHAR,
                source_file VARCHAR,
                name VARCHAR,
                address VARCHAR,
                report_title VARCHAR,
                client VARCHAR,
                date_range VARCHAR,
                from_date VARCHAR,
                to_date VARCHAR,
                user_id VARCHAR,
                session_id VARCHAR,
                filename VARCHAR,
                row_count INTEGER,
                column_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Initialized DuckDB metadata table")
    
    # Create view_metadata table if it doesn't exist
    try:
        result = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'view_metadata'").fetchone()
        view_metadata_exists = result[0] > 0
    except Exception:
        view_metadata_exists = False
    
    if not view_metadata_exists:
        conn.execute("""
            CREATE TABLE view_metadata (
                view_name VARCHAR,
                view_definition VARCHAR,
                user_id VARCHAR,
                session_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Initialized DuckDB view_metadata table")


def create_view(
    view_name: str,
    view_definition: str,
    user_id: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Create a persistent view in DuckDB with user_id and session_id tracking.
    
    Args:
        view_name: Name for the view (will be validated and sanitized)
        view_definition: SQL SELECT statement defining the view
        user_id: User ID who owns this view
        session_id: Session ID that owns this view
        
    Returns:
        Dictionary with view_name and success status
    """
    conn = get_connection()
    
    # Sanitize view name (only alphanumeric and underscores)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', view_name):
        raise ValueError(f"Invalid view name: {view_name}. Only alphanumeric characters and underscores are allowed.")
    
    # Ensure view_definition is a SELECT statement
    view_def_upper = view_definition.strip().upper()
    if not view_def_upper.startswith('SELECT'):
        raise ValueError("View definition must be a SELECT statement")
    
    # Check for dangerous operations in view definition
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
    for keyword in dangerous_keywords:
        if f' {keyword} ' in f' {view_def_upper} ':
            raise ValueError(f"View definition contains forbidden keyword: {keyword}")
    
    try:
        # Create the view
        conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {view_definition}")
        
        # Store metadata
        conn.execute(
            """
            INSERT INTO view_metadata (view_name, view_definition, user_id, session_id)
            VALUES (?, ?, ?, ?)
            """,
            [view_name, view_definition, user_id, session_id]
        )
        
        logger.info("Created view %s for user %s, session %s", view_name, user_id, session_id)
        
        return {
            'view_name': view_name,
            'user_id': user_id,
            'session_id': session_id,
            'success': True
        }
    except Exception as e:
        logger.error("Error creating view %s: %s", view_name, e)
        raise


def list_views(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all views, optionally filtered by user_id and/or session_id.
    
    Args:
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        
    Returns:
        List of dictionaries with view metadata
    """
    conn = get_connection()
    
    try:
        query = "SELECT * FROM view_metadata WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY created_at DESC"
        
        result = conn.execute(query, params).fetchdf()
        return result.to_dict('records')
    except Exception as e:
        logger.error("Error listing views: %s", e)
        return []


def get_accessible_views(
    user_id: str,
    session_id: Optional[str] = None
) -> set:
    """
    Get set of view names accessible by the given user_id and optionally session_id.
    
    Args:
        user_id: User ID
        session_id: Optional session ID
        
    Returns:
        Set of view names
    """
    views = list_views(user_id=user_id, session_id=session_id)
    return {view['view_name'] for view in views}


def drop_view(
    view_name: str,
    user_id: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Drop a view if it belongs to the specified user_id (and optionally session_id).
    
    Args:
        view_name: Name of the view to drop
        user_id: User ID who owns the view
        session_id: Optional session ID filter
        
    Returns:
        Dictionary with success status
    """
    conn = get_connection()
    
    # Check if view exists and belongs to this user
    query = "SELECT view_name FROM view_metadata WHERE view_name = ? AND user_id = ?"
    params = [view_name, user_id]
    
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    
    result = conn.execute(query, params).fetchdf()
    
    if result.empty:
        raise ValueError(f"View '{view_name}' not found or access denied")
    
    try:
        # Drop the view
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        
        # Remove from metadata
        delete_query = "DELETE FROM view_metadata WHERE view_name = ? AND user_id = ?"
        delete_params = [view_name, user_id]
        if session_id:
            delete_query += " AND session_id = ?"
            delete_params.append(session_id)
        
        conn.execute(delete_query, delete_params)
        
        logger.info("Dropped view %s for user %s", view_name, user_id)
        
        return {
            'view_name': view_name,
            'success': True
        }
    except Exception as e:
        logger.error("Error dropping view %s: %s", view_name, e)
        raise

