"""Configuration constants for the chatbot backend."""

import os
from pathlib import Path

# Default model configuration (matching IResearcher-v5 OpenRouter setup)
DEFAULT_MODEL_CONFIG = {
    "model": "openai/gpt-5.1",
    "streaming": True,
    "temperature": 0.1,
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "extra_body": {
        "transforms": ["middle-out"],
        "models": ["google/gemini-2.5-pro"],
        "usage": {"include": True}
    }
}

# Database paths
DATA_DIR = Path("./data/chatbot")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DB_PATH = str(DATA_DIR / "conversations.db")
DOCUMENTS_DB_PATH = str(DATA_DIR / "documents.db")
FRONTEND_STORE_DB_PATH = str(DATA_DIR / "frontend_store.db")
DUCKDB_PATH = str(DATA_DIR / "data_warehouse.duckdb")

# Conversation settings
MAX_RECENT_ITEMS = 50  # Maximum messages to load per conversation

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

