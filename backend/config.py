"""
============================================================
AgriSahayak AI - Backend Configuration
============================================================
Central configuration file for backend resources.
Modify paths or settings here instead of changing multiple files.
"""

from pathlib import Path

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Model Paths
# ============================================================

MODEL_PATH = PROJECT_ROOT / "models" / "final_model"

# ============================================================
# Vector Database
# ============================================================

VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"

INDEX_PATH = VECTOR_DB_PATH / "faiss.index"

METADATA_PATH = VECTOR_DB_PATH / "metadata.json"

# ============================================================
# Retrieval Configuration
# ============================================================

TOP_K = 5

SEARCH_K = 15

# ============================================================
# Gemini Configuration
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash"

# ============================================================
# Streamlit Configuration
# ============================================================

APP_TITLE = "AgriSahayak AI"

APP_DESCRIPTION = (
    "AI-powered multilingual agriculture assistant "
    "using Retrieval-Augmented Generation."
)