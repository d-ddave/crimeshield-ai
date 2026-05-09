"""
CrimeShield AI — Centralised Configuration
Loads environment variables via python-dotenv and exposes all
application settings as module-level constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the crimeshield package root (or project root)
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

# Try crimeshield/.env first, then project root .env
for _candidate in [_PACKAGE_DIR / ".env", _PROJECT_ROOT / ".env"]:
    if _candidate.exists():
        load_dotenv(_candidate)
        break
else:
    load_dotenv()  # fall back to default .env search

# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
LLM_MODEL: str = "gemini-2.0-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = str(_PROJECT_ROOT / "chroma_store")
CHROMA_COLLECTION: str = "policy_corpus"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL: int = 5
SIMILARITY_THRESHOLD: float = 0.35

# ---------------------------------------------------------------------------
# Data paths (relative to project root)
# ---------------------------------------------------------------------------
ALERTS_CSV: str = str(_PROJECT_ROOT / "data" / "alerts.csv")
TYPOLOGY_CSV: str = str(_PROJECT_ROOT / "data" / "typology_thresholds.csv")
ALERT_JSON: str = str(_PROJECT_ROOT / "data" / "alert.json")
POLICY_CORPUS: str = str(_PROJECT_ROOT / "data" / "policy_corpus.txt")

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH: str = str(_PROJECT_ROOT / "audit.log")

# ---------------------------------------------------------------------------
# Graph / Supervisor
# ---------------------------------------------------------------------------
MAX_SUPERVISOR_STEPS: int = 3
CONFIDENCE_THRESHOLD: float = 0.6
