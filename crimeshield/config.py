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
# xAI Grok
# ---------------------------------------------------------------------------
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL: str = "https://api.x.ai/v1"

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
LLM_MODEL: str = "grok-3"
SMALL_LLM_MODEL: str = "grok-3-mini"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
HYDE_ENABLED: bool = True
TOP_K_RETRIEVAL: int = 10
TOP_K_RERANKED: int = 5
RRF_K: int = 60
SIMILARITY_THRESHOLD: float = 0.25  # reranker handles quality now

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
