"""
CrimeShield AI — Audit Logger.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from crimeshield.config import AUDIT_LOG_PATH

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path or AUDIT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, state: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_type": state.get("query_type", "unknown"),
            "is_safe": state.get("is_safe", True),
            "confidence_score": state.get("confidence_score", 0.0),
            "response_length": len(state.get("agent_response", "")),
            "pii_detected": state.get("pii_detected", False),
            "citations_count": len(state.get("citations", [])),
        }
        refusal = state.get("refusal_reason", "")
        if refusal:
            entry["refusal_reason"] = refusal
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to write audit log: %s", exc)
