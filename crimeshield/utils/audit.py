"""
CrimeShield AI — Audit Logger.

Writes one JSON-line per invocation to the configured audit log file.
Append-only; never truncates.
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
    """Append-only JSON-lines audit logger for CrimeShield invocations."""

    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path or AUDIT_LOG_PATH)
        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, state: Dict[str, Any]) -> None:
        """
        Write a single JSON audit line from the current AgentState.

        Fields recorded:
            timestamp, query_type, is_safe, confidence_score,
            node_visited, response_length, pii_detected,
            citations_count, refusal_reason, safety_metadata summary.
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_type": state.get("query_type", "unknown"),
            "is_safe": state.get("is_safe", True),
            "confidence_score": state.get("confidence_score", 0.0),
            "node_visited": state.get("query_type", "unknown"),
            "response_length": len(state.get("agent_response", "")),
            "pii_detected": state.get("pii_detected", False),
            "citations_count": len(state.get("citations", [])),
        }

        # Optional fields
        refusal = state.get("refusal_reason", "")
        if refusal:
            entry["refusal_reason"] = refusal

        safety_meta = state.get("safety_metadata", {})
        if safety_meta:
            entry["safety_metadata_summary"] = {
                k: v for k, v in safety_meta.items()
                if k in ("intent", "confidence", "unsafe_reason", "blocklist_hit")
            }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug("Audit entry written to %s", self.log_path)
        except OSError as exc:
            logger.error("Failed to write audit log: %s", exc)
