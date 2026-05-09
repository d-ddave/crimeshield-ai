"""
CrimeShield AI — AgentState definition.

Typed dictionary used as the shared state across every node in the
LangGraph StateGraph.  Every field has a default so that nodes can
be invoked individually during testing.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the CrimeShield LangGraph."""

    # ── Input ──────────────────────────────────────────────────────────
    query: str

    # ── Routing / classification ───────────────────────────────────────
    query_type: str  # 'policy' | 'structured_data' | 'sar' | 'refused'

    # ── Agent output ───────────────────────────────────────────────────
    agent_response: str
    citations: list
    is_safe: bool
    confidence_score: float

    # ── Planning / supervisor ──────────────────────────────────────────
    plan: list[dict]
    current_step: int

    # ── PII handling ───────────────────────────────────────────────────
    pii_detected: bool
    masked_query: str

    # ── Safety ─────────────────────────────────────────────────────────
    safety_metadata: dict

    # ── Audit ──────────────────────────────────────────────────────────
    audit_trail: list

    # ── Refusal ────────────────────────────────────────────────────────
    refusal_reason: str
