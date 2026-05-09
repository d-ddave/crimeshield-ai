"""
CrimeShield AI — AgentState definition.
"""
from __future__ import annotations
from typing import TypedDict

class AgentState(TypedDict, total=False):
    query: str
    query_type: str
    agent_response: str
    citations: list
    is_safe: bool
    confidence_score: float
    plan: list[dict]
    current_step: int
    pii_detected: bool
    masked_query: str
    safety_metadata: dict
    audit_trail: list
    refusal_reason: str
