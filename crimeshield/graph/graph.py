"""
CrimeShield AI — LangGraph StateGraph.

Builds the full 8-node state graph with conditional routing:
  START → safety_guard → (refused | supervisor) → planner →
  (rag_policy | structured_data | sar_drafting) → validator →
  (planner retry | response_merger) → END
"""

from __future__ import annotations

import json
import logging
import re
import string
from typing import Any, Dict

from langchain_community.vectorstores import Chroma
from langgraph.graph import END, START, StateGraph

import pandas as pd

from crimeshield.agents.rag import RAGPolicyAgent
from crimeshield.agents.sar import SARDraftingAgent
from crimeshield.agents.structured_data import StructuredDataAgent
from crimeshield.config import MAX_SUPERVISOR_STEPS
from crimeshield.graph.state import AgentState
from crimeshield.pipeline.load_data import DataLoader
from crimeshield.utils.audit import AuditLogger
from crimeshield.utils.llm import get_llm
from crimeshield.utils.pii import PIIRedactor
from crimeshield.utils.prompts import load_prompt

logger = logging.getLogger(__name__)

# ── Safety guard config loaded from prompts/safety_guard.yaml ──────────
_SAFETY_CFG = load_prompt("safety_guard")
_BLOCKED_PATTERNS: list[str] = _SAFETY_CFG.get("blocked_patterns", [])
_INJECTION_PREFIXES: list[str] = _SAFETY_CFG.get("injection_prefixes", [])


def build_graph(
    vectorstore: Chroma,
    alerts_df: pd.DataFrame,
    typologies_df: pd.DataFrame,
) -> Any:
    """
    Build and compile the CrimeShield LangGraph StateGraph.
    """
    rag_agent = RAGPolicyAgent(vectorstore)
    structured_agent = StructuredDataAgent(alerts_df, typologies_df)
    sar_agent = SARDraftingAgent()
    pii_redactor = PIIRedactor()
    audit_logger = AuditLogger()
    data_loader = DataLoader()

    try:
        alert_json_records = data_loader.load_alert_json()
    except FileNotFoundError:
        alert_json_records = []

    # Small LLM for safety classification (faster/cheaper)
    safety_llm = get_llm(small=True)

    # ====================================================================
    # NODE FUNCTIONS
    # ====================================================================

    def safety_guard_node(state: AgentState) -> dict:
        """
        5-stage safety check:
        1. Pre-processing (control chars, injection detection)
        2. PII masking
        3. Intent + safety classification via LLM
        4. Policy blocklist check
        5. Decision
        """
        query = state.get("query", "")
        safety_meta: Dict[str, Any] = {}

        # ── Stage 1: Pre-processing ───────────────────────────────────
        cleaned = "".join(
            ch for ch in query
            if ch in string.printable
        )

        query_lower = cleaned.lower().strip()
        injection_detected = False
        for prefix in _INJECTION_PREFIXES:
            if prefix in query_lower:
                injection_detected = True
                safety_meta["injection_pattern"] = prefix
                break

        base64_pattern = re.compile(r"[A-Za-z0-9+/=]{50,}")
        if base64_pattern.search(cleaned):
            injection_detected = True
            safety_meta["base64_detected"] = True

        # ── Stage 2: PII masking ──────────────────────────────────────
        pii_detected = pii_redactor.has_pii(cleaned)
        masked_query = pii_redactor.redact(cleaned)

        # ── Stage 3: Intent + safety classification via small LLM ─────
        system_text = _SAFETY_CFG["system"].strip()
        human_text = _SAFETY_CFG.get("human_template", "").strip()
        classification_prompt = (
            system_text + "\n\n" + human_text.format(masked_query=masked_query)
        )

        intent = "policy"
        confidence = 0.5
        unsafe_reason = ""

        if not injection_detected:
            try:
                response = safety_llm.invoke(classification_prompt)
                content = response.content.strip()
                if content.startswith("```"):
                    content = re.sub(r"```(?:json)?\s*", "", content)
                    content = content.rstrip("`").strip()
                parsed = json.loads(content)
                intent = parsed.get("intent", "policy")
                confidence = float(parsed.get("confidence", 0.5))
                unsafe_reason = parsed.get("unsafe_reason", "")
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("LLM classification failed: %s", exc)
                intent = "policy"
                confidence = 0.5

        safety_meta["llm_intent"] = intent
        safety_meta["llm_confidence"] = confidence
        if unsafe_reason:
            safety_meta["unsafe_reason"] = unsafe_reason

        # ── Stage 4: Policy blocklist check ───────────────────────────
        blocklist_hit = False
        for pattern in _BLOCKED_PATTERNS:
            if pattern in query_lower:
                blocklist_hit = True
                safety_meta["blocklist_hit"] = pattern
                break

        # ── Stage 5: Decision ─────────────────────────────────────────
        is_unsafe = (
            injection_detected
            or blocklist_hit
            or intent == "unsafe"
            or intent == "off_topic"
        )

        if is_unsafe:
            refusal_reason = ""
            if injection_detected:
                refusal_reason = f"Prompt injection detected: {safety_meta.get('injection_pattern', 'base64')}"
            elif blocklist_hit:
                refusal_reason = f"Query matches blocked pattern: {safety_meta.get('blocklist_hit', '')}"
            elif intent == "unsafe":
                refusal_reason = f"Classified as unsafe: {unsafe_reason}"
            elif intent == "off_topic":
                refusal_reason = "Query is off-topic for CrimeShield AI"

            return {
                "query": query,
                "masked_query": masked_query,
                "pii_detected": pii_detected,
                "is_safe": False,
                "query_type": "refused",
                "refusal_reason": refusal_reason,
                "safety_metadata": safety_meta,
                "confidence_score": confidence,
                "audit_trail": [{"node": "safety_guard", "result": "refused"}],
                "citations": [],
                "agent_response": "",
                "plan": [],
                "current_step": 0,
            }

        return {
            "query": query,
            "masked_query": masked_query,
            "pii_detected": pii_detected,
            "is_safe": True,
            "query_type": intent,
            "refusal_reason": "",
            "safety_metadata": safety_meta,
            "confidence_score": confidence,
            "audit_trail": [{"node": "safety_guard", "result": "passed"}],
            "citations": [],
            "agent_response": "",
            "plan": [],
            "current_step": 0,
        }

    def supervisor_node(state: AgentState) -> dict:
        """Check safety, enforce step limits, set quality reminders."""
        if not state.get("is_safe", False):
            return {}

        current_step = state.get("current_step", 0)
        if current_step > MAX_SUPERVISOR_STEPS:
            return {
                "agent_response": "Maximum processing steps exceeded. Returning best available result.",
                "safety_metadata": {
                    **state.get("safety_metadata", {}),
                    "step_limit_reached": True,
                },
            }

        plan = state.get("plan", [])
        query_type = state.get("query_type", "policy")
        if not plan:
            plan = [{"step": 1, "agent": query_type}]

        safety_meta = state.get("safety_metadata", {})
        safety_meta["quality_threshold"] = "Ensure response is comprehensive and well-cited"

        return {
            "plan": plan,
            "safety_metadata": safety_meta,
        }

    def planner_node(state: AgentState) -> dict:
        """Build execution plan based on query type."""
        if not state.get("is_safe", False):
            return {}

        query_type = state.get("query_type", "policy")
        query = state.get("query", "")

        plan = [{"step": 1, "agent": query_type}]

        if query_type == "sar":
            alert_match = re.search(r"\bALT-[A-Z]?\d+\b", query, re.IGNORECASE)
            if alert_match:
                plan[0]["alert_id"] = alert_match.group()

        return {
            "plan": plan,
            "current_step": 0,
        }

    def rag_policy_node(state: AgentState) -> dict:
        """Invoke the RAG policy agent."""
        query = state.get("masked_query", state.get("query", ""))
        result = rag_agent.invoke(query)

        audit_trail = state.get("audit_trail", [])
        audit_trail.append({
            "node": "rag_policy",
            "declined": result.get("declined", False),
            "hyde_query": result.get("hyde_query", ""),
            "metadata_filter": result.get("metadata_filter"),
        })

        return {
            "agent_response": result["answer"],
            "citations": result["citations"],
            "confidence_score": result["confidence"],
            "audit_trail": audit_trail,
        }

    def structured_data_node(state: AgentState) -> dict:
        """Invoke the structured data agent."""
        query = state.get("masked_query", state.get("query", ""))
        result = structured_agent.invoke(query)

        audit_trail = state.get("audit_trail", [])
        audit_trail.append({
            "node": "structured_data",
            "tool_calls": result.get("tool_calls", []),
        })

        return {
            "agent_response": result["answer"],
            "confidence_score": 1.0,
            "audit_trail": audit_trail,
        }

    def sar_drafting_node(state: AgentState) -> dict:
        """Invoke the SAR drafting agent with alert data."""
        query = state.get("query", "")

        alert_record = None
        plan = state.get("plan", [])
        alert_id = None

        if plan and "alert_id" in plan[0]:
            alert_id = plan[0]["alert_id"]
        else:
            match = re.search(r"\bALT-[A-Z]?\d+\b", query, re.IGNORECASE)
            if match:
                alert_id = match.group()

        if alert_id:
            for record in alert_json_records:
                if record.get("alert_id", "").upper() == alert_id.upper():
                    alert_record = record
                    break

        if alert_record is None:
            alert_record = {
                "alert_id": alert_id or "UNKNOWN",
                "query_context": query,
                "source": "parsed_from_query",
            }

        result = sar_agent.invoke(alert_record)

        audit_trail = state.get("audit_trail", [])
        audit_trail.append({
            "node": "sar_drafting",
            "alert_id": result.get("alert_id", ""),
            "sections_present": result.get("sections_present", []),
            "grounded": result.get("grounded", False),
        })

        return {
            "agent_response": result["narrative"],
            "confidence_score": 0.9 if result["grounded"] else 0.6,
            "audit_trail": audit_trail,
        }

    def validator_node(state: AgentState) -> dict:
        """
        Deterministic validation — NO LLM call.
        Checks response quality and completeness based on query type.
        """
        response = state.get("agent_response", "")
        query_type = state.get("query_type", "")
        current_step = state.get("current_step", 0)

        validation_passed = True
        validation_errors: list[str] = []

        if not response or len(response) < 50:
            validation_passed = False
            validation_errors.append(f"Response too short ({len(response)} chars)")

        if query_type == "policy":
            citations = state.get("citations", [])
            if not citations:
                validation_passed = False
                validation_errors.append("No citations in policy response")

        if query_type == "sar":
            required_sections = [
                "## Subject Summary",
                "## Suspicious Activity Description",
                "## Supporting Evidence",
                "## Recommended Action",
            ]
            for section in required_sections:
                if section not in response:
                    validation_passed = False
                    validation_errors.append(f"Missing section: {section}")

        if query_type == "structured_data":
            if not re.search(r"\d+", response):
                validation_passed = False
                validation_errors.append("Structured data response contains no numbers")

        audit_trail = state.get("audit_trail", [])
        audit_trail.append({
            "node": "validator",
            "passed": validation_passed,
            "errors": validation_errors,
        })

        if not validation_passed and current_step < MAX_SUPERVISOR_STEPS:
            return {
                "current_step": current_step + 1,
                "audit_trail": audit_trail,
                "safety_metadata": {
                    **state.get("safety_metadata", {}),
                    "validation_errors": validation_errors,
                    "retry": True,
                },
            }

        return {
            "audit_trail": audit_trail,
            "safety_metadata": {
                **state.get("safety_metadata", {}),
                "validated": validation_passed,
                "validation_errors": validation_errors,
            },
        }

    def response_merger_node(state: AgentState) -> dict:
        """
        Final output processing:
        1. PII redaction of agent response
        2. Append citations
        3. Append confidence indicator
        4. Write audit log
        """
        response = state.get("agent_response", "")

        redacted_response = pii_redactor.redact(response)

        citations = state.get("citations", [])
        if citations:
            redacted_response += "\n\n---\n**Sources:**\n"
            for cite in citations:
                section = cite.get("source_section", "Unknown")
                chunk_id = cite.get("chunk_id", "")
                score = cite.get("relevance_score", 0.0)
                redacted_response += f"- {section} (chunk: {chunk_id}, relevance: {score:.4f})\n"

        score = state.get("confidence_score", 0.0)
        if score >= 0.8:
            confidence_label = "HIGH"
        elif score >= 0.5:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"
        redacted_response += f"\nConfidence: {confidence_label}"

        final_state = {
            **state,
            "agent_response": redacted_response,
        }

        audit_logger.log(final_state)

        audit_trail = state.get("audit_trail", [])
        audit_trail.append({"node": "response_merger", "audit_written": True})

        return {
            "agent_response": redacted_response,
            "audit_trail": audit_trail,
        }

    def refused_node(state: AgentState) -> dict:
        """Return a firm, professional refusal and log it."""
        refusal_response = _SAFETY_CFG.get(
            "decline_message",
            "I'm unable to assist with that request. CrimeShield AI is designed "
            "to support economic crime prevention activities only. This query "
            "has been logged.",
        ).strip()

        audit_trail = state.get("audit_trail", [])
        audit_trail.append({"node": "refused", "logged": True})

        final_state = {
            **state,
            "agent_response": refusal_response,
        }
        audit_logger.log(final_state)

        return {
            "agent_response": refusal_response,
            "audit_trail": audit_trail,
        }

    # ====================================================================
    # BUILD THE GRAPH
    # ====================================================================

    graph = StateGraph(AgentState)

    graph.add_node("safety_guard", safety_guard_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("rag_policy", rag_policy_node)
    graph.add_node("structured_data", structured_data_node)
    graph.add_node("sar_drafting", sar_drafting_node)
    graph.add_node("validator", validator_node)
    graph.add_node("response_merger", response_merger_node)
    graph.add_node("refused", refused_node)

    graph.add_edge(START, "safety_guard")

    def route_after_safety(state: AgentState) -> str:
        if state.get("query_type") == "refused":
            return "refused"
        return "supervisor"

    graph.add_conditional_edges(
        "safety_guard",
        route_after_safety,
        {"refused": "refused", "supervisor": "supervisor"},
    )

    graph.add_edge("supervisor", "planner")

    def route_after_planner(state: AgentState) -> str:
        query_type = state.get("query_type", "policy")
        if query_type == "structured_data":
            return "structured_data"
        elif query_type == "sar":
            return "sar_drafting"
        else:
            return "rag_policy"

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "rag_policy": "rag_policy",
            "structured_data": "structured_data",
            "sar_drafting": "sar_drafting",
        },
    )

    graph.add_edge("rag_policy", "validator")
    graph.add_edge("structured_data", "validator")
    graph.add_edge("sar_drafting", "validator")

    def route_after_validator(state: AgentState) -> str:
        safety_meta = state.get("safety_metadata", {})
        if safety_meta.get("retry", False):
            return "planner"
        return "response_merger"

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {"planner": "planner", "response_merger": "response_merger"},
    )

    graph.add_edge("response_merger", END)
    graph.add_edge("refused", END)

    compiled = graph.compile()

    return compiled
