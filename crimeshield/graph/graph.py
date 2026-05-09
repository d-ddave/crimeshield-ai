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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, START, StateGraph

import pandas as pd

from crimeshield.agents.rag import RAGPolicyAgent
from crimeshield.agents.sar import SARDraftingAgent
from crimeshield.agents.structured_data import StructuredDataAgent
from crimeshield.config import (
    ALERT_JSON,
    GEMINI_API_KEY,
    LLM_MODEL,
    MAX_SUPERVISOR_STEPS,
)
from crimeshield.graph.state import AgentState
from crimeshield.pipeline.load_data import DataLoader
from crimeshield.utils.audit import AuditLogger
from crimeshield.utils.pii import PIIRedactor
from crimeshield.utils.prompts import load_prompt

logger = logging.getLogger(__name__)

_SAFETY_CFG = load_prompt("safety_guard")
_BLOCKED_PATTERNS: list[str] = _SAFETY_CFG.get("blocked_patterns", [])
_INJECTION_PREFIXES: list[str] = _SAFETY_CFG.get("injection_prefixes", [])


def build_graph(
    vectorstore: Chroma,
    alerts_df: pd.DataFrame,
    typologies_df: pd.DataFrame,
) -> Any:
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

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=GEMINI_API_KEY,
    )

    def safety_guard_node(state: AgentState) -> dict:
        query = state.get("query", "")
        safety_meta: Dict[str, Any] = {}

        cleaned = "".join(ch for ch in query if ch in string.printable)
        query_lower = cleaned.lower().strip()

        injection_detected = False
        for prefix in _INJECTION_PREFIXES:
            if prefix in query_lower:
                injection_detected = True
                safety_meta["injection_pattern"] = prefix
                break

        if re.compile(r"[A-Za-z0-9+/=]{50,}").search(cleaned):
            injection_detected = True
            safety_meta["base64_detected"] = True

        pii_detected = pii_redactor.has_pii(cleaned)
        masked_query = pii_redactor.redact(cleaned)

        system_text = _SAFETY_CFG["system"].strip()
        human_text = _SAFETY_CFG.get("human_template", "").strip()
        classification_prompt = system_text + "\n\n" + human_text.format(masked_query=masked_query)

        intent, confidence, unsafe_reason = "policy", 0.5, ""

        if not injection_detected:
            try:
                response = llm.invoke(classification_prompt)
                content = response.content.strip()
                if content.startswith("```"):
                    content = re.sub(r"```(?:json)?\s*", "", content).rstrip("`").strip()
                parsed = json.loads(content)
                intent = parsed.get("intent", "policy")
                confidence = float(parsed.get("confidence", 0.5))
                unsafe_reason = parsed.get("unsafe_reason", "")
            except Exception as exc:
                logger.warning("LLM classification failed: %s", exc)

        safety_meta["llm_intent"] = intent
        safety_meta["llm_confidence"] = confidence
        if unsafe_reason:
            safety_meta["unsafe_reason"] = unsafe_reason

        blocklist_hit = False
        for pattern in _BLOCKED_PATTERNS:
            if pattern in query_lower:
                blocklist_hit = True
                safety_meta["blocklist_hit"] = pattern
                break

        is_unsafe = injection_detected or blocklist_hit or intent in ("unsafe", "off_topic")

        if is_unsafe:
            if injection_detected:
                refusal_reason = f"Prompt injection detected: {safety_meta.get('injection_pattern', 'base64')}"
            elif blocklist_hit:
                refusal_reason = f"Query matches blocked pattern: {safety_meta.get('blocklist_hit', '')}"
            elif intent == "unsafe":
                refusal_reason = f"Classified as unsafe: {unsafe_reason}"
            else:
                refusal_reason = "Query is off-topic for CrimeShield AI"

            return {
                "query": query, "masked_query": masked_query, "pii_detected": pii_detected,
                "is_safe": False, "query_type": "refused", "refusal_reason": refusal_reason,
                "safety_metadata": safety_meta, "confidence_score": confidence,
                "audit_trail": [{"node": "safety_guard", "result": "refused"}],
                "citations": [], "agent_response": "", "plan": [], "current_step": 0,
            }

        return {
            "query": query, "masked_query": masked_query, "pii_detected": pii_detected,
            "is_safe": True, "query_type": intent, "refusal_reason": "",
            "safety_metadata": safety_meta, "confidence_score": confidence,
            "audit_trail": [{"node": "safety_guard", "result": "passed"}],
            "citations": [], "agent_response": "", "plan": [], "current_step": 0,
        }

    def supervisor_node(state: AgentState) -> dict:
        if not state.get("is_safe", False):
            return {}
        current_step = state.get("current_step", 0)
        if current_step > MAX_SUPERVISOR_STEPS:
            return {"agent_response": "Maximum processing steps exceeded."}
        plan = state.get("plan", []) or [{"step": 1, "agent": state.get("query_type", "policy")}]
        return {"plan": plan, "safety_metadata": {**state.get("safety_metadata", {}), "quality_threshold": "comprehensive"}}

    def planner_node(state: AgentState) -> dict:
        if not state.get("is_safe", False):
            return {}
        query_type = state.get("query_type", "policy")
        query = state.get("query", "")
        plan = [{"step": 1, "agent": query_type}]
        if query_type == "sar":
            m = re.search(r"\bALT-[A-Z]?\d+\b", query, re.IGNORECASE)
            if m:
                plan[0]["alert_id"] = m.group()
        return {"plan": plan, "current_step": 0}

    def rag_policy_node(state: AgentState) -> dict:
        query = state.get("masked_query", state.get("query", ""))
        result = rag_agent.invoke(query)
        trail = state.get("audit_trail", [])
        trail.append({"node": "rag_policy", "declined": result.get("declined", False)})
        return {"agent_response": result["answer"], "citations": result["citations"], "confidence_score": result["confidence"], "audit_trail": trail}

    def structured_data_node(state: AgentState) -> dict:
        query = state.get("masked_query", state.get("query", ""))
        result = structured_agent.invoke(query)
        trail = state.get("audit_trail", [])
        trail.append({"node": "structured_data", "tool_calls": result.get("tool_calls", [])})
        return {"agent_response": result["answer"], "confidence_score": 1.0, "audit_trail": trail}

    def sar_drafting_node(state: AgentState) -> dict:
        query = state.get("query", "")
        plan = state.get("plan", [])
        alert_id = plan[0].get("alert_id") if plan else None
        if not alert_id:
            m = re.search(r"\bALT-[A-Z]?\d+\b", query, re.IGNORECASE)
            if m:
                alert_id = m.group()
        alert_record = next((r for r in alert_json_records if r.get("alert_id", "").upper() == (alert_id or "").upper()), None)
        if alert_record is None:
            alert_record = {"alert_id": alert_id or "UNKNOWN", "query_context": query}
        result = sar_agent.invoke(alert_record)
        trail = state.get("audit_trail", [])
        trail.append({"node": "sar_drafting", "alert_id": result.get("alert_id", ""), "grounded": result.get("grounded", False)})
        return {"agent_response": result["narrative"], "confidence_score": 0.9 if result["grounded"] else 0.6, "audit_trail": trail}

    def validator_node(state: AgentState) -> dict:
        response = state.get("agent_response", "")
        query_type = state.get("query_type", "")
        current_step = state.get("current_step", 0)
        errors: list[str] = []

        if not response or len(response) < 50:
            errors.append(f"Response too short ({len(response)} chars)")
        if query_type == "policy" and not state.get("citations", []):
            errors.append("No citations in policy response")
        if query_type == "sar":
            for s in ["## Subject Summary", "## Suspicious Activity Description", "## Supporting Evidence", "## Recommended Action"]:
                if s not in response:
                    errors.append(f"Missing section: {s}")
        if query_type == "structured_data" and not re.search(r"\d+", response):
            errors.append("No numbers in structured data response")

        trail = state.get("audit_trail", [])
        trail.append({"node": "validator", "passed": not errors, "errors": errors})

        if errors and current_step < MAX_SUPERVISOR_STEPS:
            return {"current_step": current_step + 1, "audit_trail": trail,
                    "safety_metadata": {**state.get("safety_metadata", {}), "validation_errors": errors, "retry": True}}

        return {"audit_trail": trail, "safety_metadata": {**state.get("safety_metadata", {}), "validated": not errors, "validation_errors": errors}}

    def response_merger_node(state: AgentState) -> dict:
        response = pii_redactor.redact(state.get("agent_response", ""))
        citations = state.get("citations", [])
        if citations:
            response += "\n\n---\n**Sources:**\n"
            for c in citations:
                response += f"- {c.get('source_section', 'Unknown')} (chunk: {c.get('chunk_id', '')}, relevance: {c.get('relevance_score', 0.0):.4f})\n"
        score = state.get("confidence_score", 0.0)
        label = "HIGH" if score >= 0.8 else ("MEDIUM" if score >= 0.5 else "LOW")
        response += f"\nConfidence: {label}"
        final_state = {**state, "agent_response": response}
        audit_logger.log(final_state)
        trail = state.get("audit_trail", [])
        trail.append({"node": "response_merger", "audit_written": True})
        return {"agent_response": response, "audit_trail": trail}

    def refused_node(state: AgentState) -> dict:
        msg = _SAFETY_CFG.get("decline_message", "Unable to assist. Query logged.").strip()
        trail = state.get("audit_trail", [])
        trail.append({"node": "refused", "logged": True})
        audit_logger.log({**state, "agent_response": msg})
        return {"agent_response": msg, "audit_trail": trail}

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
    graph.add_conditional_edges("safety_guard", lambda s: "refused" if s.get("query_type") == "refused" else "supervisor", {"refused": "refused", "supervisor": "supervisor"})
    graph.add_edge("supervisor", "planner")
    graph.add_conditional_edges("planner", lambda s: "structured_data" if s.get("query_type") == "structured_data" else ("sar_drafting" if s.get("query_type") == "sar" else "rag_policy"), {"rag_policy": "rag_policy", "structured_data": "structured_data", "sar_drafting": "sar_drafting"})
    graph.add_edge("rag_policy", "validator")
    graph.add_edge("structured_data", "validator")
    graph.add_edge("sar_drafting", "validator")
    graph.add_conditional_edges("validator", lambda s: "planner" if s.get("safety_metadata", {}).get("retry", False) else "response_merger", {"planner": "planner", "response_merger": "response_merger"})
    graph.add_edge("response_merger", END)
    graph.add_edge("refused", END)

    return graph.compile()
