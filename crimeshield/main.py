#!/usr/bin/env python3
"""
CrimeShield AI — Main Entry Point & FastAPI Server.

Usage:
    # CLI Mode
    python main.py "<query>"
    python main.py --diagram        # Print Mermaid diagram and exit
    python main.py --health         # Run system health check

    # Web Server Mode
    uvicorn crimeshield.main:app --reload
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Ensure the project root is on the import path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from crimeshield.config import GEMINI_API_KEY, AUDIT_LOG_PATH
from crimeshield.graph.graph import build_graph
from crimeshield.graph.state import AgentState
from crimeshield.pipeline.build_store import build_vector_store
from crimeshield.pipeline.load_data import DataLoader

# ====================================================================
# CORE PIPELINE LOGIC
# ====================================================================

def _setup_pipeline():
    """Build vector store, load data, compile graph."""
    vectorstore = build_vector_store()
    loader = DataLoader()
    alerts_df = loader.load_alerts()
    typologies_df = loader.load_typologies()
    compiled_graph = build_graph(vectorstore, alerts_df, typologies_df)
    return compiled_graph, vectorstore, alerts_df, typologies_df

def _run_query(compiled_graph, query: str) -> dict:
    """Invoke the graph with a query."""
    initial_state: AgentState = {
        "query": query,
        "query_type": "",
        "agent_response": "",
        "citations": [],
        "is_safe": True,
        "confidence_score": 0.0,
        "plan": [],
        "current_step": 0,
        "pii_detected": False,
        "masked_query": "",
        "safety_metadata": {},
        "audit_trail": [],
        "refusal_reason": "",
    }
    return compiled_graph.invoke(initial_state)

# ====================================================================
# FASTAPI APPLICATION
# ====================================================================

app = FastAPI(title="CrimeShield AI API", version="1.0.0")

# Global variables for the shared state (lazy loaded)
_graph = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    query_type: str
    agent_response: str
    citations: List[dict]
    confidence_score: float
    pii_detected: bool
    refusal_reason: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "online", "service": "CrimeShield AI"}

@app.post("/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    global _graph
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    if _graph is None:
        _graph, *_ = _setup_pipeline()
    
    final_state = _run_query(_graph, request.query)
    
    return {
        "query": request.query,
        "query_type": final_state.get("query_type", "unknown"),
        "agent_response": final_state.get("agent_response", ""),
        "citations": final_state.get("citations", []),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "pii_detected": final_state.get("pii_detected", False),
        "refusal_reason": final_state.get("refusal_reason") or None
    }

# ====================================================================
# CLI COMMANDS
# ====================================================================

def _print_divider(char: str = "─", width: int = 80) -> None:
    print(char * width)

def _print_header(title: str) -> None:
    _print_divider("═")
    print(f"  🛡️  CrimeShield AI — {title}")
    _print_divider("═")

def _format_result(final_state: dict, query: str) -> str:
    lines = [f"📝 Query: {query}", f"🏷️  Query Type Detected: {final_state.get('query_type', 'unknown')}", ""]
    lines.append("📋 Agent Response:")
    lines.append(final_state.get("agent_response", "No response generated."))
    
    citations = final_state.get("citations", [])
    if citations:
        lines.append(f"\n📚 Citations ({len(citations)}):")
        for cite in citations:
            section = cite.get("source_section", "Unknown")
            score = cite.get("relevance_score", 0.0)
            lines.append(f"   • {section} (relevance: {score:.4f})")

    score = final_state.get("confidence_score", 0.0)
    emoji, label = ("🟢", "HIGH") if score >= 0.8 else (("🟡", "MEDIUM") if score >= 0.5 else ("🔴", "LOW"))
    lines.append(f"\n{emoji} Confidence Score: {score:.4f} ({label})")
    lines.append(f"🔒 PII Detected: {'Yes' if final_state.get('pii_detected', False) else 'No'}")
    
    refusal = final_state.get("refusal_reason", "")
    if refusal: lines.append(f"⚠️  Refusal Reason: {refusal}")
    
    return "\n".join(lines)

def _cmd_health():
    _print_header("System Health Check")
    all_ok = True
    
    checks = [
        ("Gemini API key", bool(GEMINI_API_KEY)),
        ("Vector store", lambda: build_vector_store()._collection.count() > 0),
        ("Alerts Data", lambda: len(DataLoader().load_alerts()) > 0),
        ("Typologies Data", lambda: len(DataLoader().load_typologies()) > 0),
    ]
    
    for label, check in checks:
        print(f"  {label:.<25}", end=" ")
        try:
            passed = check() if callable(check) else check
            print("[OK]" if passed else "[FAIL]")
            if not passed: all_ok = False
        except Exception as e:
            print(f"[FAIL] — {e}")
            all_ok = False
            
    _print_divider()
    print("  ✅  All checks passed" if all_ok else "  ❌  Some checks failed")
    sys.exit(0 if all_ok else 1)

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<query>\" [--diagram|--health|--run-all|--tail-log]")
        sys.exit(1)

    flag = sys.argv[1]
    
    if flag == "--health":
        _cmd_health()
    elif flag == "--diagram":
        graph, *_ = _setup_pipeline()
        print(graph.get_graph().draw_mermaid())
    elif not flag.startswith("--"):
        if not GEMINI_API_KEY:
            print("❌ ERROR: GEMINI_API_KEY not set."); sys.exit(1)
        _print_header("Processing Query")
        graph, *_ = _setup_pipeline()
        final_state = _run_query(graph, flag)
        print(_format_result(final_state, flag))
    else:
        print(f"Unsupported flag: {flag}")
        sys.exit(1)

if __name__ == "__main__":
    main()
