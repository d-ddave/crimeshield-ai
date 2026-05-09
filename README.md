# CrimeShield AI

**Multi-Agent Economic Crime Intelligence Assistant**

CrimeShield AI is a production-grade, multi-agent system built with LangGraph, LangChain, Google Gemini, and ChromaDB that provides:
- **Policy Q&A** — RAG-powered answers from AML/CFT regulatory documents
- **Structured Data Analysis** — Tool-calling agent querying alerts and typology tables
- **SAR Drafting** — Automated Suspicious Activity Report generation from alert records
- **Safety & PII Controls** — 5-stage query safety pipeline with PII masking

---

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-community langchain-google-genai langgraph chromadb google-generativeai pandas python-dotenv pyyaml fastapi uvicorn
```

### 2. Configure Environment

```bash
cp crimeshield/.env.example crimeshield/.env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your-ai-studio-key-here
```

### 3. Run as Web Server

```bash
venv/bin/python -m uvicorn crimeshield.main:app --reload --port 8000
```

### 4. Run CLI

```bash
venv/bin/python crimeshield/main.py "What are the red flags for Trade-Based Money Laundering?"
venv/bin/python crimeshield/main.py --health
venv/bin/python crimeshield/main.py --diagram
```

### 5. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the red flags for TBML?"}'
```

---

## Architecture Overview

CrimeShield uses an **8-node LangGraph StateGraph** with conditional routing:

```
START → safety_guard → [refused | supervisor] → planner →
  [rag_policy | structured_data | sar_drafting] → validator →
  [planner (retry) | response_merger] → END
```

### Node Descriptions

| Node | Purpose |
|------|---------|
| `safety_guard` | 5-stage safety pipeline: preprocessing, PII masking, LLM classification, blocklist check, decision |
| `supervisor` | Step limit enforcement, plan validation, quality thresholds |
| `planner` | Routing plan construction based on query type, alert ID extraction |
| `rag_policy` | RAG retrieval from policy corpus with similarity threshold filtering |
| `structured_data` | Tool-calling agent for alerts/typology DataFrame queries |
| `sar_drafting` | Structured 4-section SAR narrative generation from alert records |
| `validator` | Deterministic output quality checks (no LLM) with retry logic |
| `response_merger` | PII redaction, citation assembly, confidence scoring, audit logging |
| `refused` | Professional refusal with audit trail for unsafe/off-topic queries |

---

## Stack

- **LLM**: Google Gemini 2.0 Flash via `langchain-google-genai`
- **Embeddings**: `gemini-embedding-001` (3072-dim)
- **Vector Store**: ChromaDB (persistent)
- **Orchestration**: LangGraph
- **API**: FastAPI + Uvicorn
- **Data**: pandas

---

## Project Structure

```
crimeshield/
├── .env.example          # Environment variable template
├── config.py             # Centralised configuration
├── main.py               # FastAPI app + CLI entry point
├── agents/
│   ├── rag.py            # RAG policy retrieval agent
│   ├── structured_data.py # Alerts/typology tool-calling agent
│   └── sar.py            # SAR narrative drafting agent
├── graph/
│   ├── state.py          # AgentState TypedDict
│   └── graph.py          # LangGraph StateGraph (8 nodes)
├── pipeline/
│   ├── build_store.py    # ChromaDB vector store builder
│   └── load_data.py      # CSV/JSON data loader
├── prompts/
│   ├── rag_policy.yaml
│   ├── sar_drafting.yaml
│   ├── structured_data.yaml
│   ├── safety_guard.yaml
│   └── planner.yaml
└── utils/
    ├── pii.py            # PII detection and redaction
    ├── audit.py          # Append-only JSON audit logger
    └── prompts.py        # YAML prompt loader
```

## Data Files

```
data/
├── policy_corpus.txt         # 3 concatenated regulatory docs
├── alerts.csv                # Alert records with risk bands
├── typology_thresholds.csv   # NCA typology thresholds
└── alert.json                # Synthetic alert records
```

---

## License

Internal use only.
