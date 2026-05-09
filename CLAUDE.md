# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the System

```bash
# Activate virtualenv first
source venv/bin/activate

# Run a query
python crimeshield/main.py "What are the red flags for Trade-Based Money Laundering?"

# Print the Mermaid architecture diagram
python crimeshield/main.py --diagram
```

Requires `OPENAI_API_KEY` in a `.env` file at the project root (see `crimeshield/.env.example`).

The ChromaDB vector store is built automatically on first run and persisted to `chroma_store/`. Subsequent runs skip the rebuild. To force a rebuild, delete the `chroma_store/` directory.

## Architecture

CrimeShield AI is an 8-node **LangGraph StateGraph** that processes financial crime queries end-to-end:

```
START → safety_guard → [refused | supervisor] → planner →
  [rag_policy | structured_data | sar_drafting] → validator →
  [planner (retry) | response_merger] → END
```

**Framework split**: LangGraph owns orchestration (routing, retries, state flow); LangChain owns agent implementation. Each agent (`RAGPolicyAgent`, `StructuredDataAgent`, `SARDraftingAgent`) can be instantiated and called independently of the graph.

**Shared state**: `crimeshield/graph/state.py` defines `AgentState` (TypedDict). Every node receives the full state and returns only the keys it modifies.

### Key nodes

| Node | File | Role |
|------|------|------|
| `safety_guard` | `graph/graph.py` | 5-stage pipeline: control-char stripping → PII masking → GPT-4o intent classification → regex blocklist → decision |
| `planner` | `graph/graph.py` | Sets `plan` and extracts alert IDs from query for SAR routing |
| `rag_policy` | `agents/rag.py` | ChromaDB similarity retrieval + RetrievalQA chain; filters chunks below `SIMILARITY_THRESHOLD` |
| `structured_data` | `agents/structured_data.py` | OpenAI function-calling agent with two tools: `query_alerts_table` and `query_typology_table` |
| `sar_drafting` | `agents/sar.py` | Generates 4-section SAR narratives from `alert.json` records; validates grounding against alert amounts |
| `validator` | `graph/graph.py` | Deterministic (no LLM): checks response length, citation presence (policy), 4-section headers (SAR), numeric content (structured). Triggers retry by setting `safety_metadata["retry"] = True` |
| `response_merger` | `graph/graph.py` | Output PII redaction, citation assembly, confidence label, audit log write |

### Retry loop

`validator` → `planner` → agent → `validator` (up to `MAX_SUPERVISOR_STEPS = 3` times). Controlled by `safety_metadata["retry"]`.

## Configuration

All settings are in `crimeshield/config.py`. Key values:

- `LLM_MODEL = "gpt-4o"` / `EMBEDDING_MODEL = "text-embedding-3-small"`
- `CHROMA_PERSIST_DIR` → `chroma_store/` at project root
- `SIMILARITY_THRESHOLD = 0.35` — chunks below this are excluded from RAG context
- `MAX_SUPERVISOR_STEPS = 3` — retry cap
- `CONFIDENCE_THRESHOLD = 0.6`

## Data Files

```
data/
├── policy_corpus.txt         # 3 regulatory docs separated by === headers
├── alerts.csv                # Alert records (columns: alert_id, risk_band, status, alert_type, amount_gbp)
├── typology_thresholds.csv   # NCA typology codes and thresholds
└── alert.json                # 3 synthetic alert records for SAR drafting
```

The corpus is split by `===` section separators, then chunked with 500-char / 100-char overlap within each section. Each chunk carries `source_section`, `chunk_id`, and `document` metadata used for citations.

## Utilities

- **`PIIRedactor`** (`utils/pii.py`): regex-based, no external NLP. Used as both input pre-processor (in `safety_guard`) and output post-processor (in `response_merger`). Redacts sort codes, account numbers, GBP amounts, emails, UK postcodes, customer IDs, alert IDs, analyst names.
- **`AuditLogger`** (`utils/audit.py`): append-only JSON-lines writer to `audit.log` at project root.

## Query Types

The safety guard classifies every query into one of: `policy`, `structured_data`, `sar`, `unsafe`, `off_topic`. The `planner` then routes based on `query_type`. Alert IDs matching `ALT-[A-Z]?\d+` are extracted from the query and used for SAR JSON lookup.
