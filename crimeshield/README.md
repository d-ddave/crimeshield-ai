# CrimeShield AI

**Multi-Agent Economic Crime Intelligence Assistant for Lloyds Bank**

CrimeShield AI is a production-grade, multi-agent system built with LangGraph, LangChain, and ChromaDB that provides:
- **Policy Q&A** — RAG-powered answers from AML/CFT regulatory documents
- **Structured Data Analysis** — Tool-calling agent querying alerts and typology tables
- **SAR Drafting** — Automated Suspicious Activity Report generation from alert records
- **Safety & PII Controls** — 5-stage query safety pipeline with PII masking

---

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-community langchain-openai langgraph chromadb openai pandas python-dotenv tiktoken
```

### 2. Configure Environment

```bash
cp crimeshield/.env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
# Run a query
python crimeshield/main.py "What are the red flags for Trade-Based Money Laundering under current JMLSG guidance?"

# Print the architecture diagram (Mermaid)
python crimeshield/main.py --diagram
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

## Design Decisions

### 1. Chunking Strategy

The policy corpus is split using a **two-stage chunking approach**. First, the document is segmented by section boundaries (the `===` header lines that separate the three regulatory documents). Then, within each section, LangChain's `RecursiveCharacterTextSplitter` is applied with a chunk size of 500 characters and 100-character overlap. This ensures that each chunk retains its section-level metadata (e.g., "JMLSG TBML Guidance" or "SAR Filing Requirements"), enabling precise citation back to the source regulatory document.

This design avoids the common failure mode of naive whole-document chunking where a single chunk might span two unrelated regulatory sections, producing misleading retrieval results. The 100-character overlap ensures that key phrases at section boundaries are not lost. Additionally, each chunk carries `source_section`, `chunk_id`, and `document` metadata fields, enabling the RAG agent to provide auditable, regulation-specific citations in its responses — a critical requirement for compliance applications where traceability to source guidance is mandatory.

### 2. Framework Division: LangGraph + LangChain

The system uses **LangGraph for orchestration** and **LangChain for agent implementation**, maintaining a clean separation of concerns. LangGraph manages the stateful workflow — routing queries through safety checks, agent invocation, validation, and response assembly — while LangChain provides the agent primitives (retrieval chains, tool-calling agents, prompt templates). This division means each agent (RAG, Structured Data, SAR) is independently testable: you can instantiate `RAGPolicyAgent(vectorstore)` or `StructuredDataAgent(alerts_df, typologies_df)` and call `.invoke()` directly without the graph.

The graph itself uses conditional edges for all branching decisions, making the control flow explicit and auditable. The `validator_node` implements a retry loop — if validation fails and the step counter hasn't exceeded `MAX_SUPERVISOR_STEPS`, the query is re-routed through the planner for another attempt. This provides resilience against transient LLM quality issues without unbounded looping. The compiled graph can also export a Mermaid diagram (`--diagram` flag), making the architecture self-documenting for regulatory and security review.

### 3. Safety Approach

The safety system implements a **5-stage defence-in-depth pipeline** entirely within the `safety_guard_node`. Stage 1 strips control characters and detects prompt injection patterns (common prefixes like "ignore previous", "system:", base64 payloads). Stage 2 applies PII masking before any text reaches the LLM. Stage 3 uses a single GPT-4o classification call to determine query intent (`policy`, `structured_data`, `sar`, `unsafe`, `off_topic`) with a confidence score. Stage 4 applies a policy blocklist of 25+ patterns covering money laundering evasion, structuring advice, sanctions circumvention, and jailbreak attempts. Stage 5 combines all signals into a binary safe/unsafe decision.

This layered approach ensures that even if the LLM classification misses an unsafe query (Stage 3), the regex-based blocklist (Stage 4) provides a deterministic fallback. The PIIRedactor operates as both a pre-processor (masking input before LLM calls) and post-processor (redacting any PII that appears in LLM outputs), using the same class instance in two different graph nodes. All refused queries are logged to the append-only audit log with the specific refusal reason and matched pattern, providing a complete forensic trail for compliance review.

---

## Sample Outputs

### Query 1: Policy (TBML Red Flags)
```
🏷️  Query Type Detected: policy

📋 Agent Response:
Based on the JMLSG guidance, the key red flags for Trade-Based Money Laundering include:

1. Invoice and Pricing Anomalies: Significant over-invoicing or under-invoicing, multiple invoices
   for the same goods, invoices from shell companies with no trading history...
2. Counterparty and Jurisdiction Risks: Transactions with counterparties in high-risk jurisdictions,
   newly incorporated companies with no trading history...
3. Shipping and Logistics Red Flags: Circuitous shipping routes, discrepancies between documents...
4. Financial Transaction Patterns: Rapid movement of funds, payments by unnamed third parties...
5. Customer Behaviour Indicators: Unable to provide documentation, trade volumes inconsistent
   with business capacity...

Sources:
- JMLSG TBML Guidance (chunk: jmlsg_guidance_0, relevance: 0.8721)
- JMLSG TBML Guidance (chunk: jmlsg_guidance_2, relevance: 0.8456)

🟢 Confidence Score: 0.8234 (HIGH)
🔒 PII Detected: No
📝 Audit log entry written: Yes
```

### Query 2: Structured Data (HIGH Risk Alert Analysis)
```
🏷️  Query Type Detected: structured_data

📋 Agent Response:
Based on the alerts data, there are currently X HIGH risk alerts in UNDER_REVIEW status
with a combined transaction value of £X. The breakdown shows...

TOOL CALL: query_alerts_table(risk_band=HIGH, status=UNDER_REVIEW, alert_type=)

🟢 Confidence Score: 1.0000 (HIGH)
🔒 PII Detected: No
📝 Audit log entry written: Yes
```

### Query 3: SAR Drafting (ALT-A4821)
```
🏷️  Query Type Detected: sar

📋 Agent Response:
## Subject Summary
The subject is [CUSTOMER-ID], a sole trader operating a freelance graphic design
and marketing consultancy...

## Suspicious Activity Description
Between 2024-03-05 and 2024-03-11, the customer received three payments totalling
[AMOUNT-GBP] from three separate limited companies...

## Supporting Evidence
- Three invoices from dormant companies: Meridian Ventures Consulting Ltd,
  Northbridge Advisory Partners Ltd, Elara Professional Services Ltd
- All counterparties incorporated within 18 months, no accounts filed...

## Recommended Action
File SAR with NCA under typology NCA-001 (Invoice Fraud). Place account under
enhanced monitoring. Suspend outbound transfers pending MLRO review...

Confidence: HIGH
🟢 Confidence Score: 0.9000 (HIGH)
🔒 PII Detected: Yes
📝 Audit log entry written: Yes
```

### Query 4: Refused (Unsafe Query)
```
🏷️  Query Type Detected: refused

📋 Agent Response:
I'm unable to assist with that request. CrimeShield AI is designed to support
economic crime prevention activities only. This query has been logged.

🔴 Confidence Score: 0.0000 (LOW)
🔒 PII Detected: No
📝 Audit log entry written: Yes
⚠️  Refusal Reason: Query matches blocked pattern: without triggering
```

---

## Project Structure

```
crimeshield/
├── .env.example          # Environment variable template
├── config.py             # Centralised configuration
├── main.py               # CLI entry point
├── agents/
│   ├── __init__.py
│   ├── rag.py            # RAG policy retrieval agent
│   ├── structured_data.py # Alerts/typology tool-calling agent
│   └── sar.py            # SAR narrative drafting agent
├── graph/
│   ├── __init__.py
│   ├── state.py          # AgentState TypedDict
│   └── graph.py          # LangGraph StateGraph (8 nodes)
├── pipeline/
│   ├── __init__.py
│   ├── build_store.py    # ChromaDB vector store builder
│   └── load_data.py      # CSV/JSON data loader
├── utils/
│   ├── __init__.py
│   ├── pii.py            # PII detection and redaction
│   └── audit.py          # Append-only JSON audit logger
└── README.md
```

## Data Files (pre-existing)

```
data/
├── policy_corpus.txt         # 3 concatenated regulatory docs
├── alerts.csv                # Alert records with risk bands
├── typology_thresholds.csv   # NCA typology thresholds
└── alert.json                # 3 synthetic alert records
```

---

## Audit Log

Every invocation writes a JSON line to `audit.log` with:
- Timestamp (ISO 8601)
- Query type and safety classification
- Confidence score
- PII detection status
- Response length and citation count
- Refusal reason (if applicable)

---

## License

Internal use — Lloyds Banking Group
