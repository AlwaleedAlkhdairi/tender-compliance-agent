# Rubric-to-Evidence Map

Every graded rubric item mapped to the implementation evidence that proves it.

## 3.1 Agentic Architecture and Multi-Agent Orchestration (25)

| Rubric evidence | Where to see it |
|---|---|
| Clear business problem, target user, value, justification for agentic system | `README.md` §1–2; presentation |
| Supervisor + ≥2 meaningful specialists, non-duplicated responsibilities | `src/agents/` — supervisor + 3 specialists; About tab diagram |
| Clear agent contracts (responsibility, input, permitted actions, output, completion evidence) | CONTRACT docstrings in each `src/agents/*.py`; README contract table |
| Delegation, routing, reasoning pattern visible in implementation | `supervisor.route` (+ guardrails), conditional edges in `src/graph/workflow.py`; live timeline in the app |
| Structured state transitions / information passed between agents | `src/graph/state.py` + Pydantic hand-offs in `src/schemas.py` |
| Clear stopping condition + combined final result | `finalize` node + `MAX_SUPERVISOR_STEPS` bound (`src/graph/nodes.py`); `supervisor.aggregate` → `FinalReport` |

Reasoning patterns: Plan-and-Execute (supervisor) + ReAct (specialists) —
see `docs/architecture.md` §3.

## 3.2 State, Memory and Agentic RAG (15)

| Rubric evidence | Where to see it |
|---|---|
| Explicit structured state for the active task | `TenderState` in `src/graph/state.py` |
| Short-term memory preserves case/conversation info | `CaseMemory` (`src/memory/store.py`) + Case Q&A tab |
| Knowledge source prepared, embeddings + vector store | `data/knowledge/`, `data/sample_cases/`, `scripts/ingest_knowledge.py`, `src/rag/vector_store.py` (Chroma) |
| Retrieval results include source metadata | `RetrievedChunk` (file, section, chunk id) in `src/rag/retriever.py`; evidence events in the timeline; evidence trail tab |
| Retrieved evidence used in decisions/final result | Compliance statuses cite bid quotes + sources (`EvidenceItem.source`); guidance passages drive material-deviation judgements; sources appear in the final report + CSV |

## 3.3 Agent Tools and Actions (10)

| Rubric evidence | Where to see it |
|---|---|
| ≥2 meaningful tools (RAG retrieval excluded) | `build_compliance_matrix`, `calculate_weighted_score`, `export_comparison_csv` (`src/tools/`) |
| Agent selects the appropriate tool by request/state | Specialists choose tools inside their ReAct loops; timeline shows the selection (e.g. evidence agent consults guidance only when a deviation question arises) |
| Inputs validated, outputs structured | Pydantic models per tool + `execute_tool` in `src/tools/registry.py`; `tests/test_tools.py` |
| Tool results affect the next step / final result | Matrix disqualifications drive the ranking; scoring tool output is the ranking; CSV path returned to the UI |
| Tool errors handled without crashing | `is_error` tool results; registry try/except; `tests/test_tools.py::TestRegistry` |

## 3.4 Working Application and System Integration (10)

| Rubric evidence | Where to see it |
|---|---|
| Complete input → final result path | `app.py` Run analysis tab; mocked e2e proof in `tests/test_workflow.py` |
| Clear, organized interface | Tabs: run / Q&A / about; results tabs (requirements, matrix, scores, evidence, timeline) |
| Shows workflow stages (active agent, evidence, tools, status) | Live status box + event timeline; evidence trail with sources |
| Missing input / evidence / tool failure → clear response, no crash / false success | Disabled run + guidance without key; KB warning; failed cases show reasons + partial results; `finalize` never fabricates a report |

## 3.5 Presentation (5)

Suggested demo script: problem → architecture (About tab) → live run on all
three suppliers → show BetaGrid's disqualification + evidence sources →
ranking + CSV → Case Q&A question → failure demo (remove `.env` key or ask
with one supplier) → limitation + future improvement (README §10).

## 3.6 Documentation and GitHub Repository (5)

| Rubric evidence | Where to see it |
|---|---|
| Complete organized code, dependency file, meaningful history | repo tree; `requirements.txt`; incremental conventional commits |
| README explains idea, users, value, features, architecture, course, SDAIA context | `README.md` (all 25 required items) |
| Setup/run instructions work from clean environment | README §6 (verified in a clean container) |
| `.env.example`, no secrets committed | `.env.example`; `.gitignore` covers `.env`, `data/chroma/`, `outputs/` |
| Technical docs, data statement, limitations, screenshots, team, SDAIA link | `docs/architecture.md`, README §§8–10, `docs/screenshots/`, team table, SDAIA GitHub link |
