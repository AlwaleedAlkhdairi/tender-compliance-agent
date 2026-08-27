# Technical Architecture

This document explains how the system is built and why. Read the README
first for the business context; this file goes one level deeper.

## 1. High-Level Flow

```text
Streamlit UI (app.py)
      │  tender + suppliers + optional instructions
      ▼
initial_state()  ──►  LangGraph StateGraph (src/graph/workflow.py)
      │
      │   START ─► supervisor ─► (conditional edge on state.next_agent)
      │                ▲   │
      │                │   ├─► requirement_specialist ─► supervisor
      │                │   ├─► evidence_specialist    ─► supervisor
      │                │   ├─► comparison_specialist  ─► supervisor
      │                │   └─► finalize ─► END
      ▼
final TenderState  ──►  results UI + CaseMemory (follow-up Q&A)
```

The supervisor re-evaluates the case after every specialist. The conditional
edge reads `state["next_agent"]`, which the supervisor writes from a
structured `RouteDecision`.

## 2. State Design (`src/graph/state.py`)

`TenderState` is a `TypedDict` — one record per case, updated by partial node
returns. Two fields use an `operator.add` reducer so they accumulate instead
of being overwritten:

- `events`: the observable timeline (agent started/finished, reasoning
  summaries, tool calls and results, retrieved evidence, routing decisions,
  errors). Events are also forwarded live to the UI through
  `configurable.live_emit` while a node is still running.
- `errors`: user-facing failure reasons.

The structured hand-offs are stored as `model_dump()` dicts of the Pydantic
contracts in `src/schemas.py` (`RequirementSet`, `EvidenceReport`,
`ComparisonResult`, `FinalReport`) — so state stays JSON-serializable while
every producer/consumer validates through the schema.

## 3. Reasoning Patterns and Where They Appear

| Pattern | Where | Evidence in code |
|---|---|---|
| Plan-and-Execute | Supervisor plans once (`SupervisorPlan`), then executes step-by-step with `RouteDecision`s | `src/agents/supervisor.py` (`plan`, `route`) |
| ReAct | Each specialist: Thought (text) → Action (tool call) → Observation (tool result), repeated | `run_tool_loop` in `src/llm.py`; visible as timeline events |
| Deterministic guardrails | Invalid routing corrected in code; loop bounds; disqualification math in tools | `VALID_PREREQS` in supervisor; `MAX_SUPERVISOR_STEPS`, `MAX_AGENT_TURNS` in `src/config.py` |

The split of authority is deliberate: **agents make judgements, tools do
arithmetic**. The model never computes weighted totals — `calculate_weighted_score`
does, so rankings are reproducible and auditable.

## 4. LLM Integration (`src/llm.py`, `src/gemini_llm.py`)

- Two interchangeable providers behind the same two primitives
  (`run_tool_loop`, `structured_call`): the official `anthropic` SDK (v1.x,
  default `claude-opus-5`) and Google's `google-genai` SDK (default
  `gemini-3.7-flash`, free tier). `config.llm_provider()` picks by
  `LLM_PROVIDER` or by which key is configured; agents, tools, graph and UI
  are provider-agnostic.
- The Gemini backend converts the registry's Pydantic tool schemas to
  Gemini-compatible declarations ($refs inlined, unsupported keys removed),
  retries 429/500/503 with backoff, falls back to `GEMINI_FALLBACK_MODEL`
  when the primary is overloaded, and renders tool transcripts as text for
  structured-output calls (Gemini does not combine function calling with
  `response_schema`).
- `run_tool_loop`: a bounded manual tool-use loop. Handles `tool_use` /
  `end_turn` / `max_tokens` / `refusal` stop reasons; executes tools through
  the validated registry; failed tools return `is_error` tool results so the
  agent can self-correct.
- `structured_call`: `client.messages.parse(..., output_format=Model)` —
  guaranteed-schema outputs for plans, routing decisions, findings and the
  final report.
- All API failures are translated into `AgentError` with a human-readable
  message (invalid key, rate limit, network, refusal), which nodes catch and
  turn into a clean *failed* case — the app never shows a stack trace.

## 5. RAG Pipeline (`src/rag/`)

**Preparation** (offline, `scripts/ingest_knowledge.py`):

```text
markdown docs ─► loaders.py (clean, type, extract supplier)
             ─► chunking.py (split on ##/### headings; oversized sections
                             split on paragraphs with overlap; heading path
                             prepended to text and kept as metadata)
             ─► vector_store.py (Chroma PersistentClient, local ONNX MiniLM
                             embeddings, two collections)
```

Metadata per chunk: `source_file`, `section` (heading path), `doc_type`
(tender / criteria / guidance / bid), `supplier`, `title`; ids are stable
(`<file>::chunk-NNN`).

**Runtime** (agentic): retrieval is exposed as *tools* (`search_knowledge`,
`search_bids` with optional supplier scoping), so the agent chooses when to
retrieve and with what query. Results return text + full metadata; the UI
shows them as the evidence trail, and `EvidenceItem.source` carries them into
the final report.

Two collections keep tender knowledge and bids separate so the Requirement
Specialist physically cannot "retrieve" bid content, and bid searches can be
scoped per supplier.

## 6. Tool Layer (`src/tools/`)

`registry.py` is the single dispatch point: name → Pydantic input model +
handler. `execute_tool` validates first (`ValidationError` → structured error
result), then executes in a try/except (runtime failure → structured error
result). This gives all five tools the same guarantees:

- inputs validated (weights sum to 100, scores 0–100, statuses enumerated,
  findings reference known requirement ids…),
- structured JSON outputs,
- errors that inform the agent instead of crashing the app.

## 7. Memory Layers

| Layer | Lifetime | Contents | Where |
|---|---|---|---|
| Workflow state | one run | progress + structured artifacts + events | `src/graph/state.py` |
| Short-term case memory | the user's session | distilled facts/findings + Q&A conversation | `src/memory/store.py`, held in `st.session_state` |
| Long-term knowledge | across cases | embedded document chunks | Chroma store (`data/chroma/`) |

`build_case_memory(final_state)` distills a finished run; `case_qa.answer`
injects `CaseMemory.render_context()` into the assistant's system prompt and
lets it choose retrieval tools for anything memory doesn't answer.

## 8. Error Handling Strategy

| Failure | Behavior |
|---|---|
| Missing API key | Sidebar status + run disabled + instructions (no call attempted) |
| Knowledge base absent | Warning + one-click build; agents never query an empty store |
| Invalid tool input from the model | `is_error` tool result → agent corrects itself |
| Tool runtime failure | Same structured-error path; surfaced in the timeline |
| API auth/rate/network error, refusal | `AgentError` → case closes as *failed* with the reason; partial artifacts still shown |
| Runaway delegation | `MAX_SUPERVISOR_STEPS` forces finalization; specialist loops capped at `MAX_AGENT_TURNS` |
| No comparison produced | `finalize` refuses to fabricate a report — status *failed*, never false success |

## 9. Testing Strategy (`tests/`)

The suite runs fully offline (no API key):

- `test_tools.py` — tool logic + validation rules + registry dispatch.
- `test_chunking.py` — loaders, supplier extraction, chunk metadata and ids.
- `test_workflow.py` — the compiled LangGraph runs end-to-end with the LLM
  faked but **real tools, real state and real routing**: a complete case, a
  specialist-failure case, a hostile-supervisor case (guardrails + stopping
  condition), and case-memory distillation.

## 10. Notable Design Decisions

1. **Anthropic SDK directly instead of a chat-model wrapper** — full control
   over the tool loop, stop reasons and structured outputs; fewer layers for
   students to explain.
2. **Judgement in agents, arithmetic in tools** — reproducible scores,
   auditable rankings.
3. **Events as first-class state** — the same event stream powers the live
   UI, the persisted timeline and debugging.
4. **Local embeddings** — one API key total; the RAG layer works offline.
5. **Honest failure** — a case without a trustworthy comparison finishes
   *failed*, never with an invented report.
