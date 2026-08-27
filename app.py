"""Tender Compliance & Supplier Comparison — Streamlit application.

Runs the complete multi-agent workflow (supervisor + three specialists) over
the tender knowledge base and supplier bids, shows every stage live, and
presents the compliance matrix, weighted ranking, evidence trail and final
recommendation. A Case Q&A tab answers follow-up questions from short-term
case memory plus retrieval.

Developed for "Advanced Agentic AI Systems Engineering" at SDAIA Academy.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.graph.workflow import run_case
from src.llm import AgentError
from src.memory.store import CaseMemory, build_case_memory
from src.rag.loaders import list_suppliers, list_tenders
from src.rag.vector_store import index_ready, rebuild_index
from src.services.charts import criterion_breakdown_chart, weighted_total_chart

st.set_page_config(
    page_title="Tender Compliance Agent",
    page_icon="📑",
    layout="wide",
)

AGENT_LABELS = {
    "supervisor": "🧭 Supervisor",
    "requirement_specialist": "📋 Requirement Specialist",
    "evidence_specialist": "🔎 Evidence Specialist",
    "comparison_specialist": "⚖️ Comparison Specialist",
    "case_qa": "💬 Case Assistant",
}
KIND_ICONS = {
    "started": "▶️", "reasoning": "💭", "routing": "🔀", "tool_call": "🔧",
    "tool_result": "📦", "evidence": "📄", "finished": "✅", "error": "⚠️",
}
STATUS_LABELS = {
    "compliant": "✅ compliant",
    "partial": "🟡 partial",
    "missing": "❌ missing",
    "not_assessed": "⚪ not assessed",
}


# ---------------------------------------------------------------------------
# Sidebar: environment status and knowledge-base management
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.title("📑 Tender Compliance Agent")
    st.sidebar.caption(
        "Multi-agent tender evaluation — SDAIA Academy · "
        "Advanced Agentic AI Systems Engineering"
    )

    key_ok = config.api_key_present()
    kb_ok = index_ready()
    provider = config.llm_provider()
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"

    st.sidebar.subheader("Environment")
    st.sidebar.markdown(
        f"{'🟢' if key_ok else '🔴'} {provider.capitalize()} API key "
        f"{'configured' if key_ok else f'missing — set `{key_name}` in `.env`'}"
    )
    st.sidebar.markdown(
        f"{'🟢' if kb_ok else '🟠'} Knowledge base "
        f"{'ready' if kb_ok else 'not built yet'}"
    )
    st.sidebar.caption(f"Model: `{config.active_model()}` ({provider})")

    # Outcome of the previous build, persisted across the rerun.
    build_result = st.session_state.pop("kb_build_result", None)
    if build_result:
        ok, message = build_result
        (st.sidebar.success if ok else st.sidebar.error)(message)

    if st.sidebar.button("🔁 Build / rebuild knowledge base", use_container_width=True):
        with st.sidebar.status("Ingesting documents…", expanded=False):
            try:
                summary = rebuild_index()
                st.session_state.kb_build_result = (
                    True,
                    f"Indexed {summary['knowledge_chunks']} knowledge chunks and "
                    f"{summary['bid_chunks']} bid chunks for "
                    f"{len(summary['suppliers'])} suppliers.",
                )
            except Exception as exc:
                st.session_state.kb_build_result = (False, f"Ingestion failed: {exc}")
        st.rerun()

    return {"key_ok": key_ok, "kb_ok": kb_ok}


# ---------------------------------------------------------------------------
# Live timeline rendering
# ---------------------------------------------------------------------------

def format_event(event: dict) -> str:
    agent = AGENT_LABELS.get(event["agent"], event["agent"])
    icon = KIND_ICONS.get(event["kind"], "•")
    line = f"{icon} **{agent}** — {event['summary']}"
    if event["kind"] == "evidence" and isinstance(event.get("detail"), list):
        refs = ", ".join(
            f"`{d['source_file']} § {d['section'] or 'top'}`" for d in event["detail"][:3]
        )
        line += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;↳ sources: {refs}"
    return line


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_final_report(state: dict) -> None:
    report = state["final_report"]
    st.success(f"**Recommended supplier: {report['recommended_supplier']}**")
    st.markdown(f"**Executive summary.** {report['executive_summary']}")

    col_risk, col_missing, col_next = st.columns(3)
    with col_risk:
        st.markdown("**Key risks**")
        for risk in report["key_risks"] or ["—"]:
            st.markdown(f"- {risk}")
    with col_missing:
        st.markdown("**Missing / outstanding evidence**")
        for item in report["missing_evidence_summary"] or ["None identified"]:
            st.markdown(f"- {item}")
    with col_next:
        st.markdown("**Next steps**")
        for step in report["next_steps"] or ["—"]:
            st.markdown(f"- {step}")


def render_requirements(state: dict) -> None:
    req_set = state.get("requirement_set")
    if not req_set:
        st.info("No requirement set was produced.")
        return
    st.markdown(f"**{req_set['tender_ref']} — {req_set['tender_title']}**")
    st.caption(req_set["summary"])
    st.dataframe(
        pd.DataFrame([
            {
                "ID": r["requirement_id"],
                "Kind": r["kind"],
                "Requirement": r["title"],
                "Criterion": r["criterion"] or "—",
                "Source": f"{r['source']['file']} § {r['source']['section']}" if r.get("source") else "—",
            }
            for r in req_set["requirements"]
        ]),
        use_container_width=True, hide_index=True,
    )
    st.markdown("**Evaluation criteria and weights**")
    st.dataframe(
        pd.DataFrame([
            {"Criterion": c["name"], "Weight %": c["weight"], "Description": c["description"]}
            for c in req_set["criteria"]
        ]),
        use_container_width=True, hide_index=True,
    )


def render_matrix(state: dict) -> None:
    matrix = state.get("compliance_matrix")
    if not matrix:
        st.info("No compliance matrix was produced.")
        return
    suppliers = matrix["suppliers"]
    st.dataframe(
        pd.DataFrame([
            {
                "Requirement": f"{row['requirement_id']} — {row['title']}",
                "Kind": row["kind"],
                **{s: STATUS_LABELS.get(row["cells"][s]["status"], row["cells"][s]["status"])
                   for s in suppliers},
            }
            for row in matrix["rows"]
        ]),
        use_container_width=True, hide_index=True,
    )
    for supplier, stats in matrix["supplier_stats"].items():
        unassessed = stats.get("mandatory_unassessed", [])
        if stats["mandatory_missing"]:
            st.error(
                f"**{supplier}** fails mandatory requirement(s) "
                f"{', '.join(stats['mandatory_missing'])} → disqualified from award."
            )
        elif unassessed:
            st.warning(
                f"**{supplier}**: mandatory requirement(s) {', '.join(unassessed)} "
                "were not assessed — the supplier cannot be qualified until they are."
            )
        else:
            st.success(f"**{supplier}** passes all mandatory requirements.")


def render_scores(state: dict) -> None:
    comparison = state.get("comparison_result")
    if not comparison:
        st.info("No comparison result was produced.")
        return

    ranking = comparison.get("ranking") or []
    if ranking:
        st.dataframe(
            pd.DataFrame([
                {
                    "Rank": r["rank"],
                    "Supplier": r["supplier"],
                    "Weighted total": r["weighted_total"],
                    "Status": "⛔ disqualified" if r["disqualified"] else "✅ qualified",
                }
                for r in ranking
            ]),
            use_container_width=True, hide_index=True,
        )

    tool_ranking = _tool_ranking(state)
    if tool_ranking:
        col_total, col_breakdown = st.columns([2, 3])
        with col_total:
            st.altair_chart(weighted_total_chart(tool_ranking), use_container_width=True)
        with col_breakdown:
            st.altair_chart(criterion_breakdown_chart(tool_ranking), use_container_width=True)

    st.markdown(f"**Recommendation.** {comparison['recommendation']}")
    if comparison.get("caveats"):
        st.markdown("**Caveats**")
        for caveat in comparison["caveats"]:
            st.markdown(f"- ⚠️ {caveat}")


def _tool_ranking(state: dict) -> list[dict] | None:
    """Per-criterion breakdown as produced by the scoring tool (via events)."""
    for event in reversed(state.get("events", [])):
        if event["kind"] == "tool_result" and "calculate_weighted_score" in event["summary"]:
            detail = event.get("detail")
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except json.JSONDecodeError:
                    break
            if isinstance(detail, dict) and detail.get("ranking"):
                return detail["ranking"]
            break
    # Fall back to the structured comparison (no per-criterion breakdown).
    comparison = state.get("comparison_result") or {}
    cards = {c["supplier"]: c for c in comparison.get("scorecards", [])}
    ranking = []
    for r in comparison.get("ranking", []):
        card = cards.get(r["supplier"])
        if not card:
            return None
        weights = {c["name"]: c["weight"] for c in (state.get("requirement_set") or {}).get("criteria", [])}
        ranking.append({
            "supplier": r["supplier"],
            "weighted_total": r["weighted_total"],
            "rank": r["rank"],
            "disqualified": r["disqualified"],
            "breakdown": {
                s["criterion"]: {"score": s["score"], "weight": weights.get(s["criterion"], 0),
                                 "weighted": round(s["score"] * weights.get(s["criterion"], 0) / 100, 2)}
                for s in card["scores"]
            },
        })
    return ranking or None


def render_evidence_trail(state: dict) -> None:
    report = state.get("evidence_report")
    if not report:
        st.info("No evidence report was produced.")
        return
    for findings in report["findings"]:
        with st.expander(f"🔎 Evidence — {findings['supplier']}", expanded=False):
            for item in findings["items"]:
                st.markdown(
                    f"**{item['requirement_id']}** · {STATUS_LABELS.get(item['status'], item['status'])}"
                )
                if item["evidence_quote"]:
                    st.markdown(f"> {item['evidence_quote']}")
                source = item.get("source")
                caption = []
                if source:
                    caption.append(f"source: {source['file']} § {source['section'] or 'top'}")
                if item.get("note"):
                    caption.append(item["note"])
                if caption:
                    st.caption(" — ".join(caption))
            if findings.get("strengths"):
                st.markdown("**Strengths:** " + "; ".join(findings["strengths"]))
            if findings.get("concerns"):
                st.markdown("**Concerns:** " + "; ".join(findings["concerns"]))


def render_timeline(events: list[dict]) -> None:
    if not events:
        st.info("No events recorded.")
        return
    st.dataframe(
        pd.DataFrame([
            {
                "Time (UTC)": e["ts"].split("T")[1].replace("+00:00", ""),
                "Agent": AGENT_LABELS.get(e["agent"], e["agent"]),
                "Event": f"{KIND_ICONS.get(e['kind'], '•')} {e['kind']}",
                "Summary": e["summary"],
            }
            for e in events
        ]),
        use_container_width=True, hide_index=True, height=420,
    )


def render_results(state: dict) -> None:
    st.divider()
    if state["status"] == "complete" and state.get("final_report"):
        render_final_report(state)
    else:
        st.error(
            "The analysis could not be completed. "
            + ("Details: " + "; ".join(state["errors"]) if state.get("errors")
               else "No further details were recorded.")
        )
        if any(state.get(k) for k in ("requirement_set", "compliance_matrix", "comparison_result")):
            st.info("Partial results produced before the failure are shown below.")

    tab_req, tab_matrix, tab_scores, tab_evidence, tab_timeline = st.tabs(
        ["📋 Requirements", "🧮 Compliance matrix", "🏁 Scores & ranking",
         "🔎 Evidence trail", "🕒 Agent timeline"]
    )
    with tab_req:
        render_requirements(state)
    with tab_matrix:
        render_matrix(state)
    with tab_scores:
        render_scores(state)
    with tab_evidence:
        render_evidence_trail(state)
    with tab_timeline:
        render_timeline(state.get("events", []))

    export_path = state.get("export_path")
    if export_path:
        try:
            with open(export_path, "rb") as handle:
                st.download_button(
                    "⬇️ Download supplier comparison (CSV)",
                    handle.read(),
                    file_name=Path(export_path).name,
                    mime="text/csv",
                )
        except OSError:
            st.warning("The exported CSV could not be read from disk.")


# ---------------------------------------------------------------------------
# Tab: run analysis
# ---------------------------------------------------------------------------

def run_analysis_tab(env: dict) -> None:
    tenders = list_tenders()
    suppliers = list_suppliers()

    if not tenders or not suppliers:
        st.error(
            "The sample data is missing. Make sure `data/knowledge/` and "
            "`data/sample_cases/` exist (they ship with the repository)."
        )
        return

    col_tender, col_suppliers = st.columns([2, 3])
    with col_tender:
        tender = st.selectbox(
            "Tender", tenders,
            format_func=lambda t: t["title"][:80] or t["ref"],
        )
    with col_suppliers:
        chosen = st.multiselect("Suppliers to evaluate", suppliers, default=suppliers)

    request = st.text_area(
        "Instructions for the evaluation team (optional)",
        placeholder="e.g. Pay special attention to the warranty terms and the support SLA.",
        height=68,
    )

    ready = True
    if not env["key_ok"]:
        st.error("Add an API key to `.env` before running the analysis: copy "
                 "`.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` "
                 "or `GEMINI_API_KEY` (free tier).")
        ready = False
    if not env["kb_ok"]:
        st.warning("Build the knowledge base first (button in the sidebar) so the "
                   "agents can retrieve evidence.")
        ready = False
    if not chosen:
        st.warning("Select at least one supplier to evaluate.")
        ready = False
    elif len(chosen) == 1:
        st.info("Only one supplier selected — the comparison will be trivial. "
                "Select two or more for a meaningful ranking.")

    if not st.button("🚀 Run compliance analysis", type="primary", disabled=not ready):
        if st.session_state.get("last_run_error"):
            st.warning(
                f"The most recent run failed: {st.session_state.last_run_error} — "
                "the results below are from the last completed case."
                if st.session_state.get("case_state")
                else f"The most recent run failed: {st.session_state.last_run_error}"
            )
        if st.session_state.get("case_state"):
            render_results(st.session_state.case_state)
        return

    status_box = st.status("Running the multi-agent analysis…", expanded=True)
    with status_box:
        st.caption("Please don't interact with the page while the analysis is "
                   "running — Streamlit would restart the script and abort the run.")
        placeholder = st.empty()
    lines: list[str] = []

    def live_emit(event: dict) -> None:
        lines.append(format_event(event))
        placeholder.markdown("\n\n".join(lines[-25:]))
        status_box.update(label=f"{AGENT_LABELS.get(event['agent'], event['agent'])} · {event['summary'][:90]}")

    try:
        final_state = run_case(tender["ref"], chosen, request, live_emit=live_emit)
    except AgentError as exc:
        status_box.update(label="Analysis failed", state="error")
        st.session_state.last_run_error = str(exc)
        st.error(str(exc))
        return
    except Exception as exc:  # never show a stack trace to the user
        status_box.update(label="Analysis failed", state="error")
        st.session_state.last_run_error = str(exc)
        st.error(f"Unexpected error while running the analysis: {exc}")
        return

    st.session_state.last_run_error = None
    if final_state["status"] == "complete":
        status_box.update(label="Analysis complete", state="complete", expanded=False)
    else:
        status_box.update(label="Analysis finished with errors", state="error", expanded=False)

    st.session_state.case_state = final_state
    st.session_state.case_memory = build_case_memory(final_state).to_dict()
    render_results(final_state)


# ---------------------------------------------------------------------------
# Tab: case Q&A (short-term memory + retrieval)
# ---------------------------------------------------------------------------

def case_qa_tab(env: dict) -> None:
    if "case_memory" not in st.session_state:
        st.info("Run an analysis first — afterwards you can ask follow-up questions "
                "about the case here (e.g. *Why was a supplier disqualified?*).")
        return

    memory = CaseMemory.from_dict(st.session_state.case_memory)
    st.caption(f"Case {memory.case_id} · tender {memory.tender_ref} — the assistant "
               "remembers this case and can search the tender and bid documents.")

    for item in memory.conversation():
        with st.chat_message("user" if item.kind == "user" else "assistant"):
            st.markdown(item.content)

    question = st.chat_input("Ask about this evaluation…")
    if not question:
        return
    if not env["key_ok"]:
        st.error("An LLM API key is required for Q&A. Configure `.env` first.")
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        activity = st.empty()
        try:
            from src.agents import case_qa
            answer = case_qa.answer(
                memory, question,
                emit=lambda e: activity.caption(f"{KIND_ICONS.get(e['kind'], '•')} {e['summary'][:120]}"),
            )
            activity.empty()
            st.markdown(answer)
        except AgentError as exc:
            activity.empty()
            failure = f"⚠️ This question could not be answered: {exc}"
            memory.add_assistant_turn(failure)
            st.error(str(exc))
        except Exception as exc:  # never show a stack trace to the user
            activity.empty()
            memory.add_assistant_turn(f"⚠️ This question could not be answered: {exc}")
            st.error(f"Unexpected error while answering: {exc}")

    # Persist the conversation (including failures) so nothing silently vanishes.
    st.session_state.case_memory = memory.to_dict()


# ---------------------------------------------------------------------------
# Tab: about
# ---------------------------------------------------------------------------

WORKFLOW_DOT = """
digraph {
  rankdir=LR; bgcolor=transparent;
  node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6",
        fontname="Helvetica", fontsize=11];
  user [label="Procurement officer", fillcolor="#fdf1ea", color="#eb6834"];
  sup [label="Supervisor\\n(plan · route · aggregate)"];
  req [label="Requirement\\nSpecialist"];
  ev  [label="Evidence\\nSpecialist"];
  cmp [label="Comparison\\nSpecialist"];
  rag [label="Vector store\\n(tender · guidance · bids)", shape=cylinder, fillcolor="#eafaf3", color="#1baf7a"];
  tools [label="Tools: compliance matrix ·\\nweighted score · CSV export", fillcolor="#fdf7e7", color="#eda100"];
  out [label="Final report + comparison CSV", fillcolor="#fdf1ea", color="#eb6834"];
  user -> sup; sup -> req; sup -> ev; sup -> cmp;
  req -> rag; ev -> rag; ev -> tools; cmp -> tools; cmp -> rag;
  sup -> out;
}
"""


def about_tab() -> None:
    st.markdown(
        """
### What this application does

Procurement teams spend days checking supplier bids against tender requirements
and comparing offers consistently. This app runs that evaluation as a
**supervised multi-agent workflow**:

1. **Supervisor** plans the case and delegates (plan-and-execute).
2. **Requirement Specialist** extracts the tender's mandatory/scored requirements
   and weighted criteria from the tender documents (Agentic RAG).
3. **Evidence Specialist** audits every supplier bid against every requirement,
   quoting evidence with sources, and builds the compliance matrix (tool).
4. **Comparison Specialist** scores suppliers per criterion, ranks them with the
   deterministic weighted-score tool and exports the comparison CSV (tool).
5. The Supervisor combines everything into a final recommendation with risks
   and missing evidence.

All data in this demo is **synthetic** (fictional tender RFP-2026-014 and three
fictional suppliers).
        """
    )
    st.graphviz_chart(WORKFLOW_DOT, use_container_width=True)
    st.markdown(
        "Built for **Advanced Agentic AI Systems Engineering** at **SDAIA Academy** · "
        "[SDAIA Academy on GitHub](https://github.com/SDAIAAcademy) · "
        "see the repository README for architecture details."
    )


# ---------------------------------------------------------------------------

def main() -> None:
    env = render_sidebar()
    st.title("Procurement Tender Compliance & Supplier Comparison")
    st.caption(
        "Checks every supplier bid against the tender requirements, flags missing "
        "evidence, and produces a transparent weighted comparison — with sources."
    )
    tab_run, tab_qa, tab_about = st.tabs(["🚀 Run analysis", "💬 Case Q&A", "ℹ️ About"])
    with tab_run:
        run_analysis_tab(env)
    with tab_qa:
        case_qa_tab(env)
    with tab_about:
        about_tab()


main()
