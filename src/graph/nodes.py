"""Graph nodes: thin wrappers that connect agents to the shared state.

Each node collects timeline events (also forwarded live to the UI through
`configurable.live_emit`), calls its agent, and returns a partial state
update. Agent failures never crash the graph - they set status='failed'
and the supervisor closes the case cleanly.
"""

from langchain_core.runnables import RunnableConfig

from src import config as app_config
from src.agents import comparison_specialist, evidence_specialist, requirement_specialist, supervisor
from src.graph.state import TenderState, make_event
from src.llm import AgentError
from src.schemas import EvidenceReport, RequirementSet


def _collector(config: RunnableConfig | None):
    """Buffer events for the state update; forward each live to the UI."""
    live_emit = ((config or {}).get("configurable") or {}).get("live_emit")
    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        if live_emit:
            live_emit(event)

    return events, emit


def supervisor_node(state: TenderState, config: RunnableConfig | None = None) -> dict:
    events, emit = _collector(config)
    update: dict = {"events": events, "supervisor_steps": state["supervisor_steps"] + 1}

    # A failed case is closed, not re-planned.
    if state["status"] == "failed":
        emit(make_event("supervisor", "routing", "A specialist failed — closing the case"))
        update["next_agent"] = "finalize"
        return update

    # Stopping condition: hard bound on delegation steps.
    if state["supervisor_steps"] >= app_config.MAX_SUPERVISOR_STEPS:
        emit(make_event("supervisor", "error",
                        f"Safety bound reached ({app_config.MAX_SUPERVISOR_STEPS} routing steps) — forcing finalization"))
        update["next_agent"] = "finalize"
        update["errors"] = [f"Supervisor step bound {app_config.MAX_SUPERVISOR_STEPS} reached"]
        return update

    try:
        if not state["plan"]:
            case_plan = supervisor.plan(state, emit)
            update["plan"] = case_plan.steps
            state = {**state, "plan": case_plan.steps}
        decision = supervisor.route(state, emit)
        update["next_agent"] = decision.next_agent
    except AgentError as exc:
        emit(make_event("supervisor", "error", str(exc)))
        update.update(next_agent="finalize", status="failed", errors=[str(exc)])
    return update


def requirement_node(state: TenderState, config: RunnableConfig | None = None) -> dict:
    events, emit = _collector(config)
    try:
        requirement_set = requirement_specialist.run(
            state["tender_ref"], state["user_request"], emit
        )
        return {
            "events": events,
            "requirement_set": requirement_set.model_dump(),
            "completed_agents": state["completed_agents"] + ["requirement_specialist"],
        }
    except AgentError as exc:
        emit(make_event("requirement_specialist", "error", str(exc)))
        return {"events": events, "status": "failed", "errors": [str(exc)]}


def evidence_node(state: TenderState, config: RunnableConfig | None = None) -> dict:
    events, emit = _collector(config)
    try:
        requirement_set = RequirementSet.model_validate(state["requirement_set"])
        report, matrix = evidence_specialist.run(
            requirement_set, state["supplier_names"], emit
        )
        return {
            "events": events,
            "evidence_report": report.model_dump(),
            "compliance_matrix": matrix,
            "completed_agents": state["completed_agents"] + ["evidence_specialist"],
        }
    except AgentError as exc:
        emit(make_event("evidence_specialist", "error", str(exc)))
        return {"events": events, "status": "failed", "errors": [str(exc)]}


def comparison_node(state: TenderState, config: RunnableConfig | None = None) -> dict:
    events, emit = _collector(config)
    try:
        comparison, export_path = comparison_specialist.run(
            RequirementSet.model_validate(state["requirement_set"]),
            EvidenceReport.model_validate(state["evidence_report"]),
            state["compliance_matrix"],
            emit,
        )
        return {
            "events": events,
            "comparison_result": comparison.model_dump(),
            "export_path": export_path,
            "completed_agents": state["completed_agents"] + ["comparison_specialist"],
        }
    except AgentError as exc:
        emit(make_event("comparison_specialist", "error", str(exc)))
        return {"events": events, "status": "failed", "errors": [str(exc)]}


def finalize_node(state: TenderState, config: RunnableConfig | None = None) -> dict:
    events, emit = _collector(config)

    # Without a comparison there is nothing trustworthy to report: close the
    # case as failed instead of pretending success.
    if not state.get("comparison_result"):
        emit(make_event("supervisor", "finished",
                        "Case closed without a full result — see errors for the reason"))
        return {"events": events, "status": "failed"}

    try:
        report = supervisor.aggregate(state, emit)
        return {"events": events, "final_report": report.model_dump(), "status": "complete"}
    except AgentError as exc:
        emit(make_event("supervisor", "error", str(exc)))
        return {"events": events, "status": "failed", "errors": [str(exc)]}
