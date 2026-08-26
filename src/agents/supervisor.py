"""Supervisor.

CONTRACT
  Responsibility : Owns the case end-to-end - plans the evaluation, routes
                   work to specialists (plan-and-execute), tracks progress,
                   enforces the stopping condition and combines all
                   specialist findings into the final report.
  Input          : The user's request + the full workflow state.
  Permitted      : Structured planning/routing/aggregation calls only.
                   The supervisor performs no retrieval and no scoring
                   itself - specialists do the work.
  Output         : `SupervisorPlan`, `RouteDecision`s, `FinalReport`.
  Completion     : A final report exists, or the safety bound was reached
                   and the case is closed with a clear failure status.
"""

import json

from src.graph.state import TenderState, make_event
from src.llm import EventFn, structured_call
from src.schemas import FinalReport, RouteDecision, SupervisorPlan

PLAN_SYSTEM = """You are the Supervisor of a procurement tender evaluation team.
You own the case: plan it, delegate to specialists, and combine their findings.

Your specialists (each with a bounded contract):
- requirement_specialist : extracts tender requirements and weighted criteria.
- evidence_specialist    : audits each supplier bid against the requirements and
                           builds the compliance matrix.
- comparison_specialist  : scores, ranks and exports the supplier comparison.

Produce a short, ordered plan for the case you are given."""

ROUTE_SYSTEM = """You are the Supervisor of a procurement tender evaluation team,
executing your plan step by step.

Routing rules (dependencies you must respect):
- evidence_specialist needs the requirement set to exist.
- comparison_specialist needs the compliance matrix to exist.
- Route to 'finalize' only when the comparison result exists (or the case cannot
  proceed and must be closed).
- Never re-run a specialist that already completed successfully.

Decide which agent must act next and explain why in one or two sentences."""

AGGREGATE_SYSTEM = """You are the Supervisor closing a procurement tender evaluation case.
Combine the specialists' findings into the final report for the procurement team.
Be faithful to the data: the recommendation must match the comparison ranking, every
missing mandatory item must appear in missing_evidence_summary (grouped per supplier),
and key risks must reflect the audit's caveats. Do not invent facts."""


def _progress_snapshot(state: TenderState) -> str:
    """Compact, structured view of the case the supervisor routes from."""
    matrix = state.get("compliance_matrix") or {}
    return json.dumps(
        {
            "tender_ref": state["tender_ref"],
            "suppliers": state["supplier_names"],
            "plan": state["plan"],
            "completed_agents": state["completed_agents"],
            "artifacts": {
                "requirement_set": bool(state.get("requirement_set")),
                "evidence_report": bool(state.get("evidence_report")),
                "compliance_matrix": bool(matrix),
                "comparison_result": bool(state.get("comparison_result")),
            },
            "disqualified": [
                s for s, st in (matrix.get("supplier_stats") or {}).items()
                if st.get("disqualified")
            ],
            "errors": state["errors"][-3:],
            "supervisor_steps_used": state["supervisor_steps"],
        },
        indent=1,
    )


def plan(state: TenderState, emit: EventFn) -> SupervisorPlan:
    result = structured_call(
        agent="supervisor",
        system=PLAN_SYSTEM,
        messages=(
            f"New case.\nTender: {state['tender_ref']}\n"
            f"Suppliers to evaluate: {', '.join(state['supplier_names'])}\n"
            f"Request from the procurement officer: {state['user_request'] or 'full evaluation'}"
        ),
        output_model=SupervisorPlan,
    )
    emit(make_event("supervisor", "reasoning",
                    f"Objective: {result.objective}", detail=result.steps))
    return result


VALID_PREREQS = {
    "requirement_specialist": lambda s: True,
    "evidence_specialist": lambda s: bool(s.get("requirement_set")),
    "comparison_specialist": lambda s: bool(s.get("compliance_matrix")),
    "finalize": lambda s: True,
}


def route(state: TenderState, emit: EventFn) -> RouteDecision:
    decision = structured_call(
        agent="supervisor",
        system=ROUTE_SYSTEM,
        messages=f"Current case state:\n{_progress_snapshot(state)}\n\nWho acts next?",
        output_model=RouteDecision,
    )

    # Deterministic guardrail: an invalid delegation is corrected, not obeyed.
    if not VALID_PREREQS[decision.next_agent](state):
        fallback = (
            "requirement_specialist" if not state.get("requirement_set")
            else "evidence_specialist" if not state.get("compliance_matrix")
            else "comparison_specialist"
        )
        emit(make_event("supervisor", "routing",
                        f"Overrode invalid routing to {decision.next_agent} "
                        f"(missing prerequisite); delegating to {fallback} instead"))
        return RouteDecision(
            next_agent=fallback,
            reasoning=f"Corrected: {decision.next_agent} lacks its input; {fallback} must run first.",
        )

    emit(make_event("supervisor", "routing",
                    f"Delegating to {decision.next_agent}", detail=decision.reasoning))
    return decision


def aggregate(state: TenderState, emit: EventFn) -> FinalReport:
    emit(make_event("supervisor", "started", "Combining specialist findings into the final report"))
    payload = {
        "tender_ref": state["tender_ref"],
        "requirement_set": state.get("requirement_set"),
        "compliance_matrix": state.get("compliance_matrix"),
        "comparison_result": state.get("comparison_result"),
        "errors": state["errors"],
    }
    report = structured_call(
        agent="supervisor",
        system=AGGREGATE_SYSTEM,
        messages=f"Case data:\n{json.dumps(payload, indent=1, default=str)}\n\nProduce the final report.",
        output_model=FinalReport,
    )
    emit(make_event("supervisor", "finished",
                    f"Case closed — recommended supplier: {report.recommended_supplier}"))
    return report
