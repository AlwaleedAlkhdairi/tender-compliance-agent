"""Explicit workflow state for one tender-analysis case.

The state is the single structured record every agent reads from and writes
to.  Nodes return partial updates; list fields with an `operator.add` reducer
accumulate (events, errors) so the timeline is never overwritten.
"""

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict


class TenderState(TypedDict):
    """Shared state passed between the supervisor and the specialists."""

    # --- case identity & input ---
    case_id: str
    tender_ref: str
    supplier_names: list[str]
    user_request: str

    # --- supervisor control ---
    plan: list[str]                 # ordered plan produced up-front
    next_agent: str                 # routing decision for the conditional edge
    supervisor_steps: int           # how many times the supervisor has routed
    completed_agents: list[str]     # specialists that have finished

    # --- specialist results (structured hand-offs) ---
    requirement_set: Optional[dict]     # RequirementSet.model_dump()
    evidence_report: Optional[dict]     # EvidenceReport.model_dump()
    compliance_matrix: Optional[dict]   # build_compliance_matrix tool output
    comparison_result: Optional[dict]   # ComparisonResult.model_dump()
    export_path: Optional[str]          # CSV written by the export tool

    # --- final output ---
    final_report: Optional[dict]        # FinalReport.model_dump()
    status: str                         # "in_progress" | "complete" | "failed"

    # --- observability (accumulating) ---
    events: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


def make_event(agent: str, kind: str, summary: str, detail: Any = None) -> dict:
    """One timeline entry shown in the UI (agent activity, tool call, evidence)."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent,
        "kind": kind,  # "started" | "reasoning" | "tool_call" | "tool_result"
                       # | "evidence" | "finished" | "error" | "routing"
        "summary": summary,
        "detail": detail,
    }


def initial_state(tender_ref: str, supplier_names: list[str], user_request: str) -> TenderState:
    """Validated starting state for a new case."""
    return TenderState(
        case_id=f"case-{uuid.uuid4().hex[:8]}",
        tender_ref=tender_ref,
        supplier_names=supplier_names,
        user_request=user_request,
        plan=[],
        next_agent="",
        supervisor_steps=0,
        completed_agents=[],
        requirement_set=None,
        evidence_report=None,
        compliance_matrix=None,
        comparison_result=None,
        export_path=None,
        final_report=None,
        status="in_progress",
        events=[],
        errors=[],
    )
