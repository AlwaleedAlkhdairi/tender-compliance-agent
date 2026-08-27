"""The supervisor-led workflow graph (plan-and-execute with LangGraph).

    intake -> supervisor -+-> requirement_specialist -+
                          +-> evidence_specialist    -+-> back to supervisor
                          +-> comparison_specialist  -+
                          +-> finalize -> END

The supervisor routes after every specialist; `finalize` is the single
stopping point (reached on success, on failure, or at the safety bound).
"""

from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph

from src.graph import nodes
from src.graph.state import TenderState, initial_state, make_event


def _route_from_supervisor(state: TenderState) -> str:
    return state["next_agent"] or "finalize"


def build_workflow():
    graph = StateGraph(TenderState)
    graph.add_node("supervisor", nodes.supervisor_node)
    graph.add_node("requirement_specialist", nodes.requirement_node)
    graph.add_node("evidence_specialist", nodes.evidence_node)
    graph.add_node("comparison_specialist", nodes.comparison_node)
    graph.add_node("finalize", nodes.finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "requirement_specialist": "requirement_specialist",
            "evidence_specialist": "evidence_specialist",
            "comparison_specialist": "comparison_specialist",
            "finalize": "finalize",
        },
    )
    graph.add_edge("requirement_specialist", "supervisor")
    graph.add_edge("evidence_specialist", "supervisor")
    graph.add_edge("comparison_specialist", "supervisor")
    graph.add_edge("finalize", END)
    return graph.compile()


_compiled = None


def get_workflow():
    global _compiled
    if _compiled is None:
        _compiled = build_workflow()
    return _compiled


def run_case(
    tender_ref: str,
    supplier_names: list[str],
    user_request: str = "",
    live_emit: Optional[Callable[[dict], None]] = None,
) -> TenderState:
    """Run one complete case and return the final state."""
    state = initial_state(tender_ref, supplier_names, user_request)
    if live_emit:
        live_emit(make_event("supervisor", "started",
                             f"New case {state['case_id']}: {tender_ref} with "
                             f"{len(supplier_names)} suppliers"))
    result = get_workflow().invoke(
        state,
        config={
            "configurable": {"live_emit": live_emit},
            # LangGraph's own safety net, above our supervisor-step bound.
            "recursion_limit": 50,
        },
    )
    return result
