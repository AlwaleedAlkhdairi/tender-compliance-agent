"""End-to-end workflow tests with the LLM faked.

The graph wiring, state transitions, routing guardrails, stopping condition
and the real business tools all run for real - only the Claude calls are
replaced with deterministic fakes, so these tests need no API key.
"""

import json
from pathlib import Path

import pytest

from src.graph.workflow import build_workflow
from src.graph.state import initial_state
from src.llm import AgentError, AgentRunResult, ToolCallRecord
from src.memory.store import build_case_memory
from src.schemas import (
    ComparisonResult,
    CriterionScore,
    Criterion,
    EvidenceItem,
    FinalReport,
    RankedSupplier,
    Requirement,
    RequirementSet,
    RouteDecision,
    SupervisorPlan,
    SupplierFindings,
    SupplierScorecard,
)
from src.tools.registry import execute_tool

ALPHA = "AlphaTech Networks"
BETA = "BetaGrid Solutions"

REQUIREMENT_SET = RequirementSet(
    tender_ref="RFP-2026-014",
    tender_title="Campus Data Center Network Upgrade",
    summary="Network upgrade across two server halls.",
    requirements=[
        Requirement(requirement_id="M1", title="ISO 27001", kind="mandatory",
                    description="Valid ISO 27001 at submission"),
        Requirement(requirement_id="M4", title="Warranty", kind="mandatory",
                    description="36-month warranty minimum"),
        Requirement(requirement_id="S2", title="Equipment", kind="scored",
                    description="100G core / 25G access", criterion="Technical Solution"),
    ],
    criteria=[
        Criterion(name="Technical Solution", weight=60),
        Criterion(name="Financial", weight=40),
    ],
)


def _findings_for(supplier: str) -> SupplierFindings:
    if supplier == ALPHA:
        statuses = {"M1": "compliant", "M4": "compliant", "S2": "compliant"}
    else:
        statuses = {"M1": "missing", "M4": "partial", "S2": "partial"}
    return SupplierFindings(
        supplier=supplier,
        items=[
            EvidenceItem(requirement_id=rid, supplier=supplier, status=status,
                         evidence_quote="quoted evidence" if status == "compliant" else "",
                         note=f"{rid} judged {status}")
            for rid, status in statuses.items()
        ],
    )


def _empty_run(**kwargs) -> AgentRunResult:
    # Mirror the real loop's transcript shape: it starts with the user prompt
    # and always ends with an assistant turn (src/llm.py appends the final
    # response content before returning).
    messages = [
        {"role": "user", "content": kwargs.get("user_prompt", "")},
        {"role": "assistant", "content": [{"type": "text", "text": "research complete"}]},
    ]
    return AgentRunResult(final_text="research complete", messages=messages)


@pytest.fixture
def fake_agents(monkeypatch):
    """Patch every LLM touchpoint with deterministic fakes."""
    from src.agents import (
        comparison_specialist,
        evidence_specialist,
        requirement_specialist,
        supervisor,
    )

    # --- requirement specialist ---
    monkeypatch.setattr(requirement_specialist, "run_tool_loop",
                        lambda **kwargs: _empty_run(**kwargs))
    monkeypatch.setattr(requirement_specialist, "structured_call",
                        lambda **kwargs: REQUIREMENT_SET)

    # --- evidence specialist: findings per audited supplier; matrix falls
    # back to the real build_compliance_matrix tool (run returns no calls) ---
    monkeypatch.setattr(evidence_specialist, "run_tool_loop",
                        lambda **kwargs: _empty_run(**kwargs))

    def fake_evidence_structured(**kwargs):
        first_user = kwargs["messages"][0]["content"]
        supplier = next(s for s in (ALPHA, BETA) if s in first_user)
        return _findings_for(supplier)

    monkeypatch.setattr(evidence_specialist, "structured_call", fake_evidence_structured)

    # --- comparison specialist: really executes the scoring + export tools ---
    def fake_comparison_loop(**kwargs):
        score_input = {
            "weights": {"Technical Solution": 60, "Financial": 40},
            "scorecards": [
                {"supplier": ALPHA, "scores": {"Technical Solution": 90, "Financial": 70}},
                {"supplier": BETA, "scores": {"Technical Solution": 55, "Financial": 95}},
            ],
            "disqualified": [BETA],
        }
        score_json, score_err = execute_tool("calculate_weighted_score", score_input)
        assert not score_err
        scoring = json.loads(score_json)

        matrix = json.loads(kwargs["user_prompt"].split(
            "Compliance matrix (pass to export_comparison_csv unchanged):\n")[1].split("\n\nProduce")[0])
        export_json, export_err = execute_tool("export_comparison_csv", {
            "tender_ref": "RFP-2026-014",
            "matrix": matrix,
            "scoring": scoring,
            "recommendation": f"Award to {ALPHA}",
        })
        assert not export_err
        run = AgentRunResult(
            final_text="comparison done",
            messages=[
                {"role": "user", "content": kwargs.get("user_prompt", "")},
                {"role": "assistant", "content": [{"type": "text", "text": "comparison done"}]},
            ],
        )
        run.tool_calls = [
            ToolCallRecord("calculate_weighted_score", score_input, scoring, False),
            ToolCallRecord("export_comparison_csv", {}, json.loads(export_json), False),
        ]
        return run

    monkeypatch.setattr(comparison_specialist, "run_tool_loop", fake_comparison_loop)
    monkeypatch.setattr(
        comparison_specialist, "structured_call",
        lambda **kwargs: ComparisonResult(
            scorecards=[
                SupplierScorecard(supplier=ALPHA, scores=[
                    CriterionScore(criterion="Technical Solution", score=90, justification="strong"),
                    CriterionScore(criterion="Financial", score=70, justification="pricey"),
                ]),
            ],
            ranking=[
                RankedSupplier(supplier=ALPHA, weighted_total=82.0, rank=1),
                RankedSupplier(supplier=BETA, weighted_total=71.0, rank=2, disqualified=True),
            ],
            recommendation=f"Award to {ALPHA}",
            caveats=["BetaGrid missing M1"],
        ),
    )

    # --- supervisor: plan, dependency-aware routing, aggregation ---
    def fake_supervisor_structured(**kwargs):
        model = kwargs["output_model"]
        if model is SupervisorPlan:
            return SupervisorPlan(objective="Evaluate RFP-2026-014",
                                  steps=["requirements", "evidence", "comparison", "report"])
        if model is RouteDecision:
            snapshot = json.loads(
                kwargs["messages"].split("Current case state:\n")[1].split("\n\nWho acts next?")[0]
            )
            artifacts = snapshot["artifacts"]
            if not artifacts["requirement_set"]:
                agent = "requirement_specialist"
            elif not artifacts["compliance_matrix"]:
                agent = "evidence_specialist"
            elif not artifacts["comparison_result"]:
                agent = "comparison_specialist"
            else:
                agent = "finalize"
            return RouteDecision(next_agent=agent, reasoning="dependency order")
        if model is FinalReport:
            return FinalReport(
                tender_ref="RFP-2026-014",
                executive_summary="AlphaTech leads; BetaGrid disqualified on M1.",
                recommended_supplier=ALPHA,
                key_risks=["Premium price"],
                missing_evidence_summary=[f"{BETA}: M1 certification missing"],
                next_steps=["Verify references", "Negotiate price"],
            )
        raise AssertionError(f"unexpected structured model {model}")

    monkeypatch.setattr(supervisor, "structured_call", fake_supervisor_structured)
    return monkeypatch


class TestWorkflowEndToEnd:
    def test_complete_run_produces_final_report(self, fake_agents, tmp_path, monkeypatch):
        from src import config
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        events_seen = []
        graph = build_workflow()
        final = graph.invoke(
            initial_state("RFP-2026-014", [ALPHA, BETA], "full evaluation"),
            config={"configurable": {"live_emit": events_seen.append}},
        )

        assert final["status"] == "complete"
        assert final["final_report"]["recommended_supplier"] == ALPHA
        # Real matrix tool ran and disqualified Beta on the missing mandatory M1.
        assert final["compliance_matrix"]["supplier_stats"][BETA]["disqualified"] is True
        assert final["compliance_matrix"]["supplier_stats"][ALPHA]["disqualified"] is False
        # The comparison result carries Alpha first, and the CSV written by the
        # REAL export tool (fed by the REAL scoring tool) confirms the ranking.
        assert final["comparison_result"]["ranking"][0]["supplier"] == ALPHA
        export = Path(final["export_path"])
        assert export.exists() and export.parent == tmp_path
        csv_text = export.read_text(encoding="utf-8-sig")
        assert f"1,{ALPHA}" in csv_text  # rank 1 row from the real scoring output
        assert "EVIDENCE TRAIL" in csv_text
        # All three specialists completed, in dependency order.
        assert final["completed_agents"] == [
            "requirement_specialist", "evidence_specialist", "comparison_specialist",
        ]
        # Timeline events reached both the state and the live callback.
        kinds = {e["kind"] for e in final["events"]}
        assert {"started", "routing", "finished"} <= kinds
        assert len(events_seen) == len(final["events"])

    def test_specialist_failure_closes_case_cleanly(self, fake_agents, monkeypatch):
        from src.agents import requirement_specialist

        def boom(**_kwargs):
            raise AgentError("simulated API outage")

        monkeypatch.setattr(requirement_specialist, "run_tool_loop", boom)

        final = build_workflow().invoke(initial_state("RFP-2026-014", [ALPHA], ""))
        assert final["status"] == "failed"
        assert final["final_report"] is None
        assert any("simulated API outage" in e for e in final["errors"])
        assert any(e["kind"] == "error" for e in final["events"])

    def test_supervisor_guardrail_and_stopping_condition(self, fake_agents, monkeypatch, tmp_path):
        """A supervisor that always routes wrongly cannot break or hang the case."""
        from src import config
        from src.agents import supervisor
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        def bad_router(**kwargs):
            if kwargs["output_model"] is RouteDecision:
                return RouteDecision(next_agent="comparison_specialist", reasoning="wrong")
            if kwargs["output_model"] is SupervisorPlan:
                return SupervisorPlan(objective="x", steps=["y"])
            return FinalReport(
                tender_ref="RFP-2026-014", executive_summary="s",
                recommended_supplier=ALPHA, key_risks=[],
                missing_evidence_summary=[], next_steps=[],
            )

        monkeypatch.setattr(supervisor, "structured_call", bad_router)

        final = build_workflow().invoke(initial_state("RFP-2026-014", [ALPHA, BETA], ""))
        # Guardrail redirected the invalid routings, so the pipeline still ran
        # in dependency order — and each specialist ran exactly once (re-running
        # a completed specialist is also an invalid delegation).
        assert final["compliance_matrix"] is not None
        assert final["comparison_result"] is not None
        assert final["status"] == "complete"
        assert final["completed_agents"] == [
            "requirement_specialist", "evidence_specialist", "comparison_specialist",
        ]
        assert any("Overrode invalid routing" in e["summary"] for e in final["events"]
                   if e["kind"] == "routing")


class TestCaseMemory:
    def test_memory_distills_final_state(self, fake_agents, tmp_path, monkeypatch):
        from src import config
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        final = build_workflow().invoke(initial_state("RFP-2026-014", [ALPHA, BETA], ""))
        memory = build_case_memory(final)

        context = memory.render_context()
        assert "RFP-2026-014" in context
        assert ALPHA in context
        assert any("missing mandatory" in i.content for i in memory.facts())

        memory.add_user_turn("Why was BetaGrid disqualified?")
        memory.add_assistant_turn("Missing valid ISO 27001 (M1).")
        assert "Why was BetaGrid" in memory.render_context()
