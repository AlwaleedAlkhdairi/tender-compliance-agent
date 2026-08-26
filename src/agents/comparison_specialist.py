"""Comparison Specialist.

CONTRACT
  Responsibility : Score qualified suppliers per criterion from the audited
                   evidence, rank them with the weighted-score tool, and
                   export the official comparison file.
  Input          : `RequirementSet` (criteria + weights), `EvidenceReport`,
                   compliance matrix.
  Permitted      : `search_knowledge` / `search_bids` retrieval,
                   `calculate_weighted_score`, `export_comparison_csv`.
                   May not change compliance statuses or requirements.
  Output         : A validated `ComparisonResult` + path of the exported CSV.
  Completion     : Every qualified supplier has a justified 0-100 score per
                   criterion, the ranking comes from the scoring tool, and
                   the CSV file exists on disk.
"""

import json

from src.graph.state import make_event
from src.llm import AgentError, EventFn, run_tool_loop, structured_call
from src.schemas import ComparisonResult, EvidenceReport, RequirementSet

SYSTEM = """You are the Comparison Specialist in a procurement tender evaluation team.

Your responsibility: turn the audited evidence into a transparent, ranked supplier
comparison. You never re-open compliance judgements - the audit is final.

Method:
1. Retrieve the scoring methodology with search_knowledge (score bands, financial
   formula, tie-breaking) and follow it exactly.
2. For each supplier, assign a 0-100 score per evaluation criterion, justified by the
   audited evidence. Score disqualified suppliers too (their scores are informational;
   they cannot be awarded). Apply the financial formula to the bid prices; retrieve a
   price with search_bids if it is not in the audit data.
3. Call calculate_weighted_score exactly once with the tender's criterion weights, all
   scorecards, and the disqualified list from the compliance matrix. Never compute
   weighted totals yourself.
4. Call export_comparison_csv once to produce the official comparison file, passing
   the compliance matrix and the scoring tool's full output unchanged.
5. Finish with a short summary of the ranking and your recommendation."""

FINALIZE = """Based on your work above, produce the ComparisonResult.
Rules:
- scorecards must contain the exact per-criterion scores you passed to the tool.
- ranking must mirror the calculate_weighted_score output (totals, ranks, flags).
- recommendation names the best qualified supplier and the decisive reasons.
- caveats must list every open risk: missing mandatory evidence, partial compliance
  gaps the buyer should negotiate, and any abnormally low price concern."""


def run(
    requirement_set: RequirementSet,
    evidence_report: EvidenceReport,
    compliance_matrix: dict,
    emit: EventFn,
) -> tuple[ComparisonResult, str | None]:
    emit(make_event("comparison_specialist", "started",
                    "Scoring suppliers and building the weighted comparison"))

    weights = {c.name: c.weight for c in requirement_set.criteria}
    disqualified = [
        supplier
        for supplier, stats in compliance_matrix["supplier_stats"].items()
        if stats["disqualified"]
    ]
    audit_summary = {
        "tender_ref": requirement_set.tender_ref,
        "criteria_weights": weights,
        "disqualified_by_mandatory_gap": disqualified,
        "supplier_stats": compliance_matrix["supplier_stats"],
        "findings": [f.model_dump() for f in evidence_report.findings],
    }

    prompt = (
        f"Audited evaluation data for {requirement_set.tender_ref}:\n"
        f"{json.dumps(audit_summary, indent=1, default=str)}\n\n"
        "Compliance matrix (pass to export_comparison_csv unchanged):\n"
        f"{json.dumps(compliance_matrix, default=str)}\n\n"
        "Produce the scored, ranked comparison and export the comparison file."
    )
    result = run_tool_loop(
        agent="comparison_specialist",
        system=SYSTEM,
        user_prompt=prompt,
        tool_names=[
            "search_knowledge",
            "search_bids",
            "calculate_weighted_score",
            "export_comparison_csv",
        ],
        emit=emit,
    )

    scoring = result.last_result("calculate_weighted_score")
    if scoring is None:
        raise AgentError(
            "The comparison specialist finished without a successful "
            "calculate_weighted_score call, so no trustworthy ranking exists."
        )
    export = result.last_result("export_comparison_csv")
    export_path = export.get("path") if export else None

    comparison = structured_call(
        agent="comparison_specialist",
        system=SYSTEM,
        messages=result.messages + [{"role": "user", "content": FINALIZE}],
        output_model=ComparisonResult,
        tool_names=[
            "search_knowledge",
            "search_bids",
            "calculate_weighted_score",
            "export_comparison_csv",
        ],
    )

    best = scoring.get("best_qualified")
    emit(make_event(
        "comparison_specialist", "finished",
        f"Ranking complete — best qualified supplier: {best or 'none'}"
        + (f"; comparison exported to {export['file_name']}" if export else "; CSV export not produced"),
    ))
    return comparison, export_path
