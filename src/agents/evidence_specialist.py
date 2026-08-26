"""Supplier Evidence Specialist.

CONTRACT
  Responsibility : Check every supplier bid against every requirement,
                   quote the supporting evidence, flag missing items, and
                   build the compliance matrix.
  Input          : The `RequirementSet` + the list of suppliers.
  Permitted      : `search_bids` and `search_knowledge` retrieval, and the
                   `build_compliance_matrix` tool. May not score, rank or
                   recommend suppliers.
  Output         : A validated `EvidenceReport` + the compliance matrix
                   (tool output).
  Completion     : Every (requirement x supplier) pair has a recorded
                   status backed by a quote or an explicit 'missing' note.
"""

import json

from src.graph.state import make_event
from src.llm import AgentError, EventFn, run_tool_loop, structured_call
from src.schemas import EvidenceReport, RequirementSet, SupplierFindings
from src.tools.registry import execute_tool

SYSTEM = """You are the Supplier Evidence Specialist in a procurement tender evaluation team.

Your responsibility: audit ONE supplier's bid against the tender requirements you are
given. For every requirement, search the supplier's bid with the search_bids tool and
decide:
- compliant : the bid clearly meets the requirement - quote the exact evidence.
- partial   : the bid addresses it but falls short of what the tender demands
              (for example a shorter warranty, a slower SLA, an optional add-on
              instead of an included feature). Explain the gap.
- missing   : no evidence exists in the bid, or the evidence fails the standard
              (for example a certification only 'in progress'). Say what is absent.

Consult the procurement guidance with search_knowledge when you must judge whether a
gap is a material deviation (e.g. certificates must be VALID at submission).
Judge only against the tender's requirements; never invent evidence, and never guess.
You do not score or rank suppliers - that is another agent's job.

When your audit of this supplier is complete, summarize the compliance picture."""

FINALIZE = """Based on your audit above, produce the SupplierFindings for this supplier.
Rules:
- Include one EvidenceItem per requirement (every requirement id exactly once).
- evidence_quote must be a short verbatim quote from the bid ('' only when missing).
- Set source (file/section) for every item backed by evidence.
- note must explain each partial/missing judgement, citing the guidance when used."""

MATRIX_SYSTEM = """You are the Supplier Evidence Specialist consolidating your audit.
Submit the findings through the build_compliance_matrix tool to produce the official
compliance matrix. Pass the requirements and every finding exactly as recorded; fix a
finding only if it contradicts its own quoted evidence. After the tool returns,
state one sentence confirming the matrix is built."""


def _audit_one_supplier(
    requirement_set: RequirementSet, supplier: str, emit: EventFn
) -> SupplierFindings:
    emit(make_event("evidence_specialist", "started",
                    f"Auditing bid of {supplier} against {len(requirement_set.requirements)} requirements"))

    requirement_lines = "\n".join(
        f"- {r.requirement_id} ({r.kind}): {r.title} — {r.description}"
        for r in requirement_set.requirements
    )
    prompt = (
        f"Supplier to audit: {supplier}\n"
        f"Tender: {requirement_set.tender_ref} — {requirement_set.tender_title}\n\n"
        f"Requirements to check:\n{requirement_lines}\n\n"
        f"Audit this supplier's bid. Always pass supplier='{supplier}' to search_bids."
    )
    result = run_tool_loop(
        agent="evidence_specialist",
        system=SYSTEM,
        user_prompt=prompt,
        tool_names=["search_bids", "search_knowledge"],
        emit=emit,
    )

    findings = structured_call(
        agent="evidence_specialist",
        system=SYSTEM,
        messages=result.messages + [{"role": "user", "content": FINALIZE}],
        output_model=SupplierFindings,
        tool_names=["search_bids", "search_knowledge"],
    )
    findings.supplier = supplier  # keep the state key stable regardless of spelling
    missing = [i.requirement_id for i in findings.items if i.status == "missing"]
    emit(make_event(
        "evidence_specialist", "finished",
        f"{supplier}: {sum(1 for i in findings.items if i.status == 'compliant')} compliant, "
        f"{sum(1 for i in findings.items if i.status == 'partial')} partial, "
        f"{len(missing)} missing" + (f" ({', '.join(missing)})" if missing else ""),
    ))
    return findings


def _build_matrix(
    requirement_set: RequirementSet, report: EvidenceReport, emit: EventFn
) -> dict:
    """Have the agent consolidate its findings through the matrix tool."""
    matrix_input = {
        "requirements": [
            {"requirement_id": r.requirement_id, "title": r.title, "kind": r.kind}
            for r in requirement_set.requirements
        ],
        "findings": [
            {
                "requirement_id": item.requirement_id,
                "supplier": findings.supplier,
                "status": item.status,
                "evidence_quote": item.evidence_quote,
                "source_file": item.source.file if item.source else "",
                "note": item.note,
            }
            for findings in report.findings
            for item in findings.items
        ],
    }
    result = run_tool_loop(
        agent="evidence_specialist",
        system=MATRIX_SYSTEM,
        user_prompt=(
            "Recorded audit data:\n"
            f"{json.dumps(matrix_input, indent=1)}\n\n"
            "Build the compliance matrix."
        ),
        tool_names=["build_compliance_matrix"],
        emit=emit,
        max_turns=4,
    )
    matrix = result.last_result("build_compliance_matrix")
    if matrix is None:
        # Deterministic safety net: the findings are already structured, so the
        # matrix can be built directly rather than failing the whole case.
        emit(make_event("evidence_specialist", "reasoning",
                        "Agent did not call the matrix tool; building matrix directly from findings."))
        result_json, is_error = execute_tool("build_compliance_matrix", matrix_input)
        if is_error:
            raise AgentError(f"Could not build the compliance matrix: {result_json}")
        matrix = json.loads(result_json)
    return matrix


def run(
    requirement_set: RequirementSet, suppliers: list[str], emit: EventFn
) -> tuple[EvidenceReport, dict]:
    report = EvidenceReport(
        findings=[_audit_one_supplier(requirement_set, s, emit) for s in suppliers]
    )
    matrix = _build_matrix(requirement_set, report, emit)

    disqualified = [s for s, st in matrix["supplier_stats"].items() if st["disqualified"]]
    emit(make_event(
        "evidence_specialist", "finished",
        "Compliance matrix built"
        + (f"; mandatory gaps for: {', '.join(disqualified)}" if disqualified else "; no mandatory gaps"),
    ))
    return report, matrix
