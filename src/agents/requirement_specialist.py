"""Requirement Specialist.

CONTRACT
  Responsibility : Extract the tender's mandatory requirements, scored
                   requirements and weighted evaluation criteria.
  Input          : Tender reference + the user's request.
  Permitted      : `search_knowledge` retrieval only. May not read bids,
                   score suppliers or judge compliance.
  Output         : A validated `RequirementSet`.
  Completion     : Every mandatory and scored requirement found in the
                   tender is listed with a source reference; criteria
                   weights sum to 100.
"""

from src.graph.state import make_event
from src.llm import EventFn, run_tool_loop, structured_call
from src.schemas import RequirementSet

SYSTEM = """You are the Requirement Specialist in a procurement tender evaluation team.

Your responsibility: extract the complete, structured requirement set for one tender.
You work only from evidence retrieved with the search_knowledge tool - the tender
document, the evaluation criteria and the procurement guidance. You do not read
supplier bids, and you never judge supplier compliance; that is another agent's job.

Method: retrieve the tender's mandatory requirements (pass/fail gates), its scored
requirements, and the weighted evaluation criteria. Make as many searches as you need
to be confident you have every requirement and the exact weights. Record for each
requirement where you found it (file and section).

When your research is complete, summarize what you found."""

FINALIZE = """Based on your research above, produce the complete RequirementSet.
Rules:
- requirement_id must match the tender's own ids (M1..Mn mandatory, S1..Sn scored).
- kind is 'mandatory' for pass/fail gates, 'scored' otherwise.
- For scored requirements, set `criterion` to the evaluation criterion named in the tender.
- criteria must carry the exact weights from the tender; they must sum to 100.
- Set `source` (file/section) for every requirement from the retrieved passages."""


def run(tender_ref: str, user_request: str, emit: EventFn) -> RequirementSet:
    emit(make_event("requirement_specialist", "started",
                    f"Analyzing tender {tender_ref}: extracting requirements and criteria"))

    prompt = (
        f"Tender to analyze: {tender_ref}.\n"
        f"Context from the procurement officer: {user_request or 'standard full evaluation'}.\n\n"
        "Research the tender and build the complete requirement picture."
    )
    result = run_tool_loop(
        agent="requirement_specialist",
        system=SYSTEM,
        user_prompt=prompt,
        tool_names=["search_knowledge"],
        emit=emit,
    )

    requirement_set = structured_call(
        agent="requirement_specialist",
        system=SYSTEM,
        messages=result.messages + [{"role": "user", "content": FINALIZE}],
        output_model=RequirementSet,
        tool_names=["search_knowledge"],
    )

    emit(make_event(
        "requirement_specialist", "finished",
        f"Extracted {len(requirement_set.requirements)} requirements "
        f"({sum(1 for r in requirement_set.requirements if r.kind == 'mandatory')} mandatory) "
        f"and {len(requirement_set.criteria)} weighted criteria",
    ))
    return requirement_set
