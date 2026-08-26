"""Tool registry: one place that defines every tool an agent may call.

Each entry pairs an Anthropic tool definition (schema generated from the
Pydantic input model) with a validated handler. The agent loop dispatches
through `execute_tool`, so every tool input is validated before running and
every failure comes back as a structured error instead of a crash.
"""

import json
from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel, ValidationError

from src.tools.compliance_matrix import ComplianceMatrixInput, build_compliance_matrix
from src.tools.export_comparison import ExportComparisonInput, export_comparison_csv
from src.tools.retrieval import (
    SearchBidsInput,
    SearchKnowledgeInput,
    run_search_bids,
    run_search_knowledge,
)
from src.tools.weighted_score import WeightedScoreInput, calculate_weighted_score


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[BaseModel], dict]


REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            name="search_knowledge",
            description=(
                "Search the tender document, evaluation criteria and procurement "
                "guidance. Returns passages with source file, section and chunk id "
                "for citation."
            ),
            input_model=SearchKnowledgeInput,
            handler=run_search_knowledge,
        ),
        ToolSpec(
            name="search_bids",
            description=(
                "Search the supplier bid documents for evidence. Optionally restrict "
                "to a single supplier. Returns passages with source file, section and "
                "chunk id for citation."
            ),
            input_model=SearchBidsInput,
            handler=run_search_bids,
        ),
        ToolSpec(
            name="build_compliance_matrix",
            description=(
                "Build the requirement x supplier compliance matrix from recorded "
                "findings. Returns per-supplier statistics, missing mandatory "
                "requirements and disqualification flags."
            ),
            input_model=ComplianceMatrixInput,
            handler=build_compliance_matrix,
        ),
        ToolSpec(
            name="calculate_weighted_score",
            description=(
                "Calculate weighted totals from per-criterion scorecards and rank "
                "suppliers. Weights must sum to 100; disqualified suppliers rank "
                "last. Use this for all scoring arithmetic - never compute totals "
                "yourself."
            ),
            input_model=WeightedScoreInput,
            handler=calculate_weighted_score,
        ),
        ToolSpec(
            name="export_comparison_csv",
            description=(
                "Export the final compliance matrix, weighted scores and "
                "recommendation as a CSV file the procurement team can download."
            ),
            input_model=ExportComparisonInput,
            handler=export_comparison_csv,
        ),
    ]
}


def anthropic_tool_defs(names: list[str]) -> list[dict]:
    """Anthropic `tools` parameter for a subset of the registry."""
    defs = []
    for name in names:
        spec = REGISTRY[name]
        defs.append(
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_model.model_json_schema(),
            }
        )
    return defs


def execute_tool(name: str, raw_input: dict) -> tuple[str, bool]:
    """Validate and run a tool.

    Returns (result_json, is_error). Validation problems and runtime
    failures are returned as structured errors so the calling agent can
    correct itself instead of the application crashing.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return json.dumps({"error": f"Unknown tool '{name}'"}), True
    try:
        payload = spec.input_model.model_validate(raw_input or {})
    except ValidationError as exc:
        return (
            json.dumps({"error": "Invalid tool input", "details": exc.errors(include_url=False)}, default=str),
            True,
        )
    try:
        result = spec.handler(payload)
    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"}), True
    return json.dumps(result, default=str), False
