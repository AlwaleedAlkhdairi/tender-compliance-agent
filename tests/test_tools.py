"""Offline tests for the business tools (no API key, no network)."""

import csv
import json

import pytest
from pydantic import ValidationError

from src.tools.compliance_matrix import (
    ComplianceMatrixInput,
    build_compliance_matrix,
)
from src.tools.export_comparison import ExportComparisonInput, export_comparison_csv
from src.tools.registry import REGISTRY, anthropic_tool_defs, execute_tool
from src.tools.weighted_score import WeightedScoreInput, calculate_weighted_score


REQUIREMENTS = [
    {"requirement_id": "M1", "title": "ISO 27001", "kind": "mandatory"},
    {"requirement_id": "M4", "title": "Warranty 36 months", "kind": "mandatory"},
    {"requirement_id": "S2", "title": "Equipment performance", "kind": "scored"},
]

FINDINGS = [
    {"requirement_id": "M1", "supplier": "Alpha", "status": "compliant",
     "evidence_quote": "valid until 2027", "source_file": "bid_alpha.md"},
    {"requirement_id": "M4", "supplier": "Alpha", "status": "compliant"},
    {"requirement_id": "S2", "supplier": "Alpha", "status": "compliant"},
    {"requirement_id": "M1", "supplier": "Beta", "status": "missing",
     "note": "certification only in progress"},
    {"requirement_id": "M4", "supplier": "Beta", "status": "partial",
     "note": "24 months offered"},
    {"requirement_id": "S2", "supplier": "Beta", "status": "partial"},
]


class TestComplianceMatrix:
    def test_matrix_structure_and_disqualification(self):
        result = build_compliance_matrix(
            ComplianceMatrixInput(requirements=REQUIREMENTS, findings=FINDINGS)
        )
        assert result["suppliers"] == ["Alpha", "Beta"]
        assert len(result["rows"]) == 3
        assert result["supplier_stats"]["Alpha"]["disqualified"] is False
        beta = result["supplier_stats"]["Beta"]
        assert beta["disqualified"] is True
        assert beta["mandatory_missing"] == ["M1", "M4"]

    def test_unassessed_scored_pair_is_marked_but_not_disqualifying(self):
        # Alpha has findings for M1 and M4 only -> S2 (scored) is unassessed.
        result = build_compliance_matrix(
            ComplianceMatrixInput(requirements=REQUIREMENTS, findings=FINDINGS[:2])
        )
        assert result["suppliers"] == ["Alpha"]
        s2_row = next(r for r in result["rows"] if r["requirement_id"] == "S2")
        assert s2_row["cells"]["Alpha"]["status"] == "not_assessed"
        stats = result["supplier_stats"]["Alpha"]
        assert stats["counts"]["not_assessed"] == 1
        assert stats["mandatory_unassessed"] == []
        assert stats["disqualified"] is False

    def test_unassessed_mandatory_fails_closed(self):
        # No finding for mandatory M4 -> the supplier cannot count as qualified.
        findings = [FINDINGS[0], FINDINGS[2]]  # Alpha: M1 compliant, S2 compliant
        result = build_compliance_matrix(
            ComplianceMatrixInput(requirements=REQUIREMENTS, findings=findings)
        )
        stats = result["supplier_stats"]["Alpha"]
        assert stats["mandatory_unassessed"] == ["M4"]
        assert stats["mandatory_missing"] == []
        assert stats["disqualified"] is True

    def test_unknown_requirement_id_rejected(self):
        bad = FINDINGS + [{"requirement_id": "X9", "supplier": "Alpha", "status": "compliant"}]
        with pytest.raises(ValidationError, match="X9"):
            ComplianceMatrixInput(requirements=REQUIREMENTS, findings=bad)


class TestWeightedScore:
    WEIGHTS = {"Technical Solution": 40, "Implementation and Delivery": 20,
               "Experience and Team": 15, "Support and Warranty": 10, "Financial": 15}

    def make_input(self, **overrides):
        data = {
            "weights": self.WEIGHTS,
            "scorecards": [
                {"supplier": "Alpha", "scores": {
                    "Technical Solution": 90, "Implementation and Delivery": 85,
                    "Experience and Team": 88, "Support and Warranty": 92, "Financial": 70}},
                {"supplier": "Gamma", "scores": {
                    "Technical Solution": 70, "Implementation and Delivery": 75,
                    "Experience and Team": 72, "Support and Warranty": 80, "Financial": 90}},
            ],
            "disqualified": [],
        }
        data.update(overrides)
        return WeightedScoreInput(**data)

    def test_weighted_totals_and_ranking(self):
        result = calculate_weighted_score(self.make_input())
        totals = {r["supplier"]: r["weighted_total"] for r in result["ranking"]}
        # Alpha: 90*.4 + 85*.2 + 88*.15 + 92*.1 + 70*.15 = 85.9
        assert totals["Alpha"] == pytest.approx(85.9)
        assert result["ranking"][0]["supplier"] == "Alpha"
        assert result["ranking"][0]["rank"] == 1
        assert result["best_qualified"] == "Alpha"

    def test_disqualified_ranks_last_even_with_top_score(self):
        result = calculate_weighted_score(self.make_input(disqualified=["Alpha"]))
        assert result["ranking"][-1]["supplier"] == "Alpha"
        assert result["ranking"][-1]["disqualified"] is True
        assert result["best_qualified"] == "Gamma"

    def test_weights_must_sum_to_100(self):
        with pytest.raises(ValidationError, match="sum to 100"):
            self.make_input(weights={"Technical Solution": 50, "Financial": 30})

    def test_scores_must_be_in_range(self):
        cards = [{"supplier": "Alpha", "scores": dict(self.WEIGHTS, **{"Technical Solution": 140})}]
        with pytest.raises(ValidationError, match="between 0 and 100"):
            self.make_input(scorecards=cards)

    def test_scorecard_must_cover_all_criteria(self):
        cards = [{"supplier": "Alpha", "scores": {"Technical Solution": 80}}]
        with pytest.raises(ValidationError, match="missing criteria"):
            self.make_input(scorecards=cards)

    def test_within_one_point_tie_breaks_on_technical_score(self):
        # Methodology §6: totals within 1 point are tied; higher Technical wins.
        result = calculate_weighted_score(WeightedScoreInput(
            weights={"Technical Solution": 50, "Financial": 50},
            scorecards=[
                {"supplier": "A", "scores": {"Technical Solution": 90, "Financial": 70}},  # 80.0
                {"supplier": "B", "scores": {"Technical Solution": 61, "Financial": 100}},  # 80.5
            ],
        ))
        assert result["ranking"][0]["supplier"] == "A"  # higher Technical despite lower total
        assert result["best_qualified"] == "A"

    def test_tie_then_financial_breaks_on_lower_tco(self):
        # Equal totals and equal Technical: higher Financial score (= lower TCO) wins.
        result = calculate_weighted_score(WeightedScoreInput(
            weights={"Technical Solution": 50, "Financial": 50},
            scorecards=[
                {"supplier": "A", "scores": {"Technical Solution": 80, "Financial": 80}},
                {"supplier": "B", "scores": {"Technical Solution": 80, "Financial": 80.5}},
            ],
        ))
        assert result["ranking"][0]["supplier"] == "B"


class TestExportComparison:
    def test_export_writes_csv(self, tmp_path, monkeypatch):
        from src import config
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        matrix = build_compliance_matrix(
            ComplianceMatrixInput(requirements=REQUIREMENTS, findings=FINDINGS)
        )
        scoring = calculate_weighted_score(WeightedScoreInput(
            weights={"Technical Solution": 60, "Financial": 40},
            scorecards=[
                {"supplier": "Alpha", "scores": {"Technical Solution": 90, "Financial": 70}},
                {"supplier": "Beta", "scores": {"Technical Solution": 60, "Financial": 95}},
            ],
            disqualified=["Beta"],
        ))
        result = export_comparison_csv(ExportComparisonInput(
            tender_ref="RFP-2026-014",
            matrix=matrix,
            scoring=scoring,
            recommendation="Award to Alpha.",
        ))

        assert result["rows_written"] > 10
        content = list(csv.reader(open(result["path"], encoding="utf-8-sig")))
        flat = ["|".join(row) for row in content]
        assert any("COMPLIANCE MATRIX" in line for line in flat)
        assert any("DISQUALIFIED" in line for line in flat)
        assert any("EVIDENCE TRAIL" in line for line in flat)
        assert any("valid until 2027" in line for line in flat)  # evidence quote + source
        assert any("Award to Alpha." in line for line in flat)


class TestRegistry:
    def test_tool_defs_have_schemas(self):
        defs = anthropic_tool_defs(list(REGISTRY))
        assert {d["name"] for d in defs} == set(REGISTRY)
        for d in defs:
            assert d["input_schema"]["type"] == "object"
            assert d["description"]

    def test_execute_tool_validates_input(self):
        result, is_error = execute_tool("calculate_weighted_score", {"weights": {"A": 10}})
        assert is_error is True
        assert "Invalid tool input" in result

    def test_execute_tool_unknown_name(self):
        result, is_error = execute_tool("no_such_tool", {})
        assert is_error is True

    def test_execute_tool_happy_path(self):
        result, is_error = execute_tool(
            "calculate_weighted_score",
            {
                "weights": {"Technical Solution": 100},
                "scorecards": [{"supplier": "A", "scores": {"Technical Solution": 80}}],
            },
        )
        assert is_error is False
        assert json.loads(result)["best_qualified"] == "A"
