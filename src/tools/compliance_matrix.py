"""Tool: build a requirement x supplier compliance matrix.

Turns the evidence specialist's per-requirement findings into a structured
matrix with per-supplier statistics and the list of missing mandatory
requirements (which drives disqualification downstream).
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MatrixRequirement(BaseModel):
    requirement_id: str
    title: str
    kind: Literal["mandatory", "scored"]


class MatrixFinding(BaseModel):
    requirement_id: str
    supplier: str
    status: Literal["compliant", "partial", "missing"]
    evidence_quote: str = ""
    source_file: str = ""
    note: str = ""


class ComplianceMatrixInput(BaseModel):
    """Validated input for build_compliance_matrix."""

    requirements: list[MatrixRequirement] = Field(min_length=1)
    findings: list[MatrixFinding] = Field(min_length=1)

    @field_validator("findings")
    @classmethod
    def findings_reference_known_requirements(cls, findings, info):
        requirements = info.data.get("requirements") or []
        known = {r.requirement_id for r in requirements}
        unknown = sorted({f.requirement_id for f in findings} - known)
        if known and unknown:
            raise ValueError(
                f"findings reference unknown requirement ids {unknown}; "
                f"valid ids are {sorted(known)}"
            )
        return findings


def build_compliance_matrix(payload: ComplianceMatrixInput) -> dict:
    """Structured matrix + per-supplier compliance statistics."""
    suppliers = sorted({f.supplier for f in payload.findings})
    by_key = {(f.requirement_id, f.supplier): f for f in payload.findings}

    rows = []
    for req in payload.requirements:
        cells = {}
        for supplier in suppliers:
            finding = by_key.get((req.requirement_id, supplier))
            cells[supplier] = {
                "status": finding.status if finding else "not_assessed",
                "evidence_quote": finding.evidence_quote if finding else "",
                "source_file": finding.source_file if finding else "",
                "note": finding.note if finding else "No finding recorded for this requirement.",
            }
        rows.append(
            {
                "requirement_id": req.requirement_id,
                "title": req.title,
                "kind": req.kind,
                "cells": cells,
            }
        )

    supplier_stats = {}
    for supplier in suppliers:
        counts = {"compliant": 0, "partial": 0, "missing": 0, "not_assessed": 0}
        mandatory_missing = []
        for row in rows:
            status = row["cells"][supplier]["status"]
            counts[status] += 1
            if row["kind"] == "mandatory" and status in ("missing", "partial"):
                mandatory_missing.append(row["requirement_id"])
        assessed = len(rows) - counts["not_assessed"]
        supplier_stats[supplier] = {
            "counts": counts,
            "mandatory_missing": mandatory_missing,
            "disqualified": bool(mandatory_missing),
            "compliance_rate": round(counts["compliant"] / assessed, 3) if assessed else 0.0,
        }

    return {
        "suppliers": suppliers,
        "rows": rows,
        "supplier_stats": supplier_stats,
    }
