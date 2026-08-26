"""Tool: export the supplier comparison as a CSV file.

Produces the tangible business artifact the procurement team can attach to
the evaluation record (opens directly in Excel).
"""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from src import config


class ExportComparisonInput(BaseModel):
    """Validated input for export_comparison_csv."""

    tender_ref: str = Field(min_length=1)
    matrix: dict = Field(description="Output of build_compliance_matrix")
    scoring: dict = Field(description="Output of calculate_weighted_score")
    recommendation: str = ""


def export_comparison_csv(payload: ExportComparisonInput) -> dict:
    """Write a multi-section CSV; returns the path for the UI download."""
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_ref = re.sub(r"[^A-Za-z0-9_-]+", "-", payload.tender_ref)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"comparison_{safe_ref}_{stamp}.csv"

    suppliers = payload.matrix.get("suppliers", [])
    rows_written = 0

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)

        def write(row):
            nonlocal rows_written
            writer.writerow(row)
            rows_written += 1

        write([f"Supplier comparison — {payload.tender_ref}"])
        write([f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}"])
        write([])

        write(["COMPLIANCE MATRIX"])
        write(["Requirement", "Title", "Kind"] + suppliers)
        for row in payload.matrix.get("rows", []):
            write(
                [row["requirement_id"], row["title"], row["kind"]]
                + [row["cells"][s]["status"] for s in suppliers]
            )
        write([])

        write(["MISSING MANDATORY REQUIREMENTS"])
        for supplier, stats in payload.matrix.get("supplier_stats", {}).items():
            missing = ", ".join(stats.get("mandatory_missing", [])) or "none"
            write([supplier, missing, "DISQUALIFIED" if stats.get("disqualified") else "qualified"])
        write([])

        write(["WEIGHTED SCORES"])
        ranking = payload.scoring.get("ranking", [])
        criteria = list(ranking[0]["breakdown"].keys()) if ranking else []
        write(["Rank", "Supplier", "Weighted total", "Status"] + criteria)
        for result in ranking:
            write(
                [
                    result["rank"],
                    result["supplier"],
                    result["weighted_total"],
                    "DISQUALIFIED" if result["disqualified"] else "qualified",
                ]
                + [result["breakdown"][c]["score"] for c in criteria]
            )
        write([])

        if payload.recommendation:
            write(["RECOMMENDATION"])
            write([payload.recommendation])

    return {
        "path": str(path),
        "file_name": path.name,
        "rows_written": rows_written,
    }
