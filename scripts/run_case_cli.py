"""Run one complete tender analysis from the terminal (no UI).

Usage:
    python scripts/run_case_cli.py                  # all suppliers, default tender
    python scripts/run_case_cli.py "AlphaTech Networks" "GammaWave Technologies"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.graph.workflow import run_case  # noqa: E402
from src.rag.loaders import list_suppliers, list_tenders  # noqa: E402
from src.rag.vector_store import index_ready  # noqa: E402


def main() -> int:
    if not config.api_key_present():
        print("No API key configured — see .env.example.")
        return 1
    if not index_ready():
        print("Vector store not built — run: python scripts/ingest_knowledge.py")
        return 1

    tender = list_tenders()[0]
    suppliers = sys.argv[1:] or list_suppliers()
    print(f"Provider: {config.llm_provider()} · model: {config.active_model()}")
    print(f"Tender:   {tender['ref']} — {tender['title']}")
    print(f"Suppliers: {', '.join(suppliers)}\n")

    def live(event):
        detail = ""
        if event["kind"] == "evidence" and isinstance(event.get("detail"), list):
            detail = "  <- " + "; ".join(d["source_file"] for d in event["detail"][:3])
        print(f"[{event['ts'].split('T')[1][:8]}] {event['agent']:24s} {event['kind']:11s} "
              f"{event['summary'][:110]}{detail}", flush=True)

    state = run_case(tender["ref"], suppliers, "full evaluation", live_emit=live)

    print(f"\n=== STATUS: {state['status']} ===")
    for error in state["errors"]:
        print("error:", error)
    report = state.get("final_report")
    if report:
        print("\nRecommended supplier:", report["recommended_supplier"])
        print("Executive summary:", report["executive_summary"])
        print("Missing evidence:", "; ".join(report["missing_evidence_summary"]) or "none")
    matrix = state.get("compliance_matrix")
    if matrix:
        for supplier, stats in matrix["supplier_stats"].items():
            gaps = stats["mandatory_missing"] + stats.get("mandatory_unassessed", [])
            print(f"  {supplier}: {'DISQUALIFIED ' + str(gaps) if stats['disqualified'] else 'qualified'}")
    comparison = state.get("comparison_result")
    if comparison:
        for entry in comparison["ranking"]:
            print(f"  rank {entry['rank']}: {entry['supplier']} = {entry['weighted_total']}"
                  + ("  [DQ]" if entry["disqualified"] else ""))
    print("CSV:", state.get("export_path"))

    out = Path("outputs/last_cli_state.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(state, indent=1, default=str))
    print(f"Full state saved to {out}")
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
