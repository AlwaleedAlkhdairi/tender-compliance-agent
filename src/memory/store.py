"""Short-term case memory.

Distinct from the other layers of the system:
- *State* (graph/state.py) tracks the active workflow run and its progress.
- *Short-term memory* (this module) preserves the useful facts and the
  conversation of the current case after the run, so follow-up questions
  can be answered without re-running the analysis.
- *Long-term retrieval* (rag/) finds knowledge reused across cases.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class MemoryItem:
    kind: str      # "fact" | "finding" | "user" | "assistant"
    content: str
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CaseMemory:
    case_id: str = ""
    tender_ref: str = ""
    items: list[MemoryItem] = field(default_factory=list)

    # ------------------------------------------------------------------ write
    def remember(self, kind: str, content: str) -> None:
        self.items.append(MemoryItem(kind=kind, content=content))

    def add_user_turn(self, text: str) -> None:
        self.remember("user", text)

    def add_assistant_turn(self, text: str) -> None:
        self.remember("assistant", text)

    # ------------------------------------------------------------------- read
    def facts(self) -> list[MemoryItem]:
        return [i for i in self.items if i.kind in ("fact", "finding")]

    def conversation(self) -> list[MemoryItem]:
        return [i for i in self.items if i.kind in ("user", "assistant")]

    def render_context(self, max_conversation_turns: int = 10) -> str:
        """Compact context block injected into follow-up Q&A prompts."""
        lines = [f"Case {self.case_id} — tender {self.tender_ref}", "", "Known case facts:"]
        lines += [f"- {i.content}" for i in self.facts()] or ["- (none)"]
        turns = self.conversation()[-max_conversation_turns:]
        if turns:
            lines += ["", "Conversation so far:"]
            lines += [f"{i.kind}: {i.content}" for i in turns]
        return "\n".join(lines)

    # ------------------------------------------------------------- serialize
    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "tender_ref": self.tender_ref,
                "items": [asdict(i) for i in self.items]}

    @classmethod
    def from_dict(cls, data: dict) -> "CaseMemory":
        memory = cls(case_id=data.get("case_id", ""), tender_ref=data.get("tender_ref", ""))
        memory.items = [MemoryItem(**item) for item in data.get("items", [])]
        return memory


def build_case_memory(final_state: dict) -> CaseMemory:
    """Distill a finished workflow run into short-term case memory."""
    memory = CaseMemory(
        case_id=final_state.get("case_id", ""),
        tender_ref=final_state.get("tender_ref", ""),
    )
    memory.remember("fact", f"Suppliers evaluated: {', '.join(final_state.get('supplier_names', []))}")
    memory.remember("fact", f"Case status: {final_state.get('status', 'unknown')}")

    matrix = final_state.get("compliance_matrix") or {}
    for supplier, stats in (matrix.get("supplier_stats") or {}).items():
        missing = ", ".join(stats.get("mandatory_missing", [])) or "none"
        memory.remember(
            "finding",
            f"{supplier}: missing mandatory items: {missing}"
            + (" (disqualified)" if stats.get("disqualified") else ""),
        )

    comparison = final_state.get("comparison_result") or {}
    for entry in comparison.get("ranking", []):
        memory.remember(
            "finding",
            f"Rank {entry.get('rank')}: {entry.get('supplier')} — weighted total "
            f"{entry.get('weighted_total')}"
            + (" (disqualified)" if entry.get("disqualified") else ""),
        )

    report = final_state.get("final_report") or {}
    if report.get("recommended_supplier"):
        memory.remember("finding", f"Recommended supplier: {report['recommended_supplier']}")
    if report.get("executive_summary"):
        memory.remember("fact", f"Executive summary: {report['executive_summary']}")
    for item in report.get("missing_evidence_summary", []):
        memory.remember("finding", f"Outstanding evidence: {item}")

    if final_state.get("export_path"):
        memory.remember("fact", f"Comparison CSV exported to {final_state['export_path']}")
    for error in final_state.get("errors", []):
        memory.remember("fact", f"Error during analysis: {error}")
    return memory
