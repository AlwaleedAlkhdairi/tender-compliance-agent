"""Load the raw knowledge and bid documents from disk.

Knowledge preparation starts here: documents -> extract -> clean -> chunk
-> embed -> store (see chunking.py and vector_store.py for the later steps).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from src import config


@dataclass
class RawDoc:
    """One source document plus the metadata carried into every chunk."""

    file: str
    text: str
    doc_type: str            # "tender" | "criteria" | "guidance" | "bid"
    supplier: str = ""       # only for bids
    title: str = ""
    extra: dict = field(default_factory=dict)


def _first_heading(text: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _clean(text: str) -> str:
    """Normalize whitespace; drop the synthetic-data blockquote banners."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _doc_type_for(file_name: str) -> str:
    if file_name.startswith("tender_"):
        return "tender"
    if "criteria" in file_name:
        return "criteria"
    if "guidance" in file_name:
        return "guidance"
    return "knowledge"


def supplier_from_title(title: str) -> str:
    """'Bid — AlphaTech Networks — RFP-2026-014' -> 'AlphaTech Networks'."""
    parts = [p.strip() for p in re.split(r"[—-]{1,2}", title) if p.strip()]
    if len(parts) >= 2 and parts[0].lower().startswith("bid"):
        return parts[1]
    return title


def load_knowledge_docs(knowledge_dir: Path | None = None) -> list[RawDoc]:
    """Tender, evaluation criteria and procurement guidance documents."""
    directory = Path(knowledge_dir or config.KNOWLEDGE_DIR)
    docs = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            RawDoc(
                file=path.name,
                text=_clean(text),
                doc_type=_doc_type_for(path.name),
                title=_first_heading(text),
            )
        )
    return docs


def load_bid_docs(cases_dir: Path | None = None) -> list[RawDoc]:
    """Supplier bid documents, one per supplier."""
    directory = Path(cases_dir or config.CASES_DIR)
    docs = []
    for path in sorted(directory.glob("bid_*.md")):
        text = path.read_text(encoding="utf-8")
        title = _first_heading(text)
        docs.append(
            RawDoc(
                file=path.name,
                text=_clean(text),
                doc_type="bid",
                supplier=supplier_from_title(title),
                title=title,
            )
        )
    return docs


def list_suppliers() -> list[str]:
    """Supplier names available for analysis (drives the UI selector)."""
    return [doc.supplier for doc in load_bid_docs()]


def list_tenders() -> list[dict]:
    """Available tenders as {ref, title, file} (drives the UI selector)."""
    tenders = []
    for doc in load_knowledge_docs():
        if doc.doc_type != "tender":
            continue
        ref_match = re.search(r"RFP-\d{4}-\d{3}", doc.title or doc.text)
        tenders.append(
            {
                "ref": ref_match.group(0) if ref_match else doc.file,
                "title": doc.title,
                "file": doc.file,
            }
        )
    return tenders
