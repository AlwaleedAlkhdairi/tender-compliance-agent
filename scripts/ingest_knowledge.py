"""Build (or rebuild) the vector store from the documents in data/.

Usage:
    python scripts/ingest_knowledge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_store import rebuild_index  # noqa: E402


def main() -> int:
    print("Building vector store (first run downloads a small local embedding model)...")
    try:
        summary = rebuild_index()
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Ingestion failed: {exc}")
        return 1
    print(f"  knowledge chunks : {summary['knowledge_chunks']}")
    print(f"  bid chunks       : {summary['bid_chunks']}")
    print(f"  suppliers        : {', '.join(summary['suppliers'])}")
    print("Done. The application can now retrieve evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
