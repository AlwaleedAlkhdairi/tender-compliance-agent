"""Runtime retrieval: query -> ranked chunks with source metadata.

Agents call these functions through their retrieval tools; every result
keeps its source metadata so evidence can be cited in the final report.
"""

from dataclasses import dataclass

from src import config
from src.rag.vector_store import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    section: str
    doc_type: str
    supplier: str
    chunk_id: str
    distance: float

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "section": self.section,
            "doc_type": self.doc_type,
            "supplier": self.supplier,
            "chunk_id": self.chunk_id,
            "distance": round(self.distance, 4),
        }


def _query(collection_name: str, query: str, top_k: int, where: dict | None = None) -> list[RetrievedChunk]:
    collection = get_collection(collection_name)
    result = collection.query(
        query_texts=[query],
        n_results=max(1, top_k),
        where=where or None,
    )
    chunks = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source_file=meta.get("source_file", ""),
                section=meta.get("section", ""),
                doc_type=meta.get("doc_type", ""),
                supplier=meta.get("supplier", ""),
                chunk_id=chunk_id,
                distance=dist,
            )
        )
    return chunks


def search_knowledge(query: str, top_k: int = config.RETRIEVAL_TOP_K, doc_type: str = "") -> list[RetrievedChunk]:
    """Search tender documents, evaluation criteria and procurement guidance."""
    where = {"doc_type": doc_type} if doc_type else None
    return _query(config.KNOWLEDGE_COLLECTION, query, top_k, where)


def search_bids(query: str, supplier: str = "", top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """Search supplier bid documents, optionally scoped to one supplier."""
    where = {"supplier": supplier} if supplier else None
    return _query(config.BIDS_COLLECTION, query, top_k, where)


def format_for_agent(chunks: list[RetrievedChunk]) -> str:
    """Render retrieval results as a compact, citable block for an agent."""
    if not chunks:
        return "No matching passages found."
    parts = []
    for i, c in enumerate(chunks, start=1):
        cite = f"{c.source_file} | section: {c.section or 'top'} | chunk: {c.chunk_id}"
        parts.append(f"[{i}] ({cite})\n{c.text}")
    return "\n\n".join(parts)
