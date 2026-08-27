"""Agentic-RAG retrieval exposed as agent tools.

These wrap the vector-store retriever so agents can decide when and what to
retrieve. Note: retrieval does not count toward the project's two required
business tools (those are the scoring, matrix and export tools).
"""

from pydantic import BaseModel, Field

from src import config
from src.rag.retriever import search_bids, search_knowledge


class SearchKnowledgeInput(BaseModel):
    """Search the tender, evaluation criteria and procurement guidance."""

    query: str = Field(min_length=3, description="What to look for")
    doc_type: str = Field(
        default="",
        description="Optional filter: 'tender', 'criteria' or 'guidance'",
    )
    top_k: int = Field(default=config.RETRIEVAL_TOP_K, ge=1, le=10)


class SearchBidsInput(BaseModel):
    """Search the supplier bid documents."""

    query: str = Field(min_length=3, description="What to look for")
    supplier: str = Field(
        default="", description="Optional: restrict to one supplier's bid"
    )
    top_k: int = Field(default=config.RETRIEVAL_TOP_K, ge=1, le=10)


def run_search_knowledge(payload: SearchKnowledgeInput) -> dict:
    chunks = search_knowledge(payload.query, top_k=payload.top_k, doc_type=payload.doc_type)
    return {"results": [c.as_dict() for c in chunks]}


def run_search_bids(payload: SearchBidsInput) -> dict:
    chunks = search_bids(payload.query, supplier=payload.supplier, top_k=payload.top_k)
    return {"results": [c.as_dict() for c in chunks]}
