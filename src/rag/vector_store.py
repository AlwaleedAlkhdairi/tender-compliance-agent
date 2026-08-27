"""Chroma vector store: embed chunks and persist them for retrieval.

Two collections are kept separate so retrieval can be scoped:
- procurement_knowledge: tender, evaluation criteria, procurement guidance
- supplier_bids: one bid document per supplier (metadata carries the supplier)

Embeddings use Chroma's default local ONNX MiniLM model, so no embedding
API key is required.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src import config
from src.rag.chunking import Chunk, chunk_documents
from src.rag.loaders import load_bid_docs, load_knowledge_docs

_client: chromadb.ClientAPI | None = None


def get_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    global _client
    if _client is None or persist_dir is not None:
        path = Path(persist_dir or config.CHROMA_DIR)
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(path))
    return _client


def _embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection(name: str):
    return get_client().get_or_create_collection(
        name=name,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _add_chunks(collection, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )


def rebuild_index() -> dict:
    """Full knowledge preparation: load -> clean -> chunk -> embed -> store.

    Returns a summary used by the ingest script and the UI.
    """
    client = get_client()

    for name in (config.KNOWLEDGE_COLLECTION, config.BIDS_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass  # collection did not exist yet

    knowledge_chunks = chunk_documents(load_knowledge_docs())
    bid_chunks = chunk_documents(load_bid_docs())

    _add_chunks(get_collection(config.KNOWLEDGE_COLLECTION), knowledge_chunks)
    _add_chunks(get_collection(config.BIDS_COLLECTION), bid_chunks)

    return {
        "knowledge_chunks": len(knowledge_chunks),
        "bid_chunks": len(bid_chunks),
        "suppliers": sorted({c.metadata["supplier"] for c in bid_chunks if c.metadata["supplier"]}),
    }


def index_ready() -> bool:
    """True when both collections exist and contain chunks."""
    try:
        client = get_client()
        names = {c.name for c in client.list_collections()}
        if not {config.KNOWLEDGE_COLLECTION, config.BIDS_COLLECTION} <= names:
            return False
        return (
            get_collection(config.KNOWLEDGE_COLLECTION).count() > 0
            and get_collection(config.BIDS_COLLECTION).count() > 0
        )
    except Exception:
        return False
