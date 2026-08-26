"""Offline tests for document loading and chunking (no vector store)."""

from src.rag.chunking import chunk_document, chunk_documents
from src.rag.loaders import (
    RawDoc,
    load_bid_docs,
    load_knowledge_docs,
    supplier_from_title,
)


class TestLoaders:
    def test_knowledge_docs_typed(self):
        docs = load_knowledge_docs()
        types = {d.doc_type for d in docs}
        assert {"tender", "criteria", "guidance"} <= types

    def test_bid_docs_have_suppliers(self):
        docs = load_bid_docs()
        assert len(docs) == 3
        assert all(d.supplier for d in docs)
        assert all(d.doc_type == "bid" for d in docs)

    def test_supplier_from_title(self):
        assert supplier_from_title("Bid — AlphaTech Networks — RFP-2026-014") == "AlphaTech Networks"


class TestChunking:
    def test_chunks_carry_metadata(self):
        doc = RawDoc(
            file="bid_x.md",
            text="## Profile\nSome text.\n\n### Details\nMore text here.",
            doc_type="bid",
            supplier="X Corp",
            title="Bid — X Corp",
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 2
        assert chunks[0].metadata["supplier"] == "X Corp"
        assert chunks[0].metadata["source_file"] == "bid_x.md"
        assert "Profile" in chunks[0].metadata["section"]
        assert chunks[1].metadata["section"] == "Profile > Details"
        assert all(c.chunk_id.startswith("bid_x::chunk-") for c in chunks)

    def test_long_sections_are_split_with_ids_unique(self):
        long_text = "## Section\n" + "\n\n".join(f"Paragraph {i} " + "x" * 400 for i in range(10))
        doc = RawDoc(file="doc.md", text=long_text, doc_type="tender")
        chunks = chunk_document(doc)
        assert len(chunks) > 1
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(len(c.text) < 2200 for c in chunks)

    def test_real_corpus_chunks(self):
        chunks = chunk_documents(load_knowledge_docs() + load_bid_docs())
        assert len(chunks) > 40
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
