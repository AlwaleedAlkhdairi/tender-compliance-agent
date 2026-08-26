"""Split documents into coherent, metadata-rich chunks.

Strategy: split on markdown headings so each chunk stays on one topic, then
split oversized sections on paragraph boundaries with a small overlap. The
heading path is prepended to the chunk text (better embeddings) and stored
as metadata (better citations).
"""

import re
from dataclasses import dataclass

from src.rag.loaders import RawDoc

MAX_CHARS = 1400
OVERLAP_CHARS = 150


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict  # source_file, section, doc_type, supplier, title


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading-path, body) pairs on ##/### headings."""
    sections: list[tuple[str, str]] = []
    current_h2 = ""
    current_h3 = ""
    body: list[str] = []

    def flush():
        content = "\n".join(body).strip()
        if content:
            path = " > ".join(p for p in (current_h2, current_h3) if p)
            sections.append((path, content))
        body.clear()

    for line in text.splitlines():
        h2 = re.match(r"^##\s+(.+)$", line)
        h3 = re.match(r"^###\s+(.+)$", line)
        if h2:
            flush()
            current_h2, current_h3 = h2.group(1).strip(), ""
        elif h3:
            flush()
            current_h3 = h3.group(1).strip()
        else:
            body.append(line)
    flush()
    return sections


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split an oversized section on paragraph boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\n+", text)
    parts: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = current[-OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = candidate
    if current.strip():
        parts.append(current)
    return parts


def chunk_document(doc: RawDoc) -> list[Chunk]:
    """Chunk one document, carrying source metadata onto every chunk."""
    chunks: list[Chunk] = []
    stem = doc.file.rsplit(".", 1)[0]
    for section_path, section_body in _split_sections(doc.text) or [("", doc.text)]:
        for part in _split_long(section_body):
            index = len(chunks)
            header = f"[{doc.title or doc.file} | {section_path}]" if section_path else f"[{doc.title or doc.file}]"
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}::chunk-{index:03d}",
                    text=f"{header}\n{part}",
                    metadata={
                        "source_file": doc.file,
                        "section": section_path,
                        "doc_type": doc.doc_type,
                        "supplier": doc.supplier,
                        "title": doc.title,
                    },
                )
            )
    return chunks


def chunk_documents(docs: list[RawDoc]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]
