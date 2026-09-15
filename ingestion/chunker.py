"""
chunker.py
----------
Fixed-size fallback chunking -- used ONLY for documents that don't match
any of the 3 known formats. Every real document is chunked by
parsers.py using its own semantic boundaries; this is just a safety net
so an unexpected file doesn't crash the pipeline or get silently
dropped. Every chunk it makes is tagged format="fallback_fixed_size" so
it's always obvious which chunks came from here.

Uses LangChain's RecursiveCharacterTextSplitter (from the lightweight
`langchain-text-splitters` package) instead of a hand-rolled word-window
loop. It tries paragraph breaks first, then sentences, then words, and
only forces a hard character cut as a last resort -- so fallback chunks
stay closer to natural boundaries than a blind word count would give.
It does not call an LLM; it's pure text splitting.

Chunk size/overlap come from config.py (overridable via .env), so
tuning them never means editing this file.
"""

import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


def _flatten_to_text(doc):
    if doc["file_type"] == "md":
        return doc["text"]
    if doc["file_type"] == "pdf":
        lines = [line["text"] for page in doc["pages"] for line in page["lines"]]
        return "\n".join(lines)
    raise ValueError(f"Unsupported file_type: {doc['file_type']}")


def fixed_size_chunks(doc, title):
    """Split on paragraph -> sentence -> word -> character boundaries,
    in that priority order, via RecursiveCharacterTextSplitter."""
    text = _flatten_to_text(doc).strip()
    if not text:
        logger.warning("'%s' matched no known format and had no text to fall back on", doc["filename"])
        return []

    logger.info(
        "'%s' matched no known format -- using fixed-size fallback (chunk_size=%d, overlap=%d)",
        doc["filename"], config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_CHARS,
        chunk_overlap=config.CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]

    stem = os.path.splitext(doc["filename"])[0]
    chunks = []
    for i, piece in enumerate(pieces, start=1):
        chunks.append({
            "chunk_id": f"{stem}_{i:03d}",
            "text": piece,
            "source": doc["filename"],
            "title": title,
            "file_type": doc["file_type"],
            "format": "fallback_fixed_size",
            "section": None,
            "page": None,
            "chunk_index": i,
        })

    logger.debug("Fallback produced %d chunks for %s", len(chunks), doc["filename"])
    return chunks
