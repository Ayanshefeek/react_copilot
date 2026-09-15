"""
retrieval/keyword.py
---------------------
Stage 2 of the hybrid pipeline (skipped when config.USE_BM25=False):
sparse/keyword retrieval via LangChain's BM25Retriever. This is what
catches exact-term/lexical matches (acronyms, specific numbers, rare
jargon) that dense embeddings can underweight.

Built from the same chunk texts ingestion produced, read back from
chunk_store.json -- cheap (no embedding involved) and keeps BM25 in
sync with whatever data/ currently holds, with no re-ingestion needed.

No LLM involved -- BM25 is a classic lexical scoring algorithm, not a
model.
"""

import json
import logging
import os

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def load_documents_for_bm25(data_dir=None):
    """Rebuild the full list of LangChain Document objects from
    chunk_store.json (chunk_id -> chunk dict)."""
    data_dir = data_dir or config.DATA_DIR
    path = os.path.join(data_dir, "chunk_store.json")
    with open(path, "r", encoding="utf-8") as f:
        chunk_store = json.load(f)

    documents = [
        Document(page_content=chunk["text"], metadata=chunk)
        for chunk in chunk_store.values()
    ]
    logger.info("Loaded %d chunks for BM25 from %s", len(documents), path)
    return documents


def build_bm25_retriever(documents, top_k=None):
    """documents: list of Document objects, e.g. from
    load_documents_for_bm25()."""
    top_k = top_k or config.TOP_K_BM25
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k
    return retriever


def keyword_search(bm25_retriever, query):
    """Returns a list of (chunk_id, score) tuples, ranked best-first.

    BM25Retriever doesn't expose its raw BM25 scores through the
    standard .invoke() call, only the ranked Documents -- but
    Reciprocal Rank Fusion (fusion.py) only needs RANK order, not the
    raw score, so we assign a synthetic descending rank-based score
    here purely for a consistent (chunk_id, score) shape.
    """
    docs = bm25_retriever.invoke(query)
    ranked = [(doc.metadata["chunk_id"], 1.0 / (rank + 1)) for rank, doc in enumerate(docs)]
    logger.debug("Keyword search: %d candidates for query=%r", len(ranked), query)
    return ranked