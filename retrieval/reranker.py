"""
retrieval/reranker.py
-----------------------
Stage 3 of the hybrid pipeline (skipped when config.USE_RERANKER=False):
cross-encoder reranking. Unlike semantic/BM25 search -- which score the
query and each chunk independently, then compare -- a cross-encoder
scores the (query, chunk_text) PAIR jointly, which is slower but more
precise. It's used here only to re-sort a small candidate set the first
two stages already narrowed down, not to search the whole corpus.

Uses LangChain's HuggingFaceCrossEncoder wrapper around a small,
CPU-friendly model (cross-encoder/ms-marco-MiniLM-L-6-v2 by default,
see config.RERANKER_MODEL_NAME), following the same LangChain-first
pattern as ingestion/embeddings.py.

No LLM -- a cross-encoder is a small classifier-style model, not a
generative one.
"""

import logging

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import config

logger = logging.getLogger(__name__)

_cross_encoder = None  # lazy singleton -- loaded once, reused everywhere


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading reranker model: %s", config.RERANKER_MODEL_NAME)
        _cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL_NAME)
        logger.info("Reranker model ready")
    return _cross_encoder


def rerank(query, chunk_ids, chunk_texts, top_k=5):
    """chunk_ids and chunk_texts are parallel lists (the fused candidates
    to re-sort). Returns a list of (chunk_id, score) tuples, best-first,
    truncated to top_k."""
    if not chunk_ids:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(query, text) for text in chunk_texts]
    scores = cross_encoder.score(pairs)

    scored = list(zip(chunk_ids, scores))
    scored.sort(key=lambda item: item[1], reverse=True)

    logger.debug("Reranked %d candidates for query=%r", len(scored), query)
    return scored[:top_k]