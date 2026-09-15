"""
retrieval/semantic.py
----------------------
Stage 1 of the hybrid pipeline: dense/semantic retrieval. Embeds the
query and searches the FAISS index built during ingestion, returning a
ranked list of (chunk_id, score) tuples -- score is cosine similarity
(higher is better), thanks to the normalized embeddings + MAX_INNER_PRODUCT
choice made in ingestion/vector_store.py.

No LLM involved -- this is a local embedding model doing similarity
search.
"""

import logging

import config

logger = logging.getLogger(__name__)


def semantic_search(vector_store, query, top_k=None):
    """vector_store: an ingestion.vector_store.VectorStore already loaded
    via VectorStore.load(...). Returns a list of (chunk_id, score)
    tuples, ranked best-first."""
    top_k = top_k or config.TOP_K_SEMANTIC
    results = vector_store.store.similarity_search_with_score(query, k=top_k)
    ranked = [(doc.metadata["chunk_id"], float(score)) for doc, score in results]
    logger.debug("Semantic search: %d candidates for query=%r", len(ranked), query)
    return ranked