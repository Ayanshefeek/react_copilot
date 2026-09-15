"""
retrieval/tools.py
-------------------
The two tools the ReAct agent will call: search(query) and
read_chunk(chunk_id). Wires semantic search, BM25, RRF fusion, and the
reranker together behind config.USE_BM25 / config.USE_RERANKER toggles,
so the exact same code path runs three ways for the evaluation report's
ablation:

    semantic only                   USE_BM25=False, USE_RERANKER=False
    semantic + BM25 + RRF           USE_BM25=True,  USE_RERANKER=False
    semantic + BM25 + RRF + rerank  USE_BM25=True,  USE_RERANKER=True

No LLM is used anywhere in this file -- retrieval, fusion, and
reranking are all local, non-generative steps.
"""

import json
import logging
import os
import time

import config
from ingestion.embeddings import get_embeddings
from ingestion.vector_store import VectorStore
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.keyword import build_bm25_retriever, keyword_search, load_documents_for_bm25
from retrieval.reranker import rerank
from retrieval.semantic import semantic_search

logger = logging.getLogger(__name__)


class RetrievalTools:
    """Loads the FAISS index + chunk store once (and the BM25 retriever,
    if enabled), then exposes search()/read_chunk() as plain methods --
    these become the ReAct agent's tool functions when the agent loop is
    built on top of this."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or config.DATA_DIR

        embeddings = get_embeddings()
        self.vector_store = VectorStore.load(self.data_dir, embeddings)

        chunk_store_path = os.path.join(self.data_dir, "chunk_store.json")
        with open(chunk_store_path, "r", encoding="utf-8") as f:
            self.chunk_store = json.load(f)

        self.bm25_retriever = None
        if config.USE_BM25:
            documents = load_documents_for_bm25(self.data_dir)
            self.bm25_retriever = build_bm25_retriever(documents)

        logger.info(
            "RetrievalTools ready (%d chunks, USE_BM25=%s, USE_RERANKER=%s)",
            len(self.chunk_store), config.USE_BM25, config.USE_RERANKER,
        )

    def search(self, query, top_k=5):
        """Returns a list of dicts: chunk_id + light metadata + score --
        never full chunk text, never an LLM call. This is the shape the
        ReAct agent's search() tool will return."""
        start = time.time()

        semantic_ranked = semantic_search(self.vector_store, query, top_k=config.TOP_K_SEMANTIC)

        if config.USE_BM25 and self.bm25_retriever is not None:
            keyword_ranked = keyword_search(self.bm25_retriever, query)
            fused = reciprocal_rank_fusion(semantic_ranked, keyword_ranked)
        else:
            fused = semantic_ranked

        candidates = fused[:config.TOP_K_RERANK_CANDIDATES]
        candidate_ids = [chunk_id for chunk_id, _score in candidates]

        if config.USE_RERANKER and candidate_ids:
            candidate_texts = [self.chunk_store[cid]["text"] for cid in candidate_ids]
            final_ranked = rerank(query, candidate_ids, candidate_texts, top_k=top_k)
        else:
            final_ranked = candidates[:top_k]

        results = []
        for chunk_id, score in final_ranked:
            meta = self.chunk_store[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "source": meta["source"],
                "section": meta.get("section"),
                "page": meta.get("page"),
                "score": float(score),
            })

        elapsed_ms = (time.time() - start) * 1000
        logger.info("search(%r) -> %d results in %.1fms", query, len(results), elapsed_ms)
        return results

    def read_chunk(self, chunk_id):
        """ Returns just the chunk's text -- not the rest of its metadata.
        source/section/page were already surfaced by search() and are
        still sitting in the agent's ReAct trajectory (that's how it got
        this chunk_id in the first place), so returning them again here
        would just be repeated tokens for no new information. No LLM
        involved -- a direct dictionary lookup. """
        if chunk_id not in self.chunk_store: 
            raise KeyError(f"chunk_id not found: {chunk_id}")
        return self.chunk_store[chunk_id]["text"]