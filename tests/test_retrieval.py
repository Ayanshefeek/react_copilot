"""
test_retrieval.py
------------------
Tests for the hybrid retrieval pipeline (retrieval/). Covers:

  - RRF fusion arithmetic against a hand-computed example
  - BM25 actually finding a lexical match
  - the reranker re-sorting candidates (using a stubbed cross-encoder --
    no network needed, since the real model requires a Hugging Face
    download)
  - RetrievalTools.search() end-to-end with the ablation toggles on and
    off (using stubbed embeddings + reranker, same reasoning)

Run:
    pytest tests/test_retrieval.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.keyword import build_bm25_retriever, keyword_search

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def test_rrf_combines_two_lists_by_hand_computed_score():
    semantic = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    keyword = [("c", 1.0), ("a", 0.5)]

    fused = reciprocal_rank_fusion(semantic, keyword, k=60)
    fused_dict = dict(fused)

    # a: rank 0 in semantic, rank 1 in keyword -> 1/61 + 1/62
    # b: rank 1 in semantic only               -> 1/62
    # c: rank 2 in semantic, rank 0 in keyword  -> 1/63 + 1/61
    expected_a = 1 / 61 + 1 / 62
    expected_b = 1 / 62
    expected_c = 1 / 63 + 1 / 61

    assert fused_dict["a"] == pytest.approx(expected_a)
    assert fused_dict["b"] == pytest.approx(expected_b)
    assert fused_dict["c"] == pytest.approx(expected_c)

    # "a" appears in both lists near the top -> should rank highest overall
    assert fused[0][0] == "a"


def test_rrf_single_list_is_passthrough_order():
    """With only one ranked list (the semantic-only ablation case), RRF
    should preserve that list's relative order."""
    semantic = [("x", 0.99), ("y", 0.5), ("z", 0.1)]
    fused = reciprocal_rank_fusion(semantic)
    assert [chunk_id for chunk_id, _ in fused] == ["x", "y", "z"]


def test_rrf_boosts_chunks_found_by_both_retrievers():
    semantic = [("only_semantic", 0.9), ("shared", 0.5)]
    keyword = [("shared", 1.0), ("only_keyword", 0.3)]
    fused = reciprocal_rank_fusion(semantic, keyword)
    fused_ids = [chunk_id for chunk_id, _ in fused]
    # "shared" appears in both lists -- should outrank a chunk found by
    # only one retriever, even if that chunk ranked #1 there.
    assert fused_ids.index("shared") < fused_ids.index("only_semantic")


# ---------------------------------------------------------------------------
# BM25 keyword search
# ---------------------------------------------------------------------------
def test_bm25_finds_exact_lexical_match():
    docs = [
        Document(page_content="Evaluation should separate retrieval quality from generation quality.",
                 metadata={"chunk_id": "a_001"}),
        Document(page_content="A simple regression suite is a fixed set of prompts run on every change.",
                 metadata={"chunk_id": "a_002"}),
    ]
    retriever = build_bm25_retriever(docs, top_k=2)
    ranked = keyword_search(retriever, "regression suite prompts")
    assert ranked[0][0] == "a_002"  # the lexically matching chunk ranks first


# ---------------------------------------------------------------------------
# Reranker (stubbed cross-encoder -- no network / no real model download)
# ---------------------------------------------------------------------------
def test_reranker_reorders_by_stubbed_score(monkeypatch):
    import retrieval.reranker as reranker_module

    class FakeCrossEncoder:
        """Scores a pair higher if the chunk text shares more words with
        the query -- deterministic, no model download required."""
        def score(self, pairs):
            scores = []
            for query, text in pairs:
                query_words = set(query.lower().split())
                text_words = set(text.lower().split())
                scores.append(float(len(query_words & text_words)))
            return scores

    monkeypatch.setattr(reranker_module, "get_cross_encoder", lambda: FakeCrossEncoder())

    chunk_ids = ["low_overlap", "high_overlap"]
    chunk_texts = [
        "completely unrelated sentence about cooking",
        "retrieval quality and generation quality evaluation",
    ]
    result = reranker_module.rerank("retrieval quality evaluation", chunk_ids, chunk_texts, top_k=2)

    assert result[0][0] == "high_overlap"  # the more relevant chunk moved to rank 0


# ---------------------------------------------------------------------------
# RetrievalTools end-to-end (stubbed embeddings + reranker)
# ---------------------------------------------------------------------------
class FakeEmbeddings(Embeddings):
    """Deterministic stand-in for HuggingFaceEmbeddings -- this sandbox
    has no network route to Hugging Face. Random but seeded, and
    normalized like the real model's output."""
    def embed_documents(self, texts):
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(len(texts), 384)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture()
def indexed_data_dir(tmp_path, monkeypatch):
    """Run the ingestion pipeline (with stubbed embeddings) against the
    sample corpus into a temp data/ dir, so retrieval tests have a real
    index + chunk_store.json to work against."""
    import ingestion.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "get_embeddings", lambda: FakeEmbeddings())

    out_dir = str(tmp_path / "data")
    pipeline_module.run_pipeline(CORPUS_DIR, out_dir)
    return out_dir


def test_retrieval_tools_semantic_only(indexed_data_dir, monkeypatch):
    monkeypatch.setattr(config, "USE_BM25", False)
    monkeypatch.setattr(config, "USE_RERANKER", False)

    import retrieval.tools as tools_module
    monkeypatch.setattr(tools_module, "get_embeddings", lambda: FakeEmbeddings())

    rt = tools_module.RetrievalTools(data_dir=indexed_data_dir)
    assert rt.bm25_retriever is None  # BM25 never built when the toggle is off

    results = rt.search("evaluation quality", top_k=3)
    assert len(results) <= 3
    for r in results:
        assert set(r.keys()) == {"chunk_id", "source", "section", "page", "score"}

    # read_chunk returns just the text -- source/section/page were
    # already surfaced by search() above, no need to repeat them
    text = rt.read_chunk(results[0]["chunk_id"])
    assert isinstance(text, str) and text.strip() != ""


def test_retrieval_tools_hybrid_with_rerank(indexed_data_dir, monkeypatch):
    monkeypatch.setattr(config, "USE_BM25", True)
    monkeypatch.setattr(config, "USE_RERANKER", True)

    import retrieval.reranker as reranker_module
    import retrieval.tools as tools_module
    monkeypatch.setattr(tools_module, "get_embeddings", lambda: FakeEmbeddings())

    class FakeCrossEncoder:
        def score(self, pairs):
            return [float(len(text)) for _query, text in pairs]  # arbitrary but deterministic

    monkeypatch.setattr(reranker_module, "get_cross_encoder", lambda: FakeCrossEncoder())

    rt = tools_module.RetrievalTools(data_dir=indexed_data_dir)
    assert rt.bm25_retriever is not None  # BM25 built when the toggle is on

    results = rt.search("evaluation quality", top_k=3)
    assert len(results) <= 3
    for r in results:
        assert set(r.keys()) == {"chunk_id", "source", "section", "page", "score"}


def test_read_chunk_raises_on_unknown_id(indexed_data_dir, monkeypatch):
    import retrieval.tools as tools_module
    monkeypatch.setattr(tools_module, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(config, "USE_BM25", False)

    rt = tools_module.RetrievalTools(data_dir=indexed_data_dir)
    with pytest.raises(KeyError):
        rt.read_chunk("does_not_exist_001")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))