"""
retrieval/fusion.py
--------------------
Reciprocal Rank Fusion (RRF): combines two independently-ranked lists
(semantic + keyword) into one ranked list, without needing to calibrate
BM25 scores against cosine similarity scores -- RRF only looks at RANK
position in each list, which sidesteps that scale-mismatch problem
entirely.

    RRF_score(chunk) = sum, over every ranked list containing the chunk,
                        of 1 / (RRF_K + rank_in_that_list)

Standard choice: RRF_K = 60 (from the original RRF paper). A chunk that
ranks near the top of either list scores highly; a chunk found by BOTH
retrievers gets a boost from appearing in both sums -- which is exactly
the "agreement between semantic and lexical signals" effect hybrid
search is meant to capture.

Pure arithmetic -- no LLM, no model of any kind.
"""

import logging

import config

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(*ranked_lists, k=None):
    """Each ranked_list is a list of (chunk_id, score) tuples, best-first
    -- the score itself is ignored here, only rank position matters for
    RRF. Returns a single list of (chunk_id, rrf_score) tuples,
    best-first, deduplicated across the input lists.
    """
    k = k if k is not None else config.RRF_K
    fused_scores = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list):
            fused_scores.setdefault(chunk_id, 0.0)
            fused_scores[chunk_id] += 1.0 / (k + rank + 1)

    fused = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    logger.debug(
        "RRF fused %d ranked list(s) into %d unique candidates",
        len(ranked_lists), len(fused),
    )
    return fused