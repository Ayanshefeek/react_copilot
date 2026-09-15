"""
config.py
---------
One place for settings that might change between environments or
between people running this project: the embedding model name, chunk
sizes, retrieval/reranking parameters, default paths, and logging setup.

Secrets are NOT stored here. python-dotenv loads .env into the process
environment (load_dotenv() below), and libraries that need a token
(huggingface_hub, used under the hood by langchain_huggingface) read it
straight from the environment -- nothing in this codebase ever reads or
prints HF_TOKEN itself.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # populates os.environ from a .env file, if one exists

# --- Embedding model ---------------------------------------------------
# Override via EMBEDDING_MODEL_NAME in .env to swap models without
# touching any code.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Hugging Face token --------------------------------------------------
# Optional. all-MiniLM-L6-v2 is a public model and works with no token
# at all. Set HF_TOKEN in .env only if you switch to a gated/private
# model, or want to avoid anonymous-download rate limits.
# huggingface_hub picks HF_TOKEN up automatically from the environment
# once load_dotenv() above has run -- no code here needs to pass it
# around explicitly.
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Fallback chunking (used only for documents matching none of the
#     3 known formats -- see ingestion/chunker.py) ----------------------
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

# --- Default paths -------------------------------------------------------
CORPUS_DIR = os.getenv("CORPUS_DIR", "corpus")
DATA_DIR = os.getenv("DATA_DIR", "data")


def _bool_env(name, default):
    """Parse an env var as a bool ('true'/'1'/'yes' -> True), used for
    the retrieval ablation toggles below."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# --- Retrieval (hybrid search: semantic + BM25 + RRF + reranker) ---------
# Each stage is toggleable so the same search() code path can run three
# ways for the evaluation report's ablation:
#   semantic only                  -> USE_BM25=False, USE_RERANKER=False
#   semantic + BM25 + RRF          -> USE_BM25=True,  USE_RERANKER=False
#   semantic + BM25 + RRF + rerank -> USE_BM25=True,  USE_RERANKER=True
USE_BM25 = _bool_env("USE_BM25", True)
USE_RERANKER = _bool_env("USE_RERANKER", True)

# How many candidates each retriever pulls before fusion, and how many
# fused candidates get passed into the reranker.
TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
TOP_K_RERANK_CANDIDATES = int(os.getenv("TOP_K_RERANK_CANDIDATES", "10"))

# Reciprocal Rank Fusion constant -- 60 is the standard value from the
# original RRF paper; higher values flatten the difference between
# top-ranked and lower-ranked results.
RRF_K = int(os.getenv("RRF_K", "60"))

# Small, CPU-friendly cross-encoder for the reranking stage.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Logging ---------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logging():
    """Call once, at the start of the pipeline (or a notebook cell),
    to turn on consistent logging across every ingestion module."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )