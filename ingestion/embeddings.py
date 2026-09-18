"""
ingestion/embeddings.py
-------------------------
Embedding-model factory. Two providers, switched via
config.EMBEDDING_PROVIDER:

  - "huggingface" (default): HuggingFaceEmbeddings -- free, runs
    locally, downloads the model the first time
    (config.EMBEDDING_MODEL_NAME, e.g.
    "sentence-transformers/all-MiniLM-L6-v2").
  - "openai": OpenAIEmbeddings -- a paid API call per chunk/query (e.g.
    config.EMBEDDING_MODEL_NAME = "text-embedding-3-small"), needs
    OPENAI_API_KEY in .env. OpenAI's embeddings are already normalized
    to unit length by their API, so this works as a drop-in for the
    existing DistanceStrategy.MAX_INNER_PRODUCT setup with no other
    changes needed.

IMPORTANT: switching either the provider or the model name changes the
embedding vector space -- a FAISS index built with one embedding model
is NOT compatible with a different one. After changing this, you must
re-run the ingestion pipeline (python -m ingestion.pipeline) to rebuild
data/ from scratch; otherwise search() silently compares incompatible
vectors and returns meaningless results with no error at all.
"""

import logging

import config

logger = logging.getLogger(__name__)

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info(
            "Loading embedding model: provider=%s model=%s",
            config.EMBEDDING_PROVIDER, config.EMBEDDING_MODEL_NAME,
        )
        if config.EMBEDDING_PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info("Embedding model ready")
    return _embeddings