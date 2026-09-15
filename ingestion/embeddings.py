"""
embeddings.py
-------------
Local embedding model, wrapped through LangChain's Embeddings interface
(HuggingFaceEmbeddings) so the rest of the pipeline talks to embeddings
the same way it talks to every other LangChain component. No LLM, no API
key required for the default public model.

The model name comes from config.py (itself overridable via
EMBEDDING_MODEL_NAME in .env), so swapping models never means touching
this file.
"""

import logging

from langchain_huggingface import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)

_embeddings = None  # lazy singleton -- loaded once, reused everywhere


def get_embeddings():
    """Return a LangChain Embeddings object. embed_documents(texts) and
    embed_query(text) are the two methods the rest of the pipeline uses.

    normalize_embeddings=True makes embed output unit vectors, so cosine
    similarity == inner product -- this is what lets vector_store.py use
    FAISS's MAX_INNER_PRODUCT distance strategy and get similarity
    scores in a familiar 0-1-ish range.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready")
    return _embeddings
