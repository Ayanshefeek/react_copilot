"""
vector_store.py
----------------
Wraps LangChain's FAISS vector store (langchain_community.vectorstores.FAISS)
instead of talking to the faiss library directly. LangChain's FAISS class
already manages the two things this project needs kept separate:

  - the FAISS index itself (vectors)
  - a docstore mapping id -> Document (page_content + metadata)

Distance strategy: MAX_INNER_PRODUCT over normalized embeddings, i.e.
cosine similarity -- same choice as a raw faiss.IndexFlatIP, just
expressed through LangChain's API. Flat/exact search is appropriate at
this corpus size; no need for an approximate index like IVF or HNSW.

LangChain's save_local()/load_local() persist the real index (as
index.faiss + a pickle). We additionally write a plain chunk_store.json
next to it purely so a human (or a grader) can open the chunks in a text
editor -- but load_local() is what the pipeline actually reloads from.

NOTE: search()/read_chunk() used to live here as a preview of the future
retrieval tools. They've moved to retrieval/tools.py now that retrieval
is actually being built (hybrid semantic + BM25 + RRF + reranker), so
this class stays scoped to exactly what ingestion needs: build, save,
load. `self.store` (the underlying LangChain FAISS instance) is public
so retrieval/semantic.py and retrieval/keyword.py can use it directly.
"""

import json
import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHUNK_STORE_FILENAME = "chunk_store.json"  # human-readable mirror only


class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None  # a langchain_community.vectorstores.FAISS instance

    def add(self, chunks):
        """chunks: list of normalized chunk dicts (see parsers.make_chunks
        / chunker.fixed_size_chunks). Wraps each as a LangChain Document
        (page_content=text, metadata=the whole chunk dict) and embeds +
        indexes them in one call."""
        if not chunks:
            logger.warning("add() called with zero chunks -- nothing to index")
            return

        documents = [Document(page_content=c["text"], metadata=c) for c in chunks]
        ids = [c["chunk_id"] for c in chunks]

        logger.info("Embedding %d chunks and building the FAISS index...", len(chunks))
        # normalize_L2 isn't needed here: embeddings.py already asks the
        # model to return normalized (unit-length) vectors, and
        # MAX_INNER_PRODUCT over normalized vectors *is* cosine similarity.
        self.store = FAISS.from_documents(
            documents,
            self.embeddings,
            ids=ids,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        logger.info("FAISS index built with %d vectors", len(ids))

    def __len__(self):
        if self.store is None:
            return 0
        return len(self.store.index_to_docstore_id)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.store.save_local(out_dir)
        logger.info("Saved FAISS index to %s", out_dir)

        chunk_store = {
            doc_id: self.store.docstore.search(doc_id).metadata
            for doc_id in self.store.index_to_docstore_id.values()
        }
        chunk_store_path = os.path.join(out_dir, CHUNK_STORE_FILENAME)
        with open(chunk_store_path, "w", encoding="utf-8") as f:
            json.dump(chunk_store, f, indent=2, ensure_ascii=False)
        logger.info("Wrote human-readable chunk store to %s", chunk_store_path)

    @classmethod
    def load(cls, out_dir, embeddings):
        store = cls(embeddings)
        # allow_dangerous_deserialization=True is safe here because we
        # only ever load an index this same pipeline wrote -- never a
        # file from an untrusted source.
        store.store = FAISS.load_local(
            out_dir, embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Loaded FAISS index from %s (%d vectors)", out_dir, len(store))
        return store