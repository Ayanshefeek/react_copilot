"""
pipeline.py
-----------
Wires the other modules together:

    load files -> detect format + extract -> (fallback if unknown)
    -> embed (LangChain HuggingFaceEmbeddings)
    -> FAISS index + chunk store (LangChain FAISS)
    -> print report

Run:
    python -m ingestion.pipeline --corpus corpus --out data
"""

import argparse
import logging
from collections import Counter

import config
from ingestion.chunker import fixed_size_chunks
from ingestion.embeddings import get_embeddings
from ingestion.loaders import load_corpus
from ingestion.parsers import make_chunks, parse_document
from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


def chunk_corpus(documents):
    """Extract chunks for every document. Anything that matches none of
    the 3 known formats falls back to fixed-size chunking instead of
    aborting the whole run.

    Returns (all_chunks, list_of_filenames_that_used_the_fallback).
    """
    all_chunks = []
    fallback_files = []

    for doc in documents:
        fmt, title, items = parse_document(doc)
        if fmt == "unknown":
            fallback_files.append(doc["filename"])
            chunks = fixed_size_chunks(doc, title)
        else:
            chunks = make_chunks(doc, fmt, title, items)
        all_chunks.extend(chunks)

    logger.info(
        "Chunked %d documents into %d chunks (%d used the fallback)",
        len(documents), len(all_chunks), len(fallback_files),
    )
    return all_chunks, fallback_files


def print_report(documents, chunks, fallback_files, skipped_files):
    md_count = sum(1 for d in documents if d["file_type"] == "md")
    pdf_count = sum(1 for d in documents if d["file_type"] == "pdf")
    counts = Counter(c["format"] for c in chunks)

    print("=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    print(f"Total documents: {len(documents)}\n")
    print(f"Markdown documents: {md_count}")
    print(f"PDF documents: {pdf_count}\n")
    print(f"Total chunks: {len(chunks)}\n")
    print(f"Format 1 (reference_sentences) chunks: {counts.get('reference_sentences', 0)}")
    print(f"Format 2 (key_points) chunks: {counts.get('key_points', 0)}")
    print(f"Format 3 (numbered_statements) chunks: {counts.get('numbered_statements', 0)}")
    print(f"Fallback (fallback_fixed_size) chunks: {counts.get('fallback_fixed_size', 0)}")

    if fallback_files:
        print(f"\nDocuments that used the fallback ({len(fallback_files)}):")
        for name in fallback_files:
            print(f"  - {name}  (did not match Format 1/2/3)")

    if skipped_files:
        print(f"\nFiles skipped (unsupported extension, {len(skipped_files)}):")
        for name in skipped_files:
            print(f"  - {name}")

    print("\nSample chunks:")
    print("-" * 60)
    for chunk in chunks[:5]:
        for key, value in chunk.items():
            print(f"{key}: {value!r}")
        print("-" * 60)


def run_pipeline(corpus_dir=None, out_dir=None):
    corpus_dir = corpus_dir or config.CORPUS_DIR
    out_dir = out_dir or config.DATA_DIR

    logger.info("Starting ingestion pipeline (corpus=%s, out=%s)", corpus_dir, out_dir)

    documents, skipped = load_corpus(corpus_dir)
    chunks, fallback_files = chunk_corpus(documents)

    embeddings = get_embeddings()
    store = VectorStore(embeddings)
    store.add(chunks)
    store.save(out_dir)

    print_report(documents, chunks, fallback_files, skipped)
    print(f"\nSaved FAISS index + chunk store to: {out_dir}")

    logger.info("Ingestion pipeline complete")
    return store


if __name__ == "__main__":
    config.setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=config.CORPUS_DIR)
    parser.add_argument("--out", default=config.DATA_DIR)
    args = parser.parse_args()
    run_pipeline(args.corpus, args.out)
