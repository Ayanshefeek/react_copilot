"""
test_ingestion.py
------------------
Each test maps to one of the 10 validation requirements from the project
spec, run against the sample corpus in corpus/.

Run:
    pytest tests/test_ingestion.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunker import fixed_size_chunks
from ingestion.loaders import load_corpus, load_markdown, load_pdf
from ingestion.parsers import make_chunks, parse_document
from ingestion.pipeline import chunk_corpus

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")
REF_SENTENCES_FILE = os.path.join(CORPUS_DIR, "01_ai_evals.md")
KEY_POINTS_FILE = os.path.join(CORPUS_DIR, "13_ai_evals.md")
PDF_FILE = os.path.join(CORPUS_DIR, "guide_evidence_citations.pdf")
UNKNOWN_FILE = os.path.join(CORPUS_DIR, "99_unstructured_notes.md")


# 1. Format 1 -> exactly one chunk per Reference Sentence
def test_format1_one_chunk_per_reference_sentence():
    doc = load_markdown(REF_SENTENCES_FILE)
    fmt, title, items = parse_document(doc)
    assert fmt == "reference_sentences"
    chunks = make_chunks(doc, fmt, title, items)
    assert [c["text"] for c in chunks] == [
        "Evaluation should separate retrieval quality from generation quality.",
        "Common dimensions include relevance, faithfulness (groundedness), and robustness.",
        "A simple regression suite is a fixed set of prompts run on every change.",
    ]


# 2. Format 2 -> exactly one chunk per Key Point
def test_format2_one_chunk_per_key_point():
    doc = load_markdown(KEY_POINTS_FILE)
    fmt, title, items = parse_document(doc)
    assert fmt == "key_points"
    chunks = make_chunks(doc, fmt, title, items)
    assert len(chunks) == 3


# 3. Format 2 excludes Summary and Notes for Practitioners
def test_format2_excludes_summary_and_notes():
    doc = load_markdown(KEY_POINTS_FILE)
    fmt, title, items = parse_document(doc)
    chunks = make_chunks(doc, fmt, title, items)
    all_text = " ".join(c["text"] for c in chunks).lower()
    assert "cite the specific sentence" not in all_text
    assert "avoid adding extra facts" not in all_text
    assert all(c["section"] == "Key Points" for c in chunks)


# 4. Format 3 -> exactly one chunk per numbered statement
def test_format3_one_chunk_per_numbered_statement():
    doc = load_pdf(PDF_FILE)
    fmt, title, items = parse_document(doc)
    assert fmt == "numbered_statements"
    chunks = make_chunks(doc, fmt, title, items)
    assert len(chunks) == 3
    assert chunks[0]["text"] == "Cite sources using a consistent format such as [file:page]."


# 5. Format 3 ignores the trailing boilerplate sentence
def test_format3_ignores_boilerplate():
    doc = load_pdf(PDF_FILE)
    fmt, title, items = parse_document(doc)
    chunks = make_chunks(doc, fmt, title, items)
    all_text = " ".join(c["text"] for c in chunks).lower()
    assert "synthetic" not in all_text


# 6. PDF page numbers are preserved
def test_pdf_page_numbers_preserved():
    doc = load_pdf(PDF_FILE)
    fmt, title, items = parse_document(doc)
    chunks = make_chunks(doc, fmt, title, items)
    assert all(c["page"] == 1 for c in chunks)


# 7. Titles are preserved as metadata but never in the chunk text
def test_titles_preserved_but_not_in_text():
    doc = load_markdown(REF_SENTENCES_FILE)
    fmt, title, items = parse_document(doc)
    chunks = make_chunks(doc, fmt, title, items)
    assert chunks[0]["title"] == "Practical LLM Evaluation Basics (Sample Corpus)"
    assert all(c["title"] not in c["text"] for c in chunks)


# 8. Source filenames are preserved as metadata
def test_source_filenames_preserved():
    doc = load_markdown(REF_SENTENCES_FILE)
    fmt, title, items = parse_document(doc)
    chunks = make_chunks(doc, fmt, title, items)
    assert all(c["source"] == "01_ai_evals.md" for c in chunks)


# 9. No LLM is used during parsing/chunking (static check)
def test_no_llm_imports_in_parsing_modules():
    """chunker.py imports langchain_text_splitters, which is pure text
    splitting (no LLM calls) -- so we check for actual LLM SDK/client
    names, not the word "langchain" itself."""
    import ingestion.parsers as parsers_module
    import ingestion.chunker as chunker_module
    forbidden = ["openai", "anthropic", "cohere", "chat_models", "chatopenai", "chatanthropic"]
    for module in (parsers_module, chunker_module):
        with open(module.__file__, encoding="utf-8") as f:
            src = f.read().lower()
        for term in forbidden:
            assert term not in src


# 10. Known formats never use the fixed-size fallback
def test_known_formats_never_use_fallback():
    for path, loader in [
        (REF_SENTENCES_FILE, load_markdown),
        (KEY_POINTS_FILE, load_markdown),
        (PDF_FILE, load_pdf),
    ]:
        doc = loader(path)
        fmt, title, items = parse_document(doc)
        assert fmt != "unknown"
        chunks = make_chunks(doc, fmt, title, items)
        assert all(c["format"] != "fallback_fixed_size" for c in chunks)


# Extra: fallback only triggers for genuinely unmatched documents
def test_fallback_used_for_unknown_format_only():
    doc = load_markdown(UNKNOWN_FILE)
    fmt, title, items = parse_document(doc)
    assert fmt == "unknown"
    chunks = fixed_size_chunks(doc, title)
    assert len(chunks) > 0
    assert all(c["format"] == "fallback_fixed_size" for c in chunks)


# Extra: full corpus loads and every chunk has the required fields
def test_full_corpus_loads_and_every_chunk_has_required_fields():
    documents, skipped = load_corpus(CORPUS_DIR)
    assert len(documents) == 4  # 3 known-format + 1 unknown-format fixture

    chunks, fallback_files = chunk_corpus(documents)
    assert len(chunks) > 0
    assert fallback_files == ["99_unstructured_notes.md"]

    required = {"chunk_id", "text", "source", "title", "file_type", "format", "section", "page", "chunk_index"}
    for chunk in chunks:
        assert required.issubset(chunk.keys())
        assert chunk["text"].strip() != ""


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
