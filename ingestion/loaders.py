"""
loaders.py
----------
Reads raw files off disk. That's it -- no format detection, no chunking.
Markdown files become a text string, PDF files become a list of pages,
each page a list of lines with a "bold" flag we'll use later to find
the title.

Nothing here modifies the original files.
"""

import logging
import os

import fitz  # pymupdf

logger = logging.getLogger(__name__)


def load_markdown(path):
    """Read a .md file as plain text."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    logger.debug("Loaded markdown file: %s (%d chars)", os.path.basename(path), len(text))
    return {
        "filename": os.path.basename(path),
        "file_type": "md",
        "text": text,
    }


def load_pdf(path):
    """Read a .pdf file page by page. Each line keeps whether it's bold,
    so parsers.py can find the title (the bold heading) later without
    re-opening the file."""
    pages = []
    doc = fitz.open(path)
    for page_number, page in enumerate(doc, start=1):
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                # A line is "bold" if its font name says so (e.g.
                # "Helvetica-Bold") -- this is how the sample PDF's
                # heading is actually set.
                is_bold = any("Bold" in s["font"] for s in spans)
                lines.append({"text": text, "bold": is_bold})
        pages.append({"page_number": page_number, "lines": lines})
    doc.close()

    logger.debug("Loaded PDF file: %s (%d pages)", os.path.basename(path), len(pages))
    return {
        "filename": os.path.basename(path),
        "file_type": "pdf",
        "pages": pages,
    }


def load_corpus(corpus_dir):
    """Load every .md and .pdf file in corpus_dir. Anything else is
    skipped and returned separately so it can be reported, not silently
    ignored."""
    documents = []
    skipped = []
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext == ".md":
            documents.append(load_markdown(path))
        elif ext == ".pdf":
            documents.append(load_pdf(path))
        else:
            logger.warning("Skipping unsupported file: %s", name)
            skipped.append(name)

    logger.info(
        "Loaded %d documents from %s (%d skipped)",
        len(documents), corpus_dir, len(skipped),
    )
    return documents, skipped
