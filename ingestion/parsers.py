"""
parsers.py
----------
Format detection + extraction for the 3 known document formats.

Approach: whitelist, not blacklist. Each extractor only pulls in the
lines that positively match what a chunk should be for that format (a
line under "## Reference Sentences", a bullet under "## Key Points", a
"N. " line in a PDF). Everything else -- titles, summaries, notes,
trailing disclaimers -- is excluded automatically because it never
matches, instead of us having to hardcode every possible boilerplate
string.

No LLM. Just string/regex matching. Fully deterministic.

A chunk is a plain dict with these keys:
    chunk_id, text, source, title, file_type, format, section, page, chunk_index
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

H1_RE = re.compile(r"^#\s+(.*\S)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
BULLET_RE = re.compile(r"^[-*]\s+(.*\S)\s*$")
NUMBERED_LINE_RE = re.compile(r"^\d+\.\s+(.*\S)\s*$")


def _title_from_markdown(text):
    """Title = the first H1 ('# ...') line."""
    for line in text.splitlines():
        m = H1_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def _section_lines(text, heading_name):
    """Return the raw lines under `## <heading_name>`, stopping at the
    next heading. Returns None if that heading isn't in the file."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if m and m.group(2).strip().lower() == heading_name:
            start = i + 1
            break
    if start is None:
        return None

    body = []
    for line in lines[start:]:
        m = HEADING_RE.match(line.strip())
        if m and len(m.group(1)) <= 2:  # next "## ..." section starts here
            break
        body.append(line)
    return body


def detect_markdown_format(doc):
    """'reference_sentences' if a '## Reference Sentences' heading exists,
    'key_points' if a '## Key Points' heading exists, else 'unknown'."""
    headings = {
        m.group(2).strip().lower()
        for m in (HEADING_RE.match(l.strip()) for l in doc["text"].splitlines())
        if m
    }
    if "reference sentences" in headings:
        return "reference_sentences"
    if "key points" in headings:
        return "key_points"
    return "unknown"


def parse_reference_sentences(doc):
    """FORMAT 1: every non-blank line under '## Reference Sentences' is
    one chunk. Split on the newline character, not on periods -- each
    sentence already sits on its own line in this format."""
    body = _section_lines(doc["text"], "reference sentences") or []
    title = _title_from_markdown(doc["text"])
    items = [(line.strip(), "Reference Sentences", None) for line in body if line.strip()]
    return "reference_sentences", title, items


def parse_key_points(doc):
    """FORMAT 2: only bullet lines under '## Key Points' become chunks.
    Summary and Notes for Practitioners are never even read."""
    body = _section_lines(doc["text"], "key points") or []
    title = _title_from_markdown(doc["text"])
    items = []
    for line in body:
        m = BULLET_RE.match(line.strip())
        if m:
            items.append((m.group(1).strip(), "Key Points", None))
    return "key_points", title, items


def detect_pdf_format(doc):
    """'numbered_statements' if any line matches 'N. <statement>'."""
    for page in doc["pages"]:
        for line in page["lines"]:
            if NUMBERED_LINE_RE.match(line["text"].strip()):
                return "numbered_statements"
    return "unknown"


def _title_from_pdf(doc):
    """Title = the first bold line on page 1."""
    if not doc["pages"]:
        return None
    for line in doc["pages"][0]["lines"]:
        if line["bold"]:
            return line["text"].strip()
    return None


def parse_numbered_statements(doc):
    """FORMAT 3: only 'N. <statement>' lines become chunks. The bold
    heading and any trailing disclaimer are excluded automatically
    because they don't match the numbered pattern -- we never hardcode
    the disclaimer's wording."""
    title = _title_from_pdf(doc)
    items = []
    for page in doc["pages"]:
        for line in page["lines"]:
            m = NUMBERED_LINE_RE.match(line["text"].strip())
            if m:
                items.append((m.group(1).strip(), None, page["page_number"]))
    return "numbered_statements", title, items


def parse_document(doc):
    """Detect + extract in one call. Returns (format, title, items) where
    items is a list of (text, section, page) tuples. format='unknown'
    means none of the 3 known formats matched -- pipeline.py then routes
    the document to the fixed-size fallback instead of failing."""
    if doc["file_type"] == "md":
        fmt = detect_markdown_format(doc)
        if fmt == "reference_sentences":
            result = parse_reference_sentences(doc)
        elif fmt == "key_points":
            result = parse_key_points(doc)
        else:
            result = "unknown", _title_from_markdown(doc["text"]), []

    elif doc["file_type"] == "pdf":
        fmt = detect_pdf_format(doc)
        if fmt == "numbered_statements":
            result = parse_numbered_statements(doc)
        else:
            result = "unknown", _title_from_pdf(doc), []

    else:
        raise ValueError(f"Unsupported file_type: {doc['file_type']}")

    logger.debug("Detected format '%s' for %s (%d items)", result[0], doc["filename"], len(result[2]))
    return result


def make_chunks(doc, fmt, title, items):
    """Turn (text, section, page) tuples into normalized chunk dicts.

    chunk_id = '<filename-without-extension>_<3-digit-index>', e.g.
    '01_ai_evals_001'. Deterministic, so re-running the pipeline
    reproduces the same IDs."""
    stem = os.path.splitext(doc["filename"])[0]
    chunks = []
    for i, (text, section, page) in enumerate(items, start=1):
        chunks.append({
            "chunk_id": f"{stem}_{i:03d}",
            "text": text,
            "source": doc["filename"],
            "title": title,
            "file_type": doc["file_type"],
            "format": fmt,
            "section": section,
            "page": page,
            "chunk_index": i,
        })
    logger.debug("Built %d '%s' chunks for %s", len(chunks), fmt, doc["filename"])
    return chunks
