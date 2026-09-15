# ReAct Research Copilot — Ingestion Pipeline

This is **only the ingestion stage** of the ReAct Research Copilot project.
It does not include the ReAct agent, the `search()`/`read_chunk()` tools, or
the evaluation harness — those come next, on top of the index this pipeline
produces.

## What it does

```
Raw documents (corpus/)
    -> format detection
    -> content extraction (semantic boundaries, not fixed-size windows)
    -> atomic chunk creation
    -> metadata creation
    -> embedding generation (LangChain HuggingFaceEmbeddings, all-MiniLM-L6-v2)
    -> FAISS vector index (LangChain FAISS vectorstore)
    -> persistent chunk store (data/chunk_store.json + LangChain's own index files)
```

Chunks are plain Python dicts with keys:
`chunk_id, text, source, title, file_type, format, section, page, chunk_index`.
Extraction/detection logic is unchanged from earlier iterations of this
pipeline — what changed is that chunking, embeddings, and the vector store
now go through LangChain instead of talking to faiss/sentence-transformers
directly, so the whole project uses one consistent framework.

Three known document formats are extracted using their own semantic
structure (never generic fixed-size chunking):

| Format | Source | Rule |
|---|---|---|
| `reference_sentences` | `.md` with a `## Reference Sentences` section | one chunk per line in that section |
| `key_points` | `.md` with a `## Key Points` section | one chunk per bullet in that section (Summary/Notes ignored) |
| `numbered_statements` | `.pdf` with `N. <statement>` lines | one chunk per numbered line; bold first line on page 1 = title |

Any document that matches **none** of the three formats is routed through a
clearly-labeled fallback (`format: "fallback_fixed_size"`) instead of
crashing the run or being silently dropped. The fallback uses LangChain's
`RecursiveCharacterTextSplitter` (paragraph breaks first, then sentences,
then words, only forcing a hard cut as a last resort). The ingestion report
always names which files hit the fallback.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # optional -- see below
```

The first run downloads `all-MiniLM-L6-v2` from Hugging Face (~90MB) and
caches it locally; every run after that is offline. **No token is required**
for this model — it's public.

### `.env` and `config.py`

All settings that might change between machines or people live in
`config.py`, which loads `.env` automatically via `python-dotenv`:

| Setting | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Only needed for a gated/private HF model, or to avoid anonymous-download rate limits. Read automatically by `huggingface_hub` once `.env` is loaded — no code references it directly. |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Swap embedding models without touching code. |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | `800` / `150` | Fallback-chunker tuning. |
| `CORPUS_DIR` / `DATA_DIR` | `corpus` / `data` | Default pipeline paths. |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for per-document/per-chunk trace logging. |

`.env` is never committed with real secrets in it — copy `.env.example` to
`.env` and fill in only what you need (usually nothing, for this model).

## Run it

```bash
python -m ingestion.pipeline --corpus corpus --out data
```

Logging (via `config.setup_logging()`) prints a running trace to the console
— which files loaded, which document hit the fallback and why, when the
embedding model loads, when the FAISS index is built and saved — followed
by the required ingestion report (document/chunk counts by format, fallback
files, and 5 sample chunks with full metadata). It writes to `data/`:

- LangChain's own FAISS index files (`index.faiss`, plus a pickle holding
  the docstore) — the real, reloadable index. `VectorStore.load()` reads
  these back with `FAISS.load_local(...)`.
- `chunk_store.json` — a plain, human-readable `chunk_id -> full chunk dict`
  mirror of the same data, so you (or a grader) can open it in a text
  editor without unpickling anything.

Re-running overwrites all of the above, so repeated runs never accumulate
duplicate vectors.

## Run the tests

```bash
pytest tests/test_ingestion.py -v
```

12 tests cover the 10 explicit validation requirements from the project
spec (exact per-sentence/per-bullet/per-statement chunk counts, Summary/Notes
exclusion, boilerplate exclusion, page-number preservation, title/source
metadata preservation, no-LLM-in-parsing, and no-fixed-size-chunking-on-known-formats)
plus two extra checks for the fallback path and full-corpus robustness. These
tests only exercise loading/parsing/chunking (no embedding calls), so they
run offline and fast.

## Project layout

```
config.py               # embedding model name, chunk sizes, paths, logging setup — loads .env
.env.example             # copy to .env; only HF_TOKEN is a secret, everything else is an override
ingestion/
    loaders.py            # read raw .md/.pdf files (PDF via pymupdf); PDF loader keeps per-line bold info
    parsers.py             # format detection + whitelist-based extraction -> normalized chunk dicts
    chunker.py              # ONLY the fallback for non-matching documents (LangChain RecursiveCharacterTextSplitter)
    embeddings.py           # LangChain HuggingFaceEmbeddings wrapper, model name from config.py
    vector_store.py         # LangChain FAISS vectorstore wrapper + JSON chunk store mirror
    pipeline.py              # orchestrates the above, logs each step, prints the ingestion report, CLI entry point
tests/
    test_ingestion.py       # the 10 validation requirements + 2 extra robustness checks
corpus/
    (drop your ~30 .md + 5 .pdf files here; a few samples are included)
data/
    (pipeline output lands here — empty until you run it)
```

## Sample corpus included

`corpus/` currently contains the 3 files you provided as format examples,
plus `99_unstructured_notes.md`, a synthetic file that intentionally matches
none of the 3 known formats — it exists purely to exercise the fallback
path in the test suite. Replace/add your real ~30+5 file corpus here before
running the full pipeline; nothing about the code needs to change.

## Design notes worth knowing before you build the ReAct agent on top of this

- **Extraction is whitelist-based, not blacklist-based.** Each parser only
  pulls in lines that positively match the expected pattern for that
  format. Titles, summaries, notes, and trailing disclaimers are excluded
  automatically by never matching — nothing is hardcoded against specific
  boilerplate wording.
- **PDF titles are detected by font boldness** (via pymupdf's per-span font
  name), not position or size alone — confirmed against
  `guide_evidence_citations.pdf`, where the heading is set in
  `Helvetica-Bold` and body text in plain `Helvetica`.
- **Original files are never modified.** Everything ignored is simply
  excluded from the chunk representation — it stays in the source file.
- **Why LangChain end-to-end:** chunking (`RecursiveCharacterTextSplitter`),
  embeddings (`HuggingFaceEmbeddings`), and the vector store
  (`langchain_community.vectorstores.FAISS`) all go through LangChain's
  interfaces now, so the ReAct agent stage (which will almost certainly use
  LangChain/LangGraph for the agent loop itself) has one consistent set of
  abstractions to build on instead of mixing raw faiss/sentence-transformers
  calls with LangChain agent code.
- **A note on the `langchain-community` deprecation warning:** LangChain's
  FAISS integration currently still lives in `langchain-community`, which
  prints a sunset notice on import. It's still the standard way to use FAISS
  with LangChain as of this writing; nothing to fix on your end.
