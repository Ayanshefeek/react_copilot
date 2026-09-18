import logging
import os
from dotenv import load_dotenv

load_dotenv()

# --- Ingestion ---
# provider: "huggingface" (default, free/local) | "openai" (paid API,
# e.g. "text-embedding-3-small" or "text-embedding-3-large")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN")
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
CORPUS_DIR = os.getenv("CORPUS_DIR", "corpus")
DATA_DIR = os.getenv("DATA_DIR", "data")


def _bool_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# --- Retrieval ---
USE_BM25 = _bool_env("USE_BM25", True)
USE_RERANKER = _bool_env("USE_RERANKER", True)
TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
TOP_K_RERANK_CANDIDATES = int(os.getenv("TOP_K_RERANK_CANDIDATES", "10"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Agent / LLM (main -- powers the ReAct agent and the eval baseline) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Judge LLM (grounded precision + accuracy scoring during evaluation) ---
JUDGE_LLM_PROVIDER = os.getenv("JUDGE_LLM_PROVIDER", LLM_PROVIDER)
JUDGE_LLM_MODEL_NAME = os.getenv("JUDGE_LLM_MODEL_NAME", LLM_MODEL_NAME)
JUDGE_LLM_TEMPERATURE = float(os.getenv("JUDGE_LLM_TEMPERATURE", str(LLM_TEMPERATURE)))

# --- Agent guardrails ---
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
TOOL_RETRY_ATTEMPTS = int(os.getenv("TOOL_RETRY_ATTEMPTS", "2"))
TOOL_RETRY_BACKOFF_SECONDS = float(os.getenv("TOOL_RETRY_BACKOFF_SECONDS", "1.0"))
MAX_QUOTE_CHARS = int(os.getenv("MAX_QUOTE_CHARS", "200"))
ALLOWED_TOOLS = ["search", "read_chunk"]

# --- Evaluation ---
EVAL_QUESTIONS_PATH = os.getenv("EVAL_QUESTIONS_PATH", "evaluation_questions.csv")
EVAL_RUNS_OUTPUT_PATH = os.getenv("EVAL_RUNS_OUTPUT_PATH", "runs.csv")
EVAL_REPORT_OUTPUT_PATH = os.getenv("EVAL_REPORT_OUTPUT_PATH", "evaluation_report.md")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )