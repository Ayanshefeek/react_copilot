"""
api/main.py
-------------
FastAPI layer over the ReAct agent. The agent is built ONCE at startup
(via the lifespan context manager) and reused across requests -- rebuilding
it per-request would re-load the LLM/tools/retrieval wiring every call,
which is unnecessary since get_llm()/get_embeddings() are already
module-level singletons and the agent graph itself is cheap to reuse.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from agent.react_agent import build_agent, run_agent
from api.schemas import AnswerResponse, ConfigResponse, HealthResponse, QuestionRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.setup_logging()
    logger.info("Building ReAct agent (this loads the LLM, embeddings and index once)...")
    app.state.agent = build_agent()
    logger.info("Agent ready -- API is up.")
    yield
    logger.info("Shutting down API.")


app = FastAPI(title="ReAct Research Copilot API", lifespan=lifespan)

# Allow the local Streamlit app (default port 8501) to call this API.
# Add any other origin here if you serve the UI from somewhere else.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    agent_ready = getattr(app.state, "agent", None) is not None
    return HealthResponse(
        status="ok" if agent_ready else "starting",
        agent_ready=agent_ready,
        llm_provider=config.LLM_PROVIDER,
        embedding_provider=config.EMBEDDING_PROVIDER,
    )


@app.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(
        llm_provider=config.LLM_PROVIDER,
        llm_model_name=config.LLM_MODEL_NAME,
        embedding_provider=config.EMBEDDING_PROVIDER,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        use_bm25=config.USE_BM25,
        use_reranker=config.USE_RERANKER,
    )


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    agent = getattr(app.state, "agent", None)
    if agent is None:
        # Should not normally happen -- lifespan sets this before the API
        # starts accepting traffic -- but guarded in case of a race.
        raise HTTPException(status_code=503, detail="Agent is not ready yet, try again shortly.")

    result = run_agent(request.question, agent=agent)

    if result["error"]:
        # A genuine failure (timeout or unexpected exception) -- NOT the
        # "ran out of steps" case, which run_agent() now reports as a
        # normal (if unsuccessful) answer with error=None.
        logger.error("Agent run failed for question=%r: %s", request.question, result["error"])
        raise HTTPException(status_code=500, detail=result["error"])

    return AnswerResponse(
        question=result["question"],
        trace=result["trace"],
        answer=result["answer"],
        citations_valid=result["citations_valid"],
        citation_problems=result["citation_problems"],
        elapsed_seconds=result["elapsed_seconds"],
    )