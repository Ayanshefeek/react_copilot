"""
api/schemas.py
----------------
Pydantic request/response models for the FastAPI layer. Kept separate
from api/main.py so the request-validation guardrails (question length,
non-empty) are easy to find and adjust in one place.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

import config


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_QUESTION_CHARS,
        description="The question to ask the corpus.",
    )


class StepTrace(BaseModel):
    step: int
    action: str
    action_input: Dict[str, Any] = {}
    observation: str


class AnswerResponse(BaseModel):
    question: str
    trace: List[StepTrace]
    answer: str
    citations_valid: bool
    citation_problems: List[str]
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    llm_provider: str
    embedding_provider: str


class ConfigResponse(BaseModel):
    llm_provider: str
    llm_model_name: str
    embedding_provider: str
    embedding_model_name: str
    use_bm25: bool
    use_reranker: bool