"""
agent/llm.py
-------------
LLM factories for the project. Two separate lazy singletons:

  - get_llm(): the "main" LLM -- powers the ReAct agent and the
    no-tools baseline (eval/baseline.py). Controlled by
    config.LLM_PROVIDER / LLM_MODEL_NAME / LLM_TEMPERATURE.
  - get_judge_llm(): the LLM used for evaluation judging (grounded
    precision, accuracy) -- controlled by config.JUDGE_LLM_PROVIDER /
    JUDGE_LLM_MODEL_NAME / JUDGE_LLM_TEMPERATURE, which default to the
    SAME values as the main LLM if left unset in .env. This means the
    judge is only a separate model if you explicitly configure it as
    one -- e.g. point the judge at the free huggingface_endpoint route
    while keeping the main agent/baseline on a paid provider, to cut
    evaluation cost without touching answer quality.

Both paths -- "huggingface_endpoint" (HF's free hosted API) or anything
else via LangChain's init_chat_model -- go through the shared
_build_llm() helper below, so there's no duplicated provider logic
between the two singletons.
"""

import logging

import config

logger = logging.getLogger(__name__)

_llm = None
_judge_llm = None


def _build_llm(provider, model_name, temperature):
    if provider == "huggingface_endpoint":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=config.HF_TOKEN,
            # HF's endpoint rejects temperature=0.0 outright -- clamp to a
            # small positive value so temperature=0.0 still works.
            temperature=max(temperature, 0.01),
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)

    from langchain.chat_models import init_chat_model
    return init_chat_model(model_name, model_provider=provider, temperature=temperature)


def get_llm():
    """The main LLM: powers the ReAct agent and the no-tools baseline."""
    global _llm
    if _llm is None:
        logger.info("Loading main LLM: provider=%s model=%s", config.LLM_PROVIDER, config.LLM_MODEL_NAME)
        _llm = _build_llm(config.LLM_PROVIDER, config.LLM_MODEL_NAME, config.LLM_TEMPERATURE)
        logger.info("Main LLM ready")
    return _llm


def get_judge_llm():
    """The evaluation-judge LLM: grounded precision + accuracy scoring.
    Defaults to the exact same config as get_llm() unless
    JUDGE_LLM_PROVIDER/JUDGE_LLM_MODEL_NAME are set separately in .env
    -- e.g. to point judging at the free HF endpoint while the main
    agent stays on a paid provider."""
    global _judge_llm
    if _judge_llm is None:
        logger.info(
            "Loading judge LLM: provider=%s model=%s",
            config.JUDGE_LLM_PROVIDER, config.JUDGE_LLM_MODEL_NAME,
        )
        _judge_llm = _build_llm(config.JUDGE_LLM_PROVIDER, config.JUDGE_LLM_MODEL_NAME, config.JUDGE_LLM_TEMPERATURE)
        logger.info("Judge LLM ready")
    return _judge_llm