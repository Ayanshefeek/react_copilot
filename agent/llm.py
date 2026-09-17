"""
agent/llm.py
-------------
LLM factory for the ReAct agent.

Two paths, both selected purely via config.LLM_PROVIDER:

  - config.LLM_PROVIDER = "huggingface_endpoint": uses HF's hosted
    Inference API for free (HuggingFaceEndpoint + ChatHuggingFace), no
    local download, needs only config.HF_TOKEN (already in .env).
    Tool-calling reliability depends on the chosen model -- small free
    models are noticeably less consistent at it than OpenAI/Anthropic,
    worth noting as a limitation in the evaluation report if it shows up.

  - anything else (e.g. "openai", "anthropic", "groq", ...): goes
    through LangChain's init_chat_model, which handles those providers
    uniformly given the matching API key in .env.

Either way, the rest of the agent code just calls get_llm() and never
needs to know which path was used.
"""

import logging

import config

logger = logging.getLogger(__name__)

_llm = None  # lazy singleton, same pattern as ingestion/embeddings.py


def _build_huggingface_endpoint_llm():
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=config.LLM_MODEL_NAME,
        huggingfacehub_api_token=config.HF_TOKEN,
        # HF's endpoint rejects temperature=0.0 outright -- clamp to a
        # small positive value so config.LLM_TEMPERATURE=0.0 (the
        # default, for deterministic-ish answers) still works.
        temperature=max(config.LLM_TEMPERATURE, 0.01),
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)


def get_llm():
    global _llm
    if _llm is None:
        logger.info("Loading LLM: provider=%s model=%s", config.LLM_PROVIDER, config.LLM_MODEL_NAME)

        if config.LLM_PROVIDER == "huggingface_endpoint":
            _llm = _build_huggingface_endpoint_llm()
        else:
            from langchain.chat_models import init_chat_model
            _llm = init_chat_model(
                config.LLM_MODEL_NAME,
                model_provider=config.LLM_PROVIDER,
                temperature=config.LLM_TEMPERATURE,
            )

        logger.info("LLM ready")
    return _llm