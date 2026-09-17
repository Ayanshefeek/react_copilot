"""
tests/test_agent.py
---------------------
Tests for the ReAct agent loop (agent/) and its guardrails. Uses a
scripted fake chat model (FakeToolCallingModel below) instead of a real
LLM -- no API key, no network call, no cost, and fully deterministic.

Covers:
  - citation/quote validation logic in isolation
  - a full happy-path run: search -> read_chunk -> cited final answer
  - a hallucinated quote getting caught
  - the MAX_AGENT_STEPS guardrail tripping on a runaway tool-call loop
  - the AGENT_TIMEOUT_SECONDS guardrail tripping on a slow tool
  - retry: a tool that fails once then succeeds still returns a result

Run:
    pytest tests/test_agent.py -v
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import config
from agent.guardrails import call_with_retry, validate_citations
from agent.react_agent import run_agent


class FakeToolCallingModel(FakeMessagesListChatModel):
    """A scripted stand-in for a real tool-calling LLM: replays a fixed
    list of AIMessages (with or without tool_calls) in order, ignoring
    whatever it's actually asked. bind_tools is a no-op (returns self)
    since the scripted messages already encode any tool_calls -- the
    real HuggingFace/OpenAI/etc. models implement bind_tools properly,
    this is purely a test double."""

    def bind_tools(self, tools, **kwargs):
        return self


CHUNK_TEXT = "Faithfulness means the answer is grounded in the retrieved context."


# ---------------------------------------------------------------------------
# Citation/quote validation (no agent involved)
# ---------------------------------------------------------------------------
def test_validate_citations_accepts_well_formed_answer():
    answer = f'Faithfulness means grounding. [doc1.md] "{CHUNK_TEXT}"'
    is_valid, problems = validate_citations(answer, observed_chunk_texts=[CHUNK_TEXT])
    assert is_valid
    assert problems == []


def test_validate_citations_rejects_missing_citation():
    is_valid, problems = validate_citations("Faithfulness means grounding, no citation here.", observed_chunk_texts=[CHUNK_TEXT])
    assert not is_valid
    assert any("no citations" in p for p in problems)


def test_validate_citations_rejects_hallucinated_quote():
    answer = 'Faithfulness means X. [doc1.md] "this text was never actually retrieved"'
    is_valid, problems = validate_citations(answer, observed_chunk_texts=[CHUNK_TEXT])
    assert not is_valid
    assert any("not found verbatim" in p for p in problems)


def test_validate_citations_rejects_overlong_quote(monkeypatch):
    monkeypatch.setattr(config, "MAX_QUOTE_CHARS", 10)
    answer = f'Faithfulness means grounding. [doc1.md] "{CHUNK_TEXT}"'
    is_valid, problems = validate_citations(answer, observed_chunk_texts=[CHUNK_TEXT])
    assert not is_valid
    assert any("exceeds MAX_QUOTE_CHARS" in p for p in problems)


def test_validate_citations_handles_pdf_page_format():
    answer = 'Evaluation should separate concerns. [report.pdf:4] "separate retrieval quality from generation quality"'
    is_valid, problems = validate_citations(
        answer, observed_chunk_texts=["Evaluation should separate retrieval quality from generation quality."]
    )
    assert is_valid


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def test_call_with_retry_succeeds_after_transient_failure(monkeypatch):
    monkeypatch.setattr(config, "TOOL_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(config, "TOOL_RETRY_BACKOFF_SECONDS", 0.0)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient failure")
        return "ok"

    assert call_with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_call_with_retry_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(config, "TOOL_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(config, "TOOL_RETRY_BACKOFF_SECONDS", 0.0)

    def always_fails():
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError):
        call_with_retry(always_fails)


# ---------------------------------------------------------------------------
# Full agent loop (scripted fake LLM, real tool functions)
#
# Named "search"/"read_chunk" -- NOT "fake_search"/"fake_read_chunk" --
# because @tool names a tool after the function itself, and
# agent/react_agent.py's _extract_trace() matches on the literal action
# name "read_chunk" to decide which observations count as chunk text the
# agent actually read (for the citation-hallucination check). These names
# have to match the real production tool names for that check to fire.
# ---------------------------------------------------------------------------
@tool
def search(query: str) -> list:
    """search docs"""
    return [{"chunk_id": "doc1_001", "source": "doc1.md", "section": None, "page": None, "score": 0.9}]


@tool
def read_chunk(chunk_id: str) -> str:
    """read chunk text"""
    return CHUNK_TEXT


def test_agent_happy_path_produces_valid_citation():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "faithfulness"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "read_chunk", "args": {"chunk_id": "doc1_001"}, "id": "c2"}]),
        AIMessage(content=f'Faithfulness means grounding in context. [doc1.md] "{CHUNK_TEXT}"'),
    ]
    agent = create_agent(FakeToolCallingModel(responses=responses), [search, read_chunk], system_prompt="test")

    result = run_agent("What is faithfulness?", agent=agent)

    assert result["error"] is None
    assert result["citations_valid"] is True
    assert len(result["trace"]) == 2
    assert result["trace"][0]["action"] == "search"
    assert result["trace"][1]["action"] == "read_chunk"


def test_agent_catches_hallucinated_quote():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "faithfulness"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "read_chunk", "args": {"chunk_id": "doc1_001"}, "id": "c2"}]),
        AIMessage(content='Faithfulness means X. [doc1.md] "this was never actually retrieved"'),
    ]
    agent = create_agent(FakeToolCallingModel(responses=responses), [search, read_chunk], system_prompt="test")

    result = run_agent("q", agent=agent)

    assert result["error"] is None  # the run itself succeeded
    assert result["citations_valid"] is False  # but the citation is bogus
    assert any("not found verbatim" in p for p in result["citation_problems"])


def test_agent_trips_max_steps_guardrail():
    # A model that only ever calls a tool, never produces a final answer.
    loop_responses = [
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": f"c{i}"}])
        for i in range(20)
    ]
    agent = create_agent(FakeToolCallingModel(responses=loop_responses), [search, read_chunk], system_prompt="test")

    result = run_agent("q", agent=agent, max_steps=2)

    assert result["error"] is not None
    assert "MAX_AGENT_STEPS" in result["error"]


def test_agent_trips_timeout_guardrail():
    @tool
    def slow_tool(query: str) -> list:
        """a tool that takes too long"""
        time.sleep(2)
        return []

    responses = [AIMessage(content="", tool_calls=[{"name": "slow_tool", "args": {"query": "x"}, "id": "c1"}])]
    agent = create_agent(FakeToolCallingModel(responses=responses), [slow_tool], system_prompt="test")

    result = run_agent("q", agent=agent, timeout_seconds=1)

    assert result["error"] is not None
    assert "timed out" in result["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))