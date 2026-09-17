"""
tests/test_eval.py
---------------------
Tests for the evaluation harness (eval/). Fully stubbed -- no real LLM
calls, no API key needed, deterministic.

Covers:
  - grounded_precision: zero-citation short-circuit (no LLM call), score
    parsing/clamping from a judge's raw text response
  - the LangChain-to-deepeval LLM adapter used for the accuracy metric
  - harness row-building: citation formatting, retrieved-files
    extraction from a trace, and the "llm"/"react" mode values written
    to runs.csv

Run:
    pytest tests/test_eval.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.messages import AIMessage

import config
from eval.accuracy import LangChainDeepEvalLLM
from eval.grounded_precision import _parse_score, compute_grounded_precision
from eval.harness import _format_citations, _retrieved_files_from_trace


# ---------------------------------------------------------------------------
# grounded_precision.py
# ---------------------------------------------------------------------------
def test_parse_score_reads_a_plain_decimal():
    assert _parse_score("0.67") == 0.67


def test_parse_score_clamps_above_one():
    assert _parse_score("1.5") == 1.0


def test_parse_score_clamps_below_zero():
    assert _parse_score("-0.2") == 0.0


def test_parse_score_raises_on_no_number():
    with pytest.raises(ValueError):
        _parse_score("no number in this response")


def test_compute_grounded_precision_short_circuits_on_zero_citations():
    """An answer with no citations at all -- the usual llm-mode baseline
    case -- should score 0.0 with NO LLM call made."""
    calls = {"n": 0}

    class ShouldNotBeCalledLLM:
        def invoke(self, messages):
            calls["n"] += 1
            raise AssertionError("LLM should not be called when there are no citations")

    score = compute_grounded_precision("This answer has no citations at all.", llm=ShouldNotBeCalledLLM())
    assert score == 0.0
    assert calls["n"] == 0


def test_compute_grounded_precision_calls_judge_when_citations_present(monkeypatch):
    monkeypatch.setattr(config, "TOOL_RETRY_ATTEMPTS", 1)

    class FakeJudgeLLM:
        def invoke(self, messages):
            return AIMessage(content="0.75")

    answer = 'Faithfulness means grounding. [doc1.md] "the answer is supported by evidence"'
    score = compute_grounded_precision(answer, llm=FakeJudgeLLM())
    assert score == 0.75


# ---------------------------------------------------------------------------
# eval/accuracy.py -- the LangChain-to-deepeval adapter
# ---------------------------------------------------------------------------
def test_langchain_deepeval_adapter_delegates_to_the_wrapped_llm():
    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(content=f"echo:{messages[0].content}")

    adapter = LangChainDeepEvalLLM(llm=FakeLLM())
    assert adapter.generate("hello") == "echo:hello"
    assert adapter.get_model_name() == "project-configured-llm"


# ---------------------------------------------------------------------------
# eval/harness.py -- pure formatting helpers
# ---------------------------------------------------------------------------
def test_format_citations_matches_the_answers_inline_citations():
    answer = 'X. [doc1.md] "quote one" and Y. [report.pdf:4] "quote two"'
    assert _format_citations(answer) == '[doc1.md] "quote one"; [report.pdf:4] "quote two"'


def test_format_citations_empty_when_no_citations():
    assert _format_citations("No citations here.") == ""


def test_retrieved_files_from_trace_dedupes_across_search_steps():
    trace = [
        {"action": "search", "observation": json.dumps([{"source": "doc1.md"}, {"source": "doc2.pdf"}])},
        {"action": "read_chunk", "observation": "some text"},
        {"action": "search", "observation": json.dumps([{"source": "doc1.md"}])},  # duplicate
    ]
    assert _retrieved_files_from_trace(trace) == "doc1.md; doc2.pdf"


def test_retrieved_files_from_trace_ignores_malformed_observation():
    trace = [{"action": "search", "observation": "not valid json"}]
    assert _retrieved_files_from_trace(trace) == ""


# ---------------------------------------------------------------------------
# eval/harness.py -- row building for each mode (stubbed pipelines)
# ---------------------------------------------------------------------------
def test_run_llm_mode_writes_mode_llm_and_zero_steps(monkeypatch):
    import eval.harness as harness_module

    monkeypatch.setattr(harness_module, "run_baseline", lambda q: {
        "question": q, "answer": 'Answer. [doc1.md] "quote"', "error": None,
    })
    monkeypatch.setattr(harness_module, "compute_grounded_precision", lambda a: 0.5)
    monkeypatch.setattr(harness_module, "compute_accuracy", lambda q, a, g: 0.8)

    row = {"id": "Q01", "question": "What is X?", "gold_source_file": "doc1.md", "gold_supporting_snippet": "X is..."}
    result = harness_module._run_llm_mode(row)

    assert result["mode"] == "llm"
    assert result["question_id"] == "Q01"
    assert result["steps_used"] == 0
    assert result["retrieved_files"] == ""
    assert result["grounded_precision"] == 0.5
    assert result["accuracy"] == 0.8
    assert result["citations"] == '[doc1.md] "quote"'
    assert result["notes"] == ""


def test_run_llm_mode_records_error_in_notes(monkeypatch):
    import eval.harness as harness_module

    monkeypatch.setattr(harness_module, "run_baseline", lambda q: {
        "question": q, "answer": "", "error": "Baseline run failed: boom",
    })

    row = {"id": "Q02", "question": "q", "gold_source_file": "x", "gold_supporting_snippet": "y"}
    result = harness_module._run_llm_mode(row)

    assert result["notes"] == "Baseline run failed: boom"
    assert result["grounded_precision"] == 0.0
    assert result["accuracy"] == 0.0


def test_run_react_mode_writes_mode_react_and_step_count(monkeypatch):
    import eval.harness as harness_module

    fake_trace = [
        {"step": 1, "action": "search", "action_input": {"query": "q"}, "observation": json.dumps([{"source": "doc1.md"}])},
        {"step": 2, "action": "read_chunk", "action_input": {"chunk_id": "doc1_001"}, "observation": "chunk text"},
    ]
    monkeypatch.setattr(harness_module, "run_agent", lambda q, agent=None: {
        "trace": fake_trace, "answer": 'Answer. [doc1.md] "chunk text"', "error": None,
    })
    monkeypatch.setattr(harness_module, "compute_grounded_precision", lambda a: 0.9)
    monkeypatch.setattr(harness_module, "compute_accuracy", lambda q, a, g: 0.95)

    row = {"id": "Q03", "question": "What is X?", "gold_source_file": "doc1.md", "gold_supporting_snippet": "X is..."}
    result = harness_module._run_react_mode(row, agent=object())

    assert result["mode"] == "react"
    assert result["steps_used"] == 2
    assert result["retrieved_files"] == "doc1.md"
    assert result["grounded_precision"] == 0.9
    assert result["accuracy"] == 0.95


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))