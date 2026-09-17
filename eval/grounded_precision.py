"""
eval/grounded_precision.py
-----------------------------
Computes the evaluation report's "grounded precision" metric: for a
given answer, what fraction of its citations actually support the claim
they're attached to (per the project spec's own definition -- NOT
whether the quote is verbatim traceable to a retrieved chunk, which is
the separate hallucination guardrail already in agent/guardrails.py).

Design (cost-optimized, agreed on after discussion):
  - zero citations in the answer -> grounded_precision = 0.0, NO LLM
    call at all (nothing to judge, nothing to pay for -- covers most of
    the "llm" baseline rows for free)
  - otherwise -> ONE LLM call per answer, given the full answer text and
    its extracted citations, asked to internally judge each citation
    against the claim it's attached to and return ONLY a single decimal
    in [0, 1] -- the fraction of citations that are supported. This
    keeps the score anchored to an explicit definition (not a vague
    "rate this 0-1" impression) while paying for one call and a handful
    of output tokens, not one call per citation.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.guardrails import call_with_retry, extract_citations
from agent.llm import get_llm

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are grading citations in an AI-generated answer for a \
retrieval evaluation. You will be given the answer text and its citations (each a \
source, an optional page, and a quoted piece of evidence).

For each citation, judge internally whether the quoted evidence actually supports \
the specific claim it's attached to in the answer -- not just whether it's on-topic, \
but whether it genuinely backs up what's being claimed.

Then compute: (number of citations whose evidence supports its claim) / (total number \
of citations).

Output ONLY that number as a decimal between 0 and 1 (e.g. "0.67"). No words, no \
explanation, no reasoning -- just the number."""

_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def _format_citations_for_judge(citations):
    lines = []
    for i, (source, page, quote) in enumerate(citations, start=1):
        label = f"{source}:{page}" if page else source
        lines.append(f'{i}. [{label}] "{quote}"')
    return "\n".join(lines)


def _parse_score(text):
    """Extracts the first number in the judge's response and clamps it
    into [0, 1]. Raises ValueError if no number is found, so the caller
    (call_with_retry) can retry."""
    match = _NUMBER_PATTERN.search(text)
    if not match:
        raise ValueError(f"No numeric score found in judge response: {text!r}")
    score = float(match.group())
    return max(0.0, min(1.0, score))


def compute_grounded_precision(answer_text, llm=None):
    """Returns a float in [0, 1]. See module docstring for the rule."""
    citations = extract_citations(answer_text)
    if not citations:
        logger.debug("No citations found -- grounded_precision=0.0, no LLM call")
        return 0.0

    llm = llm or get_llm()
    citations_block = _format_citations_for_judge(citations)
    user_message = f"ANSWER:\n{answer_text}\n\nCITATIONS:\n{citations_block}"

    def _invoke_and_parse():
        response = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        return _parse_score(response.content)

    score = call_with_retry(_invoke_and_parse)
    logger.info("compute_grounded_precision: %d citation(s) -> score=%.2f", len(citations), score)
    return score