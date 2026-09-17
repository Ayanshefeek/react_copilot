"""
eval/baseline.py
------------------
The "LLM-only, no tools" baseline used in the evaluation report: the
question goes straight to the LLM with no retrieval and no tool access
at all -- just whatever's in the model's own training data. Deliberately
given no context, so its citation behavior (usually missing or
fabricated) is the direct point of comparison against ReAct+tools'
grounded precision.

Uses the same citation-format convention as the ReAct agent's system
prompt (agent/prompts.py), so any citations it does produce are
parseable by the same extract_citations() regex -- letting the eval
harness treat both modes identically downstream.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm

logger = logging.getLogger(__name__)

BASELINE_SYSTEM_PROMPT = """You are a research assistant answering a question from your own \
knowledge only -- you have no access to any document corpus, search tool, or external \
source. Answer as best you can from what you already know.

If you can name a source for a claim, cite it in this format immediately after the \
claim, with a short supporting quote: [filename] "quote" or [filename:page] "quote". \
Only do this if you are confident about the exact source and exact wording -- never \
invent a plausible-sounding filename or quote. If you have no real source to cite, \
just answer without a citation rather than fabricating one.

Be concise."""


def run_baseline(question, llm=None):
    """Runs one question through the no-tools baseline. Returns:
        {"question": str, "answer": str, "error": str or None}
    Shaped like a simplified version of run_agent()'s result, so the
    eval harness can handle both modes with similar code."""
    llm = llm or get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=BASELINE_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
        answer = response.content
        logger.info("run_baseline(%r) -> %d chars", question, len(answer))
        return {"question": question, "answer": answer, "error": None}
    except Exception as exc:
        logger.exception("Baseline run failed")
        return {"question": question, "answer": "", "error": f"Baseline run failed: {exc}"}