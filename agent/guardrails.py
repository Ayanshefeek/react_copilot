"""
agent/guardrails.py
---------------------
Guardrail helpers for the ReAct agent loop:

  - a small in-tool retry wrapper for transient tool failures (note:
    LangChain's Runnable.with_retry() looks like the obvious choice here,
    but it wraps the tool in a RunnableRetry object that create_agent's
    ToolNode can't register as a tool -- so retry logic lives inside the
    tool function body instead, see agent/tools.py)
  - a substring check that catches a hallucinated "minimal quote" or an
    over-long one before it reaches the final answer

Max steps and timeout are guardrails on the whole run, not a single tool
call, so those live in agent/react_agent.py where the agent is invoked.
"""

import logging
import re
import time

import config

logger = logging.getLogger(__name__)

# Matches: [source] "quote"  or  [source:page] "quote"
CITATION_QUOTE_PATTERN = re.compile(r'\[([^\]:]+)(?::(\d+))?\]\s*"([^"]+)"')


def call_with_retry(fn, *args, **kwargs):
    """Calls fn(*args, **kwargs), retrying up to config.TOOL_RETRY_ATTEMPTS
    times (with a fixed config.TOOL_RETRY_BACKOFF_SECONDS delay between
    attempts) if it raises. Re-raises the last exception if every attempt
    fails -- this is for transient failures, not for masking real bugs."""
    attempts = config.TOOL_RETRY_ATTEMPTS
    backoff = config.TOOL_RETRY_BACKOFF_SECONDS
    last_exc = None

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning("Tool call failed (attempt %d/%d): %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(backoff)

    raise last_exc


def extract_citations(answer_text):
    """Returns a list of (source, page_or_empty_string, quote) tuples
    found in the final answer, in the [source]/[source:page] "quote"
    format the system prompt requires."""
    return [(source, page, quote) for source, page, quote in CITATION_QUOTE_PATTERN.findall(answer_text)]


def validate_citations(answer_text, observed_chunk_texts):
    """Checks every citation's quote against the chunk texts the agent
    actually retrieved via read_chunk during this run. Returns
    (is_valid, problems) -- problems is a list of human-readable strings,
    empty if everything checks out.

    Per citation:
      - the quote must not be empty
      - the quote must not exceed config.MAX_QUOTE_CHARS (catches
        "quoting the whole chunk" instead of a minimal excerpt)
      - the quote must appear verbatim in at least one chunk the agent
        actually read (catches a hallucinated quote)
    """
    citations = extract_citations(answer_text)
    problems = []

    if not citations:
        problems.append("Answer contains no citations in the required [source] \"quote\" format.")
        return False, problems

    for source, page, quote in citations:
        label = f"[{source}:{page}]" if page else f"[{source}]"

        if not quote.strip():
            problems.append(f"Empty quote for citation {label}")
            continue

        if len(quote) > config.MAX_QUOTE_CHARS:
            problems.append(
                f"Quote for {label} is {len(quote)} chars, exceeds "
                f"MAX_QUOTE_CHARS={config.MAX_QUOTE_CHARS} -- not minimal"
            )

        if not any(quote in text for text in observed_chunk_texts):
            problems.append(f"Quote for {label} not found verbatim in any retrieved chunk (possible hallucination)")

    return len(problems) == 0, problems