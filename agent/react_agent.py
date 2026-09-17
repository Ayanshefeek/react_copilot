"""
agent/react_agent.py
----------------------
Builds and runs the ReAct agent: LangChain's create_agent (the current
agents API -- built on LangGraph, superseding the older
langgraph.prebuilt.create_react_agent) wired to the search/read_chunk
tools, with guardrails (max steps via recursion limit, a wall-clock
timeout, tool retries baked into the tools themselves) and a
step-by-step trace extracted from the run for the CLI to print.

create_agent IS the ReAct loop -- reason, call a tool, observe, repeat
until a final answer with no more tool calls -- backed by whatever LLM
agent/llm.py returns. No hand-rolled loop needed.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import config
from agent.guardrails import validate_citations
from agent.llm import get_llm
from agent.prompts import SYSTEM_PROMPT
from agent.tools import build_tools
from retrieval.tools import RetrievalTools

logger = logging.getLogger(__name__)


def build_agent(retrieval_tools=None, llm=None):
    """retrieval_tools/llm can be injected (tests use stubs); defaults to
    the real RetrievalTools + get_llm() for normal use."""
    retrieval_tools = retrieval_tools or RetrievalTools()
    llm = llm or get_llm()

    tools = build_tools(retrieval_tools)
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    logger.info("ReAct agent built with %d tool(s)", len(tools))
    return agent


def _extract_trace(messages):
    """Turns the agent's message history into a simple list of
    {step, action, action_input, observation} dicts for printing, plus
    the list of chunk texts observed via read_chunk (used afterwards to
    validate the final answer's quotes)."""
    trace = []
    observed_chunk_texts = []
    step = 0
    pending_calls = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                pending_calls[call["id"]] = {
                    "action": call["name"],
                    "action_input": call["args"],
                }
        elif isinstance(msg, ToolMessage):
            step += 1
            call = pending_calls.pop(msg.tool_call_id, {"action": "unknown", "action_input": {}})
            observation = msg.content
            trace.append({
                "step": step,
                "action": call["action"],
                "action_input": call["action_input"],
                "observation": observation,
            })
            if call["action"] == "read_chunk" and isinstance(observation, str):
                observed_chunk_texts.append(observation)

    return trace, observed_chunk_texts


def run_agent(question, agent=None, max_steps=None, timeout_seconds=None):
    """Runs one question through the ReAct agent under the configured
    guardrails (max steps, timeout). Returns a dict:
        {
          "question": str,
          "trace": [ {step, action, action_input, observation}, ... ],
          "answer": str,
          "citations_valid": bool,
          "citation_problems": [str, ...],
          "elapsed_seconds": float,
          "error": str or None,   # set on timeout / max-steps / failure
        }
    """
    agent = agent or build_agent()
    max_steps = max_steps or config.MAX_AGENT_STEPS
    timeout_seconds = timeout_seconds or config.AGENT_TIMEOUT_SECONDS

    start = time.time()
    result = {
        "question": question,
        "trace": [],
        "answer": "",
        "citations_valid": False,
        "citation_problems": [],
        "elapsed_seconds": 0.0,
        "error": None,
    }

    # Each ReAct step is roughly a model-node + tool-node pair in the
    # underlying graph, plus one final model-node call for the answer --
    # so allow ~2 graph transitions per step, with headroom.
    recursion_limit = max_steps * 2 + 2

    def _invoke():
        return agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": recursion_limit},
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            output = future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        result["error"] = f"Agent timed out after {timeout_seconds}s (config.AGENT_TIMEOUT_SECONDS)"
        result["elapsed_seconds"] = time.time() - start
        logger.error(result["error"])
        return result
    except GraphRecursionError:
        result["error"] = f"Agent exceeded MAX_AGENT_STEPS={max_steps} without producing a final answer"
        result["elapsed_seconds"] = time.time() - start
        logger.error(result["error"])
        return result
    except Exception as exc:
        result["error"] = f"Agent run failed: {exc}"
        result["elapsed_seconds"] = time.time() - start
        logger.exception("Agent run failed")
        return result

    messages = output["messages"]
    trace, observed_chunk_texts = _extract_trace(messages)

    final_message = messages[-1]
    answer = final_message.content if isinstance(final_message, AIMessage) else str(final_message.content)

    is_valid, problems = validate_citations(answer, observed_chunk_texts)

    result.update({
        "trace": trace,
        "answer": answer,
        "citations_valid": is_valid,
        "citation_problems": problems,
        "elapsed_seconds": time.time() - start,
    })

    logger.info(
        "run_agent(%r) -> %d step(s), citations_valid=%s, %.1fs",
        question, len(trace), is_valid, result["elapsed_seconds"],
    )
    return result