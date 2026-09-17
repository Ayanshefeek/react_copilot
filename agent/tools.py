"""
agent/tools.py
----------------
Wraps RetrievalTools.search/read_chunk as LangChain tools the ReAct
agent can call. Docstrings here are written for the LLM, not a
developer -- this is what the agent reads to decide which tool to call,
with what arguments, and what the result means.

Only config.ALLOWED_TOOLS are ever built -- an explicit safe allow-list,
so no other function is reachable from the agent loop even if the LLM
tries to invent a tool call.
"""

import logging

from langchain_core.tools import tool

import config
from agent.guardrails import call_with_retry

logger = logging.getLogger(__name__)


def build_tools(retrieval_tools):
    """retrieval_tools: a loaded RetrievalTools instance. Returns the
    list of LangChain tools bound to it, filtered to config.ALLOWED_TOOLS."""

    @tool
    def search(query: str) -> list:
        """Search the document corpus for chunks relevant to a query.

        Call this first, before answering, to find candidate source
        material. Call it again with a reformulated query if the first
        results don't look relevant enough to answer confidently.

        Args:
            query: A natural-language question or search phrase.

        Returns:
            A list of up to 5 results, each a dict with:
              - chunk_id: opaque id -- pass this to read_chunk to get the
                chunk's full text
              - source: the file name the chunk came from
              - section: markdown section heading, or null for PDFs
              - page: PDF page number, or null for markdown
              - score: relevance score, higher is more relevant
            Does NOT include the chunk's text -- call read_chunk for that.
        """
        results = call_with_retry(retrieval_tools.search, query)
        logger.info("[tool] search(%r) -> %d results", query, len(results))
        return results

    @tool
    def read_chunk(chunk_id: str) -> str:
        """Fetch the full text of one chunk found via search().

        Always call this before citing a source or quoting from it --
        search() only gives you metadata, never the chunk's actual
        content. Use the exact chunk_id string returned by search().

        Args:
            chunk_id: the chunk_id field from a search() result.

        Returns:
            The chunk's full text as a plain string.
        """
        text = call_with_retry(retrieval_tools.read_chunk, chunk_id)
        logger.info("[tool] read_chunk(%r) -> %d chars", chunk_id, len(text))
        return text

    all_tools = {"search": search, "read_chunk": read_chunk}
    allowed = [all_tools[name] for name in config.ALLOWED_TOOLS if name in all_tools]
    logger.info("Built %d tool(s) from allow-list: %s", len(allowed), config.ALLOWED_TOOLS)
    return allowed