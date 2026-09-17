"""
react_copilot.py
------------------
CLI entrypoint for the ReAct Research Copilot.

Usage:
    python react_copilot.py --question "What does 'faithfulness' mean in LLM evaluation?"

Prints the step-by-step Action/Observation trace, then the final answer
with citations and minimal supporting quotes, then a citation-validity
check.
"""

import argparse
import sys

import config
from agent.react_agent import build_agent, run_agent


def format_trace(trace):
    lines = []
    for step in trace:
        lines.append(f"\nStep {step['step']}:")
        lines.append(f"  Action: {step['action']}({step['action_input']})")
        observation = str(step["observation"])
        if len(observation) > 500:
            observation = observation[:500] + " ...[truncated]"
        lines.append(f"  Observation: {observation}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ReAct Research Copilot")
    parser.add_argument("--question", required=True, help="Question to ask the corpus")
    args = parser.parse_args()

    config.setup_logging()

    agent = build_agent()
    result = run_agent(args.question, agent=agent)

    print("=" * 70)
    print(f"QUESTION: {result['question']}")
    print("=" * 70)

    if result["error"]:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print("\n--- STEP-BY-STEP TRACE ---")
    print(format_trace(result["trace"]) if result["trace"] else "(no tool calls made)")

    print("\n--- FINAL ANSWER ---")
    print(result["answer"])

    print("\n--- CITATION CHECK ---")
    if result["citations_valid"]:
        print("All citations verified against retrieved source text.")
    else:
        print("Citation issues detected:")
        for problem in result["citation_problems"]:
            print(f"  - {problem}")

    print(f"\n(elapsed: {result['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()