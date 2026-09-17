"""
eval/harness.py
------------------
Runs the fixed 20-question evaluation set (evaluation_questions.csv)
through both modes -- "llm" (no-tools baseline) and "react" (ReAct +
tools) -- and logs one row per (question, mode) into runs.csv, matching
runs_template.csv's columns plus an added "accuracy" column (the
template didn't have one, but the spec requires it in the report).
"""

import csv
import json
import logging

import config
from agent.guardrails import extract_citations
from agent.react_agent import build_agent, run_agent
from eval.accuracy import compute_accuracy
from eval.baseline import run_baseline
from eval.grounded_precision import compute_grounded_precision

logger = logging.getLogger(__name__)

RUN_COLUMNS = [
    "question_id", "question", "mode", "steps_used", "retrieved_files",
    "answer", "citations", "grounded_precision", "accuracy", "notes",
]


def load_questions(path=None):
    path = path or config.EVAL_QUESTIONS_PATH
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _format_citations(answer_text):
    citations = extract_citations(answer_text)
    return "; ".join(f'[{s}{":" + p if p else ""}] "{q}"' for s, p, q in citations)


def _retrieved_files_from_trace(trace):
    """Pulls the distinct source filenames out of every search() step's
    observation in the trace (search results are JSON-serialized lists
    of dicts with a "source" field)."""
    files = []
    for step in trace:
        if step["action"] != "search":
            continue
        try:
            results = json.loads(step["observation"])
        except (TypeError, ValueError):
            continue
        for result in results:
            source = result.get("source")
            if source and source not in files:
                files.append(source)
    return "; ".join(files)


def _run_llm_mode(row):
    result = run_baseline(row["question"])
    if result["error"]:
        return {
            "question_id": row["id"], "question": row["question"], "mode": "llm",
            "steps_used": 0, "retrieved_files": "", "answer": "", "citations": "",
            "grounded_precision": 0.0, "accuracy": 0.0, "notes": result["error"],
        }

    answer = result["answer"]
    return {
        "question_id": row["id"], "question": row["question"], "mode": "llm",
        "steps_used": 0, "retrieved_files": "", "answer": answer,
        "citations": _format_citations(answer),
        "grounded_precision": compute_grounded_precision(answer),
        "accuracy": compute_accuracy(row["question"], answer, row["gold_supporting_snippet"]),
        "notes": "",
    }


def _run_react_mode(row, agent):
    result = run_agent(row["question"], agent=agent)
    if result["error"]:
        return {
            "question_id": row["id"], "question": row["question"], "mode": "react",
            "steps_used": len(result["trace"]), "retrieved_files": _retrieved_files_from_trace(result["trace"]),
            "answer": "", "citations": "", "grounded_precision": 0.0, "accuracy": 0.0,
            "notes": result["error"],
        }

    answer = result["answer"]
    return {
        "question_id": row["id"], "question": row["question"], "mode": "react",
        "steps_used": len(result["trace"]), "retrieved_files": _retrieved_files_from_trace(result["trace"]),
        "answer": answer, "citations": _format_citations(answer),
        "grounded_precision": compute_grounded_precision(answer),
        "accuracy": compute_accuracy(row["question"], answer, row["gold_supporting_snippet"]),
        "notes": "",
    }


def run_evaluation(questions_path=None, output_path=None):
    """Runs every question through both modes and writes runs.csv.
    Returns the list of row dicts written."""
    questions = load_questions(questions_path)
    output_path = output_path or config.EVAL_RUNS_OUTPUT_PATH

    agent = build_agent()  # built once, reused across every react-mode run
    rows = []

    for row in questions:
        logger.info("Evaluating %s: %s", row["id"], row["question"])
        rows.append(_run_llm_mode(row))
        rows.append(_run_react_mode(row, agent))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d run(s) to %s", len(rows), output_path)
    return rows


if __name__ == "__main__":
    config.setup_logging()
    run_evaluation()