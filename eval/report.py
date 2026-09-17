"""
eval/report.py
-----------------
Aggregates the logged runs (runs.csv, written by eval/harness.py) into
the short evaluation report the spec asks for: grounded precision and
accuracy, "llm" (baseline) vs "react" (ReAct+tools), side by side.
"""

import csv
import statistics
from collections import defaultdict

import config


def load_runs(path=None):
    path = path or config.EVAL_RUNS_OUTPUT_PATH
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(runs):
    """Returns {mode: {"n": count, "grounded_precision": mean, "accuracy": mean}}"""
    by_mode = defaultdict(list)
    for row in runs:
        by_mode[row["mode"]].append(row)

    summary = {}
    for mode, rows in by_mode.items():
        gp = [float(r["grounded_precision"]) for r in rows if r["grounded_precision"] not in ("", None)]
        acc = [float(r["accuracy"]) for r in rows if r["accuracy"] not in ("", None)]
        summary[mode] = {
            "n": len(rows),
            "grounded_precision": statistics.mean(gp) if gp else 0.0,
            "accuracy": statistics.mean(acc) if acc else 0.0,
        }
    return summary


def format_report(summary, runs, n_examples=3):
    lines = ["# Evaluation Report: LLM-only Baseline vs ReAct + Tools", ""]
    lines.append("| Mode | N | Grounded Precision | Accuracy |")
    lines.append("|------|---|---------------------|----------|")
    for mode in sorted(summary.keys()):
        s = summary[mode]
        lines.append(f"| {mode} | {s['n']} | {s['grounded_precision']:.2f} | {s['accuracy']:.2f} |")

    lines.append("")
    lines.append("## Example Rows")
    lines.append("")
    for row in runs[: n_examples * 2]:
        lines.append(f"**{row['question_id']}** ({row['mode']}): {row['question']}")
        lines.append(f"- Answer: {row['answer'][:300]}")
        lines.append(f"- Citations: {row['citations'] or '(none)'}")
        lines.append(f"- Grounded precision: {row['grounded_precision']}, Accuracy: {row['accuracy']}")
        lines.append("")

    return "\n".join(lines)


def generate_report(runs_path=None, report_path=None):
    runs_path = runs_path or config.EVAL_RUNS_OUTPUT_PATH
    report_path = report_path or config.EVAL_REPORT_OUTPUT_PATH

    runs = load_runs(runs_path)
    summary = summarize(runs)
    report_text = format_report(summary, runs)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text