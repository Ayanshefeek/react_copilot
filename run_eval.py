"""
run_eval.py
-------------
CLI entrypoint for the evaluation phase: runs all 20 questions through
both modes (llm baseline, react+tools), logs runs.csv, and writes the
short evaluation report.

Usage:
    python run_eval.py
"""

import config
from eval.harness import run_evaluation
from eval.report import generate_report


def main():
    config.setup_logging()
    print("Running evaluation (20 questions x 2 modes)...")
    run_evaluation()
    print(f"Runs logged to {config.EVAL_RUNS_OUTPUT_PATH}")

    report_text = generate_report()
    print(f"\nReport written to {config.EVAL_REPORT_OUTPUT_PATH}\n")
    print(report_text)


if __name__ == "__main__":
    main()