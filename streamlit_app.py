"""
streamlit_app.py
-------------------
Streamlit UI for the ReAct Research Copilot. Talks to the FastAPI
backend over HTTP (api/main.py) -- it does not import the agent
directly, so this stays a genuine separate client of the API, same as
any other consumer would be.

Run (with the API already running separately):
    uvicorn api.main:app --reload --port 8000
    streamlit run streamlit_app.py
"""

import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

import config
from eval.report import load_runs, summarize

BACKEND_URL = config.STREAMLIT_BACKEND_URL

st.set_page_config(page_title="ReAct Research Copilot", layout="wide")
st.title("ReAct Research Copilot")

ask_tab, eval_tab = st.tabs(["Ask", "Evaluation Report"])


# ---------------------------------------------------------------------------
# Tab 1: live Q&A against the FastAPI backend
# ---------------------------------------------------------------------------
with ask_tab:
    st.caption(f"Backend: {BACKEND_URL}")

    question = st.text_area("Ask a question about the corpus", height=100)
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Running the ReAct agent..."):
                    response = requests.post(
                        f"{BACKEND_URL}/ask",
                        json={"question": question},
                        timeout=config.AGENT_TIMEOUT_SECONDS + 10,
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the backend at {BACKEND_URL}. "
                    "Make sure it's running: `uvicorn api.main:app --reload --port 8000`"
                )
            except requests.exceptions.Timeout:
                st.error("The request timed out waiting for the backend.")
            else:
                if response.status_code == 200:
                    result = response.json()

                    st.subheader("Step-by-step trace")
                    if result["trace"]:
                        for step in result["trace"]:
                            with st.expander(f"Step {step['step']}: {step['action']}({step['action_input']})"):
                                observation = step["observation"]
                                if len(observation) > 800:
                                    observation = observation[:800] + " ...[truncated]"
                                st.code(observation, language=None)
                    else:
                        st.caption("(no tool calls made)")

                    st.subheader("Final answer")
                    st.markdown(result["answer"])

                    st.subheader("Citation check")
                    if result["citations_valid"]:
                        st.success("All citations verified against retrieved source text.")
                    else:
                        st.warning("Citation issues detected:")
                        for problem in result["citation_problems"]:
                            st.write(f"- {problem}")

                    st.caption(f"Elapsed: {result['elapsed_seconds']:.1f}s")
                elif response.status_code == 422:
                    st.error(f"Invalid question: {response.json()['detail']}")
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"Backend error ({response.status_code}): {detail}")


# ---------------------------------------------------------------------------
# Tab 2: read-only view of the last `python run_eval.py` results
# ---------------------------------------------------------------------------
with eval_tab:
    runs_path = config.EVAL_RUNS_OUTPUT_PATH

    if not os.path.exists(runs_path):
        st.info(
            f"No evaluation run found yet at `{runs_path}`. "
            "Run `python run_eval.py` first, then reload this page."
        )
    else:
        last_modified = datetime.fromtimestamp(os.path.getmtime(runs_path))
        st.caption(f"Showing results from the last `python run_eval.py` run -- {last_modified:%Y-%m-%d %H:%M:%S}")

        runs = load_runs(runs_path)
        summary = summarize(runs)

        summary_df = pd.DataFrame(
            [
                {
                    "mode": mode,
                    "n": s["n"],
                    "grounded_precision": round(s["grounded_precision"], 3),
                    "accuracy": round(s["accuracy"], 3),
                }
                for mode, s in sorted(summary.items())
            ]
        ).set_index("mode")

        st.subheader("LLM-only baseline vs ReAct + tools")
        st.dataframe(summary_df, use_container_width=True)
        st.subheader("Grounded Precision: LLM-only vs ReAct")
        st.bar_chart(summary_df[["grounded_precision"]])

        st.subheader("Accuracy: LLM-only vs ReAct")
        st.bar_chart(summary_df[["accuracy"]])

        st.subheader("All logged runs")
        st.dataframe(pd.DataFrame(runs), use_container_width=True)

        report_path = config.EVAL_REPORT_OUTPUT_PATH
        if os.path.exists(report_path):
            with st.expander("Raw report (evaluation_report.md)"):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())