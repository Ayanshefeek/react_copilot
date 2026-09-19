"""
tests/test_api.py
--------------------
FastAPI layer tests. The real agent (LLM + embeddings + index) is never
built here -- api.main.build_agent and api.main.run_agent are monkeypatched
before the TestClient starts the app, same pattern used in
tests/test_agent.py and tests/test_eval.py for stubbing out the LLM.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "build_agent", lambda: "STUB_AGENT")

    def fake_run_agent(question, agent=None, max_steps=None, timeout_seconds=None):
        if question == "trigger-error":
            return {
                "question": question,
                "trace": [],
                "answer": "",
                "citations_valid": False,
                "citation_problems": [],
                "elapsed_seconds": 0.1,
                "error": "Agent run failed: boom",
            }
        return {
            "question": question,
            "trace": [
                {
                    "step": 1,
                    "action": "search",
                    "action_input": {"query": question},
                    "observation": '[{"chunk_id": "c1", "source": "doc.md"}]',
                },
            ],
            "answer": '[doc.md] "answer text"',
            "citations_valid": True,
            "citation_problems": [],
            "elapsed_seconds": 1.23,
            "error": None,
        }

    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_reports_agent_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["agent_ready"] is True


def test_config_reflects_current_settings(client):
    import config

    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert body["llm_provider"] == config.LLM_PROVIDER
    assert body["embedding_provider"] == config.EMBEDDING_PROVIDER


def test_ask_returns_answer_for_a_normal_question(client):
    response = client.post("/ask", json={"question": "What is faithfulness?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations_valid"] is True
    assert body["answer"] == '[doc.md] "answer text"'
    assert len(body["trace"]) == 1
    assert body["trace"][0]["action"] == "search"


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_question_over_the_length_limit(client):
    import config

    too_long = "a" * (config.MAX_QUESTION_CHARS + 1)
    response = client.post("/ask", json={"question": too_long})
    assert response.status_code == 422


def test_ask_maps_a_genuine_agent_error_to_http_500(client):
    response = client.post("/ask", json={"question": "trigger-error"})
    assert response.status_code == 500
    assert "boom" in response.json()["detail"]