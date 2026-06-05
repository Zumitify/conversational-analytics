"""FastAPI service tests over the real pipeline (mock LLM)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cae.api import create_app


@pytest.fixture()
def client(pipeline):
    pipeline.provider.program_intent(
        "revenue by region this year",
        {"metrics": ["revenue"], "dimensions": ["region"],
         "time_range": {"relative": "ytd"}},
    )
    pipeline.provider.program_intent(
        "show me the blorp", {"metrics": ["blorp"]},
    )
    app = create_app(pipeline=pipeline)
    return TestClient(app, raise_server_exceptions=False)


class TestEndpoints:
    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "llm_cost" in body

    def test_schema(self, client):
        body = client.get("/schema").json()
        metric_names = [m["name"] for m in body["metrics"]]
        assert "revenue" in metric_names
        dim_names = [d["name"] for d in body["dimensions"]]
        assert "region" in dim_names
        assert "customer_email" not in dim_names  # PII excluded

    def test_full_ask_flow(self, client):
        session_id = client.post("/sessions").json()["session_id"]

        response = client.post(
            f"/sessions/{session_id}/ask",
            json={"question": "revenue by region this year"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["intent"]["metrics"] == ["revenue"]
        assert body["result"]["row_count"] == 4
        assert "SELECT" in body["sql"]
        assert body["chart_spec"]["chart_type"] == "bar"

        history = client.get(f"/sessions/{session_id}/history").json()
        assert len(history["turns"]) == 1
        assert history["turns"][0]["question"] == "revenue by region this year"

        sql_view = client.get(f"/sessions/{session_id}/turns/0/sql").json()
        assert "SELECT" in sql_view["sql"]
        assert sql_view["intent"]["metrics"] == ["revenue"]

    def test_ask_intent_endpoint(self, client):
        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/ask_intent",
            json={"intent": {"metrics": ["revenue"], "dimensions": ["channel"],
                             "time_range": {"relative": "last_month"}}},
        )
        assert response.status_code == 200
        assert response.json()["result"]["row_count"] == 3

    def test_clarification_returns_422(self, client):
        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/ask",
            json={"question": "show me the blorp"},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "clarification_needed"
        assert "suggestions" in body

    def test_unknown_session_404(self, client):
        response = client.post("/sessions/ghost/ask", json={"question": "hi"})
        assert response.status_code == 404

    def test_unknown_turn_404(self, client):
        session_id = client.post("/sessions").json()["session_id"]
        assert client.get(f"/sessions/{session_id}/turns/9/sql").status_code == 404
