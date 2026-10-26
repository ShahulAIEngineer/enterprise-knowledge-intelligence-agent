from fastapi.testclient import TestClient

from knowledge_agent.main import app
from knowledge_agent.service import get_agent, get_index


class FakeAgent:
    def ask(self, question):
        return {"answer": f"Test answer for: {question}", "citations": [], "retrieved_chunks": 0}


class FakeIndex:
    def ingest_path(self, path):
        return 2


def setup_function():
    app.dependency_overrides[get_agent] = lambda: FakeAgent()
    app.dependency_overrides[get_index] = lambda: FakeIndex()


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask():
    response = TestClient(app).post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert "Test answer" in response.json()["answer"]


def test_ask_rejects_short_question():
    response = TestClient(app).post("/ask", json={"question": "?"})
    assert response.status_code == 422


def test_ingest_rejects_unsupported_file():
    response = TestClient(app).post(
        "/ingest", files={"file": ("payload.exe", b"not allowed", "application/octet-stream")}
    )
    assert response.status_code == 415
