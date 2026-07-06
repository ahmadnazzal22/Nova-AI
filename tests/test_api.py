import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from part2_rag import rag_agent, api as rag_api


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    ckpt = "transformer_checkpoint.pth"
    if not os.path.exists(ckpt):
        pytest.skip("No checkpoint found — run training first")
    rag_agent.RAGAgent._instance = None
    rag_api._agent = None
    monkeypatch.setattr(rag_agent.RAGAgent, "_get_llm", lambda self: rag_agent.MockLLM())
    yield


@pytest.fixture
def client():
    return TestClient(rag_api.app)


class TestAPI:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_query_endpoint(self, client):
        resp = client.post("/query", json={"question": "What is a transformer?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_empty_question_returns_422(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_missing_field_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_ingest_endpoint(self, client):
        content = b"word " * 500 + b"Transformer uses self-attention."
        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", content, "text/plain")},
            data={"chunk_size": 256, "chunk_overlap": 32},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["chunks"] >= 1
        assert data["total_docs"] >= data["chunks"]

    def test_ingest_unsupported_extension(self, client):
        resp = client.post(
            "/ingest",
            files={"file": ("test.docx", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_docs_swagger(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_docs_redoc(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def _register_and_login(self, client, username="memoryuser", password="MemPass1!"):
        resp = client.post("/auth/register", json={"username": username, "email": f"{username}@test.com", "password": password})
        assert resp.status_code == 200, f"Register failed: {resp.status_code} {resp.text}"
        return resp.json()["access_token"]

    def test_memory_store_and_list(self, client):
        token = self._register_and_login(client, "memuser1")
        headers = {"Authorization": f"Bearer {token}"}
        resp = client.post("/memories", json={"key": "preference", "value": "likes AI", "importance": 0.8}, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "preference"
        assert data["value"] == "likes AI"
        resp2 = client.get("/memories", headers=headers)
        assert resp2.status_code == 200
        mems = resp2.json()["memories"]
        assert any(m["key"] == "preference" and m["value"] == "likes AI" for m in mems)

    def test_memory_search(self, client):
        token = self._register_and_login(client, "memuser2")
        headers = {"Authorization": f"Bearer {token}"}
        client.post("/memories", json={"key": "goal", "value": "master transformers", "importance": 0.9}, headers=headers)
        resp = client.get("/memories/search?q=transformers", headers=headers)
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert any(r["key"] == "goal" for r in results)

    def test_memory_delete(self, client):
        token = self._register_and_login(client, "memuser3")
        headers = {"Authorization": f"Bearer {token}"}
        client.post("/memories", json={"key": "temp", "value": "temporary", "importance": 0.1}, headers=headers)
        resp = client.delete("/memories/temp", headers=headers)
        assert resp.status_code == 200
        resp2 = client.get("/memories", headers=headers)
        assert all(m["key"] != "temp" for m in resp2.json()["memories"])

    def test_memory_delete_not_found(self, client):
        token = self._register_and_login(client, "memuser4")
        headers = {"Authorization": f"Bearer {token}"}
        resp = client.delete("/memories/nonexistent_key", headers=headers)
        assert resp.status_code == 404

    def test_memory_requires_auth(self, client):
        resp = client.get("/memories")
        assert resp.status_code == 401

    def test_memory_user_isolation(self, client):
        token_a = self._register_and_login(client, "memiso_a", password="Pass1234!")
        token_b = self._register_and_login(client, "memiso_b", password="Pass1234!")
        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}
        client.post("/memories", json={"key": "secret", "value": "user_a_secret", "importance": 0.9}, headers=headers_a)
        resp_b = client.get("/memories", headers=headers_b)
        assert all(m["key"] != "secret" for m in resp_b.json()["memories"])
