import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from part2_rag import api as rag_api
from part2_rag.database import get_session, UserRepository
from part2_rag.auth import hash_password


@pytest.fixture
def client():
    return TestClient(rag_api.app)


@pytest.fixture
def registered_user(client):
    resp = client.post("/auth/register", json={
        "username": "integration_user",
        "email": "int@test.com",
        "password": "testpass123",
    })
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def admin_user(registered_user):
    session = get_session()
    try:
        repo = UserRepository(session)
        admin = repo.create("admin", "admin@test.com", hash_password("adminpass"), role="admin")
        session.commit()
    finally:
        session.close()
    from fastapi.testclient import TestClient
    c = TestClient(rag_api.app)
    resp = c.post("/auth/login", json={"username": "admin", "password": "adminpass"})
    assert resp.status_code == 200
    return resp.json()


class TestIntegrationFlow:
    def test_full_auth_flow(self, client):
        r = client.post("/auth/register", json={
            "username": "flow_user", "email": "flow@test.com", "password": "pass123456"
        })
        assert r.status_code == 200
        tokens = r.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        r = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert r.status_code == 200
        assert r.json()["username"] == "flow_user"

        r = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert r.status_code == 200
        new_tokens = r.json()
        assert "access_token" in new_tokens
        assert "refresh_token" in new_tokens

    def test_chat_history_flow(self, client, registered_user):
        token = registered_user["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        r = client.get("/chat/history", headers=headers)
        assert r.status_code == 200
        assert r.json() == []

        r = client.post("/live", json={"question": "What is Python?"}, headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["conversation_id"] > 0

        conv_id = data["conversation_id"]

        r = client.get("/chat/history", headers=headers)
        assert r.status_code == 200
        convs = r.json()
        assert len(convs) >= 1
        assert any(c["id"] == conv_id for c in convs)

        r = client.get(f"/chat/{conv_id}", headers=headers)
        assert r.status_code == 200
        conv_data = r.json()
        assert len(conv_data["messages"]) >= 2
        assert conv_data["messages"][0]["role"] == "user"
        assert conv_data["messages"][1]["role"] == "assistant"

        msg_id = conv_data["messages"][1]["id"]
        assert msg_id > 0

        r = client.put(f"/chat/{conv_id}/rename", json={"title": "Python Chat"},
                       headers=headers)
        assert r.status_code == 200
        assert r.json()["status"] == "renamed"

        r = client.post("/feedback", json={"message_id": msg_id, "rating": 5, "comment": "Excellent!"},
                        headers=headers)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        r = client.delete(f"/chat/{conv_id}", headers=headers)
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

        r = client.get(f"/chat/{conv_id}", headers=headers)
        assert r.status_code == 404

    def test_unauthorized_access(self, client):
        r = client.get("/chat/history")
        assert r.status_code == 401

        r = client.get("/chat/1")
        assert r.status_code == 401

    def test_admin_endpoints(self, client, registered_user, admin_user):
        admin_headers = {"Authorization": f"Bearer {admin_user['access_token']}"}
        user_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}

        r = client.get("/admin/stats", headers=admin_headers)
        assert r.status_code == 200
        stats = r.json()
        assert "total_users" in stats

        r = client.get("/admin/users", headers=admin_headers)
        assert r.status_code == 200

        r = client.get("/admin/export", headers=admin_headers)
        assert r.status_code == 200
        export = r.json()
        assert "users" in export

        r = client.get("/admin/stats", headers=user_headers)
        assert r.status_code == 403

        r = client.get("/admin/users", headers=user_headers)
        assert r.status_code == 403

    def test_cors_headers(self, client):
        r = client.options("/", headers={"Origin": "http://localhost:8002"})
        assert "access-control-allow-origin" in r.headers

    def test_security_headers(self, client):
        r = client.get("/health")
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options") == "DENY"
        assert r.headers.get("x-xss-protection") == "1; mode=block"

    def test_rate_limiting(self):
        from part2_rag.api import _rate_limit, rate_limit_store
        rate_limit_store.clear()
        key = "test_rate_key"
        for i in range(3):
            assert _rate_limit(key, max_requests=3, window=60) is True
        assert _rate_limit(key, max_requests=3, window=60) is False
        rate_limit_store.clear()
