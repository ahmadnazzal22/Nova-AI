import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from part2_rag import api as rag_api


@pytest.fixture
def client():
    return TestClient(rag_api.app)


@pytest.fixture
def test_user(client):
    resp = client.post("/auth/register", json={
        "username": "testuser", "email": "test@example.com", "password": "testpass123"
    })
    assert resp.status_code == 200
    return resp.json()


class TestAuth:
    def test_register_success(self, client):
        resp = client.post("/auth/register", json={
            "username": "newuser", "email": "new@example.com", "password": "secure123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_username(self, client, test_user):
        resp = client.post("/auth/register", json={
            "username": "testuser", "email": "other@example.com", "password": "testpass123"
        })
        assert resp.status_code == 400
        assert "Username" in resp.json()["detail"]

    def test_register_duplicate_email(self, client, test_user):
        resp = client.post("/auth/register", json={
            "username": "other", "email": "test@example.com", "password": "testpass123"
        })
        assert resp.status_code == 400
        assert "Email" in resp.json()["detail"]

    def test_register_invalid_data(self, client):
        resp = client.post("/auth/register", json={"username": "ab", "email": "bad", "password": "12"})
        assert resp.status_code == 422

    def test_login_success(self, client, test_user):
        resp = client.post("/auth/login", json={"username": "testuser", "password": "testpass123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_wrong_password(self, client, test_user):
        resp = client.post("/auth/login", json={"username": "testuser", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_nonexistent(self, client):
        resp = client.post("/auth/login", json={"username": "nobody", "password": "testpass123"})
        assert resp.status_code == 401

    def test_login_deactivated(self, client, test_user):
        token = test_user["access_token"]
        client.delete("/auth/account", headers={"Authorization": f"Bearer {token}"})
        resp = client.post("/auth/login", json={"username": "testuser", "password": "testpass123"})
        assert resp.status_code == 403

    def test_refresh_token(self, client, test_user):
        resp = client.post("/auth/refresh", json={"refresh_token": test_user["refresh_token"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_invalid(self, client):
        resp = client.post("/auth/refresh", json={"refresh_token": "invalid-token"})
        assert resp.status_code == 401

    def test_me_endpoint(self, client, test_user):
        token = test_user["access_token"]
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert "id" in data
        assert "role" in data

    def test_me_unauthorized(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_update_profile(self, client, test_user):
        token = test_user["access_token"]
        resp = client.put("/auth/profile", json={"email": "updated@example.com"},
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["email"] == "updated@example.com"

    def test_get_settings(self, client, test_user):
        token = test_user["access_token"]
        resp = client.get("/auth/settings", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["theme"] == "light"
        assert data["streaming_enabled"] is True

    def test_update_settings(self, client, test_user):
        token = test_user["access_token"]
        resp = client.put("/auth/settings", json={"theme": "dark", "default_sources": 5},
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    def test_delete_account(self, client, test_user):
        token = test_user["access_token"]
        resp = client.delete("/auth/account", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_chat_history_empty(self, client, test_user):
        token = test_user["access_token"]
        resp = client.get("/chat/history", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json() == []
