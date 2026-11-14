from fastapi.testclient import TestClient

from invo_lens.main import app


def test_api_key_auth(monkeypatch):
    monkeypatch.setattr("invo_lens.main.settings.api_key", "secret-token-123")
    client = TestClient(app)

    # /health stays public
    assert client.get("/health").status_code == 200

    # Missing or wrong key -> 401
    assert client.get("/api/v1/jobs").status_code == 401
    assert client.get("/api/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401

    # Valid X-API-Key -> 200
    assert client.get("/api/v1/jobs", headers={"X-API-Key": "secret-token-123"}).status_code == 200

    # Valid Authorization: Bearer -> 200
    assert client.get("/api/v1/jobs", headers={"Authorization": "Bearer secret-token-123"}).status_code == 200
