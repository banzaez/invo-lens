from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from invo_lens.config import Settings, settings
from invo_lens.main import app
from invo_lens.schemas import JobStatus
from invo_lens.services.job_processor import JobProcessor
from invo_lens.services.job_store import SQLiteJobStore


@pytest.fixture()
def temp_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SQLiteJobStore:
    db_file = tmp_path / "test_jobs.db"
    store = SQLiteJobStore(db_path=str(db_file))
    processor = JobProcessor(store)
    monkeypatch.setattr("invo_lens.main.job_store", store)
    monkeypatch.setattr("invo_lens.main.job_processor", processor)
    return store


@pytest.fixture()
def client(temp_store: SQLiteJobStore) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_job_rejects_unsupported_extension(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/jobs",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert resp.status_code == 415


def test_create_job_rejects_oversized_file(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_file_size_mb", 0)
    resp = client.post(
        "/api/v1/jobs",
        files=[("files", ("invoice.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert resp.status_code == 413


def test_create_job_rejects_invalid_callback_url(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/jobs",
        files=[("files", ("invoice.pdf", b"%PDF-1.4", "application/pdf"))],
        data={"callback_url": "ftp://example.com/hook"},
    )
    assert resp.status_code == 400


def test_job_lifecycle_create_list_get_delete(client: TestClient) -> None:
    with patch("invo_lens.main.job_processor.process_job") as mock_process:
        create_resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("invoice.pdf", b"%PDF-1.4", "application/pdf"))],
        )
    assert create_resp.status_code == 202
    job_id = create_resp.json()["job_id"]
    mock_process.assert_called_once()

    list_resp = client.get("/api/v1/jobs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["job_id"] == job_id

    get_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == JobStatus.PENDING.value

    del_resp = client.delete(f"/api/v1/jobs/{job_id}")
    assert del_resp.status_code == 204
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404


def test_mark_stale_jobs_failed(tmp_path: Path) -> None:
    store = SQLiteJobStore(db_path=str(tmp_path / "stale.db"))
    job = store.create_job(total_files=1)
    assert job.status == JobStatus.PENDING

    updated = store.mark_stale_jobs_failed()
    assert updated == 1

    fetched = store.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.FAILED


def test_settings_requires_gemini_key() -> None:
    with pytest.raises(ValueError, match="gemini_api_key"):
        Settings(llm_provider="gemini", gemini_api_key=None, _env_file=None)
