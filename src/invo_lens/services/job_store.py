from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import sqlite3
import threading
from typing import List, Optional, Tuple
import uuid

import httpx

from invo_lens.config import settings
from invo_lens.schemas import (
    DocumentResult,
    DocumentStatus,
    JobResponse,
    JobStatus,
)
from invo_lens.services.image_prep import file_bytes_to_base64_images
from invo_lens.services.llm_client import extract_invoice_from_images
from invo_lens.services.validator import validate_invoice

logger = logging.getLogger(__name__)


class SQLiteJobStore:
    def __init__(self, db_path: str = settings.db_path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_files INTEGER NOT NULL,
                    processed_files INTEGER NOT NULL,
                    results_json TEXT NOT NULL,
                    callback_url TEXT
                )
                """
            )
            conn.commit()

    def create_job(self, total_files: int, callback_url: Optional[str] = None) -> JobResponse:
        now = datetime.now(timezone.utc)
        job_id = str(uuid.uuid4())
        job = JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            total_files=total_files,
            processed_files=0,
            results=[],
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, created_at, updated_at, total_files, processed_files, results_json, callback_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.status.value,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    job.total_files,
                    job.processed_files,
                    "[]",
                    callback_url,
                ),
            )
            conn.commit()
        return job

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobResponse:
        results_raw = json.loads(row["results_json"])
        return JobResponse(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            total_files=row["total_files"],
            processed_files=row["processed_files"],
            results=[DocumentResult.model_validate(item) for item in results_raw],
        )

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> List[JobResponse]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
        return cursor.rowcount > 0

    def mark_stale_jobs_failed(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE status IN (?, ?)",
                (JobStatus.FAILED.value, now, JobStatus.PENDING.value, JobStatus.PROCESSING.value),
            )
            conn.commit()
        return cursor.rowcount

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        processed_files: Optional[int] = None,
        results: Optional[List[DocumentResult]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return
            new_status = status.value if status is not None else row["status"]
            new_processed = processed_files if processed_files is not None else row["processed_files"]
            new_results = (
                json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False)
                if results is not None
                else row["results_json"]
            )
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, processed_files = ?, results_json = ?
                WHERE job_id = ?
                """,
                (new_status, now, new_processed, new_results, job_id),
            )
            conn.commit()


job_store = SQLiteJobStore()
