from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import sqlite3
import threading
from typing import List, Optional, Tuple
import uuid

import httpx

from lnvo_lens.config import settings
from lnvo_lens.schemas import (
    DocumentResult,
    DocumentStatus,
    JobResponse,
    JobStatus,
)
from lnvo_lens.services.image_prep import file_bytes_to_base64_images
from lnvo_lens.services.llm_client import extract_invoice_from_images
from lnvo_lens.services.validator import validate_invoice

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

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
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

    def _update_job(
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

    @staticmethod
    def _process_single_file(filename: str, content: bytes) -> DocumentResult:
        try:
            images = file_bytes_to_base64_images(filename, content)
            invoice_data = extract_invoice_from_images(images)
            warnings = validate_invoice(invoice_data)
            return DocumentResult(
                filename=filename,
                status=DocumentStatus.NEEDS_REVIEW if warnings else DocumentStatus.SUCCESS,
                data=invoice_data,
                warnings=warnings,
            )
        except Exception as exc:
            return DocumentResult(
                filename=filename,
                status=DocumentStatus.ERROR,
                error=str(exc),
            )

    def _send_webhook(self, job_id: str, callback_url: Optional[str]) -> None:
        if not callback_url:
            return
        job = self.get_job(job_id)
        if not job:
            return
        try:
            httpx.post(callback_url, json=job.model_dump(mode="json"), timeout=10.0)
        except Exception as exc:
            logger.warning("Failed to deliver webhook for job %s to %s: %s", job_id, callback_url, exc)

    def process_job(
        self,
        job_id: str,
        files: List[Tuple[str, bytes]],
        callback_url: Optional[str] = None,
    ) -> None:
        self._update_job(job_id, status=JobStatus.PROCESSING)

        max_workers = (
            settings.max_workers_gemini
            if settings.llm_provider == "gemini"
            else settings.max_workers_lm_studio
        )
        max_workers = max(1, min(max_workers, len(files)))

        results_by_index: List[Optional[DocumentResult]] = [None] * len(files)
        processed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_single_file, filename, content): idx
                for idx, (filename, content) in enumerate(files)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results_by_index[idx] = future.result()
                processed += 1
                completed_results = [r for r in results_by_index if r is not None]
                self._update_job(
                    job_id,
                    processed_files=processed,
                    results=completed_results,
                )

        final_results = [r for r in results_by_index if r is not None]
        self._update_job(
            job_id,
            status=JobStatus.COMPLETED,
            processed_files=len(files),
            results=final_results,
        )
        self._send_webhook(job_id, callback_url)


job_store = SQLiteJobStore()
