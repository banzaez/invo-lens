import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from invo_lens.config import settings
from invo_lens.schemas import DocumentResult, DocumentStatus, JobStatus
from invo_lens.services.image_prep import file_bytes_to_base64_images
from invo_lens.services.job_store import SQLiteJobStore, job_store
from invo_lens.services.llm_client import extract_invoice_from_images
from invo_lens.services.validator import validate_invoice

logger = logging.getLogger(__name__)


class JobProcessor:
    """Orchestrates background document recognition and webhook delivery."""

    def __init__(self, store: SQLiteJobStore) -> None:
        self._store = store

    @staticmethod
    def _process_single_file(filename: str, file_path: Path) -> DocumentResult:
        try:
            content = file_path.read_bytes()
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
        job = self._store.get_job(job_id)
        if not job:
            return
        try:
            httpx.post(callback_url, json=job.model_dump(mode="json"), timeout=10.0)
        except Exception as exc:
            logger.warning("Failed to deliver webhook for job %s to %s: %s", job_id, callback_url, exc)

    def process_job(
        self,
        job_id: str,
        files: List[Tuple[str, Path]],
        temp_dir: Optional[Path] = None,
        callback_url: Optional[str] = None,
    ) -> None:
        try:
            self._store.update_job(job_id, status=JobStatus.PROCESSING)

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
                    executor.submit(self._process_single_file, filename, file_path): idx
                    for idx, (filename, file_path) in enumerate(files)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results_by_index[idx] = future.result()
                    processed += 1
                    completed_results = [r for r in results_by_index if r is not None]
                    self._store.update_job(
                        job_id,
                        processed_files=processed,
                        results=completed_results,
                    )

            final_results = [r for r in results_by_index if r is not None]
            self._store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                processed_files=len(files),
                results=final_results,
            )
            self._send_webhook(job_id, callback_url)
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


job_processor = JobProcessor(job_store)
