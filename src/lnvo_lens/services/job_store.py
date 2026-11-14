from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import uuid

from lnvo_lens.schemas import (
    DocumentResult,
    DocumentStatus,
    JobResponse,
    JobStatus,
)
from lnvo_lens.services.llm_client import extract_invoice_from_images
from lnvo_lens.services.pdf_images import file_bytes_to_base64_images


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobResponse] = {}

    def create_job(self, total_files: int) -> JobResponse:
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
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        return self._jobs.get(job_id)

    def process_job(self, job_id: str, files: List[Tuple[str, bytes]]) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.PROCESSING
        job.updated_at = datetime.now(timezone.utc)

        for filename, content in files:
            try:
                images = file_bytes_to_base64_images(filename, content)
                invoice_data = extract_invoice_from_images(images)
                job.results.append(
                    DocumentResult(
                        filename=filename,
                        status=DocumentStatus.SUCCESS,
                        data=invoice_data,
                    )
                )
            except Exception as exc:
                job.results.append(
                    DocumentResult(
                        filename=filename,
                        status=DocumentStatus.ERROR,
                        error=str(exc),
                    )
                )
            finally:
                job.processed_files += 1
                job.updated_at = datetime.now(timezone.utc)

        job.status = JobStatus.COMPLETED
        job.updated_at = datetime.now(timezone.utc)


job_store = InMemoryJobStore()
