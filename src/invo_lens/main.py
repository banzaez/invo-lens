import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile

from invo_lens.config import settings
from invo_lens.schemas import JobCreateResponse, JobResponse
from invo_lens.services.job_processor import job_processor
from invo_lens.services.job_store import job_store

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    job_store.mark_stale_jobs_failed()
    yield


app = FastAPI(
    title="invo-lens",
    description="Сервис распознавания накладных из PDF и фото через локальную LLM (LM Studio) или Gemini",
    version="0.1.0",
    lifespan=lifespan,
)


def _validate_callback_url(callback_url: Optional[str]) -> None:
    if callback_url is None:
        return
    parsed = urlparse(callback_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Некорректный callback_url: требуется http:// или https://")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/jobs", response_model=JobCreateResponse, status_code=202)
async def create_recognition_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Один или несколько файлов (PDF, PNG, JPG, WEBP)"),
    callback_url: Optional[str] = Form(None, description="Опциональный URL для webhook-уведомления по завершении"),
) -> JobCreateResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Не передано ни одного файла")
    if len(files) > settings.max_files_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"Слишком много файлов: максимум {settings.max_files_per_job} за одну задачу",
        )
    _validate_callback_url(callback_url)

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    temp_dir = Path(tempfile.mkdtemp(prefix="invo_lens_"))
    file_paths: List[Tuple[str, Path]] = []
    try:
        for idx, upload in enumerate(files):
            filename = upload.filename or f"unnamed_{idx}"
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Неподдерживаемый формат файла '{filename}'. Разрешены: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )
            content = await upload.read()
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл '{filename}' превышает лимит {settings.max_file_size_mb} МБ",
                )
            dest_path = temp_dir / f"{idx}{ext}"
            dest_path.write_bytes(content)
            file_paths.append((filename, dest_path))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    job = job_store.create_job(total_files=len(file_paths), callback_url=callback_url)
    background_tasks.add_task(job_processor.process_job, job.job_id, file_paths, temp_dir, callback_url)

    return JobCreateResponse(
        job_id=job.job_id,
        status=job.status,
        total_files=job.total_files,
    )


@app.get("/api/v1/jobs", response_model=List[JobResponse])
def list_recognition_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[JobResponse]:
    return job_store.list_jobs(limit=limit, offset=offset)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_recognition_job(job_id: str) -> JobResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
def delete_recognition_job(job_id: str) -> None:
    if not job_store.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Задача не найдена")
