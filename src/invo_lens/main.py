import hmac
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, Security, UploadFile
from fastapi.security import APIKeyHeader

from invo_lens.config import settings
from invo_lens.schemas import (
    JobCreateResponse,
    JobResponse,
    NomenclatureItem,
    NomenclatureUploadRequest,
    NomenclatureUploadResponse,
)
from invo_lens.services.job_processor import job_processor
from invo_lens.services.job_store import job_store
from invo_lens.services.nomenclature_store import nomenclature_store

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    api_key_header: Optional[str] = Security(_api_key_header),
    authorization: Optional[str] = Header(None),
) -> None:
    if not settings.api_key:
        return
    token = api_key_header
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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


@app.post("/api/v1/jobs", response_model=JobCreateResponse, status_code=202, dependencies=[Depends(verify_api_key)])
async def create_recognition_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Один или несколько файлов (PDF, PNG, JPG, WEBP)"),
    callback_url: Optional[str] = Form(None, description="Опциональный URL для webhook-уведомления по завершении"),
    client_id: Optional[str] = Form(None, description="Опциональный ID клиента для сопоставления номенклатуры"),
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
    background_tasks.add_task(
        job_processor.process_job,
        job.job_id,
        file_paths,
        temp_dir,
        callback_url,
        client_id,
    )

    return JobCreateResponse(
        job_id=job.job_id,
        status=job.status,
        total_files=job.total_files,
    )


@app.get("/api/v1/jobs", response_model=List[JobResponse], dependencies=[Depends(verify_api_key)])
def list_recognition_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[JobResponse]:
    return job_store.list_jobs(limit=limit, offset=offset)


@app.post("/api/v1/nomenclature", response_model=NomenclatureUploadResponse, dependencies=[Depends(verify_api_key)])
def upload_nomenclature(payload: NomenclatureUploadRequest) -> NomenclatureUploadResponse:
    upserted = nomenclature_store.upsert_items(payload.client_id, payload.items)
    return NomenclatureUploadResponse(client_id=payload.client_id, upserted=upserted)


@app.get("/api/v1/nomenclature/{client_id}", response_model=List[NomenclatureItem], dependencies=[Depends(verify_api_key)])
def list_nomenclature(client_id: str) -> List[NomenclatureItem]:
    return nomenclature_store.list_items(client_id)


@app.delete("/api/v1/nomenclature/{client_id}", dependencies=[Depends(verify_api_key)])
def delete_nomenclature(client_id: str) -> dict:
    deleted = nomenclature_store.delete_client_items(client_id)
    return {"client_id": client_id, "deleted": deleted}


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(verify_api_key)])
def get_recognition_job(job_id: str) -> JobResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


@app.delete("/api/v1/jobs/{job_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_recognition_job(job_id: str) -> None:
    if not job_store.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Задача не найдена")
