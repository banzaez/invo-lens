from typing import List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from lnvo_lens.schemas import JobCreateResponse, JobResponse
from lnvo_lens.services.job_store import job_store

app = FastAPI(
    title="lnvo-lens",
    description="Сервис распознавания накладных из PDF и фото через локальную LLM (LM Studio)",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/jobs", response_model=JobCreateResponse, status_code=202)
async def create_recognition_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Один или несколько файлов (PDF, PNG, JPG)"),
    callback_url: Optional[str] = Form(None, description="Опциональный URL для webhook-уведомления по завершении"),
) -> JobCreateResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Не передано ни одного файла")

    file_payloads: List[Tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        file_payloads.append((upload.filename or "unnamed", content))

    job = job_store.create_job(total_files=len(file_payloads), callback_url=callback_url)
    background_tasks.add_task(job_store.process_job, job.job_id, file_payloads, callback_url)

    return JobCreateResponse(
        job_id=job.job_id,
        status=job.status,
        total_files=job.total_files,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_recognition_job(job_id: str) -> JobResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job
