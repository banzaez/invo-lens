# invo-lens

Сервис распознавания накладных и счетов из PDF и изображений (PNG, JPG, WEBP) через локальную LLM (LM Studio) или Gemini.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Скопируйте `.env.example` в `.env` и настройте параметры:

```bash
cp .env.example .env
```

## Запуск

```bash
uvicorn invo_lens.main:app --reload
```

Документация Swagger будет доступна по адресу `http://127.0.0.1:8000/docs`.

## Основные эндпоинты

- `GET /health` — проверка состояния сервиса.
- `POST /api/v1/jobs` — создание задачи распознавания (`multipart/form-data`, поля `files` и опционально `callback_url`).
- `GET /api/v1/jobs` — список задач (`limit`, `offset`).
- `GET /api/v1/jobs/{job_id}` — статус и результаты задачи.
- `DELETE /api/v1/jobs/{job_id}` — удаление задачи.

Пример отправки файла:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs \
  -F "files=@test/example/en_invoice_it_services.pdf"
```

## Тесты и линтер

```bash
pytest
ruff check src test
```
