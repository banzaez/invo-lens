import json
from typing import List

from openai import OpenAI

from lnvo_lens.config import settings
from lnvo_lens.schemas import InvoiceData


SYSTEM_PROMPT = (
    "Ты — точная система для извлечения структурированных данных из накладных и счетов. "
    "Проанализируй изображения документа и верни строго JSON, соответствующий заданной схеме. "
    "Если какое-то поле отсутствует или нечитаемо, укажи null. "
    "Числа передавай как числа (float), даты — в формате YYYY-MM-DD."
)


def _get_client() -> OpenAI:
    return OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        timeout=settings.request_timeout_seconds,
    )


def extract_invoice_from_images(base64_images: List[str]) -> InvoiceData:
    """Send document images to LM Studio (OpenAI-compatible API) and return validated InvoiceData."""
    client = _get_client()

    user_content = [
        {"type": "text", "text": "Извлеки данные накладной из приложенных страниц документа."}
    ]
    for img_b64 in base64_images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )

    schema = InvoiceData.model_json_schema()

    response = client.chat.completions.create(
        model=settings.lm_studio_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_data",
                "strict": True,
                "schema": schema,
            },
        },
    )

    raw_content = response.choices[0].message.content or "{}"
    parsed = json.loads(raw_content)
    return InvoiceData.model_validate(parsed)
