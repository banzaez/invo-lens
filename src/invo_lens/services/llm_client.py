import base64
import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from openai import OpenAI

from invo_lens.config import settings
from invo_lens.schemas import InvoiceData

SYSTEM_PROMPT = (
    "Ты — точная система для извлечения структурированных данных из накладных, счетов и инвойсов "
    "на английском (EN), украинском (UK) и русском (RU) языках.\n"
    "Правила:\n"
    "1. Верни строго JSON, соответствующий заданной схеме.\n"
    "2. Если поле отсутствует или нечитаемо, укажи null. НЕ ставь null, если значение видно на изображении — внимательно проверь заголовок (номер, дату), ИНН/VAT/Tax ID, правый столбец таблицы (Amount/Сумма) и блок итогов внизу (Subtotal, VAT, Total).\n"
    "3. Числа передавай как JSON-числа (float) с точностью до 2 знаков после запятой — обязательно сохраняй копейки/центы и не округляй до целых; убирай пробелы-разделители тысяч и заменяй запятую на точку "
    "(например, '140 580,50' -> 140580.5).\n"
    "4. Даты приводи строго к формату YYYY-MM-DD.\n"
    "5. Валюту указывай строго 3-буквенным ISO-кодом (RUB, UAH, USD, EUR), а не символом.\n"
    "6. В items включай все строки таблицы позиций по отдельности, не объединяй их и не включай строку 'Итого/Total/Всього'.\n"
    "7. Используй ТОЛЬКО эти ключи верхнего уровня (без обёрток вроде '.json' или '.invoice'), разделяя название, tax_id, счёт (р/р, IBAN) и адрес:\n"
    '{"document_number": "...", "document_date": "YYYY-MM-DD", '
    '"supplier": {"name": "...", "tax_id": "...", "bank_account": "...", "address": "..."}, '
    '"buyer": {"name": "...", "tax_id": "...", "bank_account": "...", "address": "..."}, '
    '"currency": "UAH", "total_without_vat": 0.0, "vat_amount": 0.0, "total_amount": 0.0, '
    '"items": [{"sku": "...", "name": "...", "quantity": 1.0, "unit": "...", "price": 0.0, "amount": 0.0, "vat_rate": "20%"}]}'
)


def _get_lm_studio_client() -> OpenAI:
    return OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        timeout=settings.request_timeout_seconds,
    )


def _extract_via_gemini(base64_images: List[str]) -> InvoiceData:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required when llm_provider='gemini'")

    client = genai.Client(api_key=settings.gemini_api_key)
    contents: List[Any] = [
        "Извлеки данные накладной из приложенных страниц документа."
    ]
    for img_b64 in base64_images:
        contents.append(
            types.Part.from_bytes(
                data=base64.b64decode(img_b64),
                mime_type="image/png",
            )
        )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=InvoiceData,
        ),
    )

    if response.parsed is not None and isinstance(response.parsed, InvoiceData):
        return response.parsed
    parsed = _normalize_parsed_json(json.loads(response.text or "{}"))
    return InvoiceData.model_validate(parsed)


def _normalize_parsed_json(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize alternative LLM JSON layouts into InvoiceData shape."""
    if len(parsed) == 1:
        only_val = next(iter(parsed.values()))
        if isinstance(only_val, str):
            try:
                only_val = json.loads(only_val)
            except json.JSONDecodeError:
                pass
        if isinstance(only_val, dict):
            parsed = only_val

    data = dict(parsed)

    if "buyer" not in data and isinstance(data.get("customer"), dict):
        data["buyer"] = data.pop("customer")

    for party, name_key, inn_key in (
        ("supplier", "supplier_name", "supplier_inn"),
        ("buyer", "buyer_name", "buyer_inn"),
    ):
        if party not in data and (name_key in data or inn_key in data):
            data[party] = {
                "name": data.pop(name_key, None),
                "tax_id": data.pop(inn_key, None),
            }
        elif isinstance(data.get(party), dict):
            obj = dict(data[party])
            if "tax_id" not in obj and "inn" in obj:
                obj["tax_id"] = obj.pop("inn")
            data[party] = obj

    if "total_without_vat" not in data and "subtotal" in data:
        data["total_without_vat"] = data.pop("subtotal")
    if "vat_amount" not in data and "tax_amount" in data:
        data["vat_amount"] = data.pop("tax_amount")

    if isinstance(data.get("items"), list):
        norm_items = []
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            if "sku" not in item:
                for alt in ("article", "code", "part_no", "vendor_code"):
                    if alt in item:
                        item["sku"] = item.pop(alt)
                        break
            if "name" not in item and "description" in item:
                item["name"] = item.pop("description")
            if "price" not in item and "unit_price" in item:
                item["price"] = item.pop("unit_price")
            if "vat_rate" not in item and "tax_rate" in item:
                item["vat_rate"] = item.pop("tax_rate")
            if isinstance(item.get("vat_rate"), (int, float)):
                item["vat_rate"] = f"{item['vat_rate']}%"
            norm_items.append(item)
        data["items"] = norm_items

    return data


def _check_totals(data: InvoiceData) -> Optional[str]:
    """Verify basic arithmetic consistency of extracted invoice totals."""
    if (
        data.total_without_vat is not None
        and data.vat_amount is not None
        and data.total_amount is not None
    ):
        expected = data.total_without_vat + data.vat_amount
        if abs(expected - data.total_amount) > 0.05:
            return (
                f"Математическое расхождение: total_without_vat ({data.total_without_vat}) + "
                f"vat_amount ({data.vat_amount}) = {expected}, но total_amount = {data.total_amount}."
            )
    return None


def _check_completeness(data: InvoiceData) -> Optional[str]:
    """Check if key invoice fields were left null despite likely being present."""
    missing = []
    if data.document_number is None:
        missing.append("document_number")
    if data.document_date is None:
        missing.append("document_date")
    if data.total_amount is None:
        missing.append("total_amount (и total_without_vat / vat_amount)")
    if data.items and any(item.amount is None for item in data.items):
        missing.append("amount в каждой позиции items")
    if missing:
        return (
            "Не заполнены важные поля: " + ", ".join(missing) + ". "
            "Внимательно посмотри на заголовок, правый столбец таблицы (Amount/Сумма) и блок итогов внизу, и заполни их числами/строками (не клади итоги в vat_rate)."
        )
    return None


def extract_invoice_from_images(base64_images: List[str]) -> InvoiceData:
    """Send document images to configured LLM provider and return validated InvoiceData."""
    if settings.llm_provider == "gemini":
        return _extract_via_gemini(base64_images)

    client = _get_lm_studio_client()

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": "Извлеки данные накладной из приложенных страниц документа."}
    ]
    for img_b64 in base64_images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "invoice_data",
            "strict": True,
            "schema": InvoiceData.model_json_schema(),
        },
    }

    response = client.chat.completions.create(
        model=settings.lm_studio_model,
        messages=messages,
        temperature=0.0,
        response_format=response_format,
        extra_body={"chat_template_kwargs": {"enable_thinking": settings.lm_studio_enable_thinking}},
    )

    raw_content = response.choices[0].message.content or "{}"
    parsed = _normalize_parsed_json(json.loads(raw_content))
    return InvoiceData.model_validate(parsed)
