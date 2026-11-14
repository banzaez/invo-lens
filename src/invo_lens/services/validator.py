import re
from typing import List, Optional

from invo_lens.schemas import DocumentStatus, InvoiceData


def _parse_vat_percent(vat_rate: Optional[str]) -> Optional[float]:
    if not vat_rate:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", vat_rate)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def validate_invoice(data: InvoiceData) -> List[str]:
    """Run post-processing checks on extracted invoice data and return warnings for human review."""
    warnings: List[str] = []

    if not data.document_number:
        warnings.append("Не распознан номер документа (document_number)")
    if not data.document_date:
        warnings.append("Не распознана дата документа (document_date)")
    if data.total_amount is None:
        warnings.append("Не распознана итоговая сумма (total_amount)")

    if (
        data.total_without_vat is not None
        and data.vat_amount is not None
        and data.total_amount is not None
    ):
        expected_total = round(data.total_without_vat + data.vat_amount, 2)
        if abs(expected_total - data.total_amount) > 0.05:
            warnings.append(
                f"Итоги не сходятся: total_without_vat ({data.total_without_vat}) + "
                f"vat_amount ({data.vat_amount}) = {expected_total}, но total_amount = {data.total_amount}"
            )

    for idx, item in enumerate(data.items, start=1):
        if item.quantity is not None and item.price is not None and item.amount is not None:
            base = round(item.quantity * item.price, 2)
            vat_pct = _parse_vat_percent(item.vat_rate)
            with_vat = round(base * (1 + vat_pct / 100.0), 2) if vat_pct is not None else None

            matches_base = abs(base - item.amount) <= 0.05
            matches_vat = with_vat is not None and abs(with_vat - item.amount) <= 0.05
            if not matches_base and not matches_vat:
                msg = (
                    f"{item.quantity} * {item.price} = {base}"
                    + (f" (с НДС {vat_pct}%: {with_vat})" if with_vat is not None else "")
                    + f", но в строке указано amount = {item.amount}"
                )
                item.status = DocumentStatus.NEEDS_REVIEW
                item.warning = msg
                warnings.append(f"Позиция #{idx} ('{item.name}'): {msg}")

    if data.items and all(item.amount is not None for item in data.items):
        items_sum = round(sum(item.amount for item in data.items if item.amount is not None), 2)
        matches_total = data.total_amount is not None and abs(items_sum - data.total_amount) <= 0.05
        matches_no_vat = (
            data.total_without_vat is not None and abs(items_sum - data.total_without_vat) <= 0.05
        )
        if (data.total_amount is not None or data.total_without_vat is not None) and not (
            matches_total or matches_no_vat
        ):
            warnings.append(
                f"Сумма по строкам ({items_sum}) не сходится ни с total_amount ({data.total_amount}), "
                f"ни с total_without_vat ({data.total_without_vat})"
            )

    return warnings
