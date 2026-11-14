from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class InvoiceItem(BaseModel):
    name: str = Field(..., description="Точное наименование товара или услуги из строки таблицы")
    quantity: Optional[float] = Field(None, description="Количество (число, например 2.5)")
    unit: Optional[str] = Field(None, description="Единица измерения (например: шт, кг, л, м, hours, pcs, послуга)")
    price: Optional[float] = Field(None, description="Цена за единицу (число без пробелов и валюты)")
    amount: Optional[float] = Field(None, description="Итоговая сумма по этой строке (число)")
    vat_rate: Optional[str] = Field(None, description="Ставка НДС/ПДВ/VAT по строке (например: '20%', '0%', 'Без НДС', 'Без ПДВ')")
    status: DocumentStatus = Field(DocumentStatus.SUCCESS, description="Статус проверки строки")
    warning: Optional[str] = Field(None, description="Причина перепроверки строки, если есть")


class PartyInfo(BaseModel):
    name: Optional[str] = Field(None, description="Только наименование организации или ФОП/ИП (без адреса и счёта)")
    tax_id: Optional[str] = Field(
        None,
        description="Универсальный налоговый/регистрационный номер: ИНН (RU), ЄДРПОУ или ІПН (UK), VAT или Tax ID (EN). Без префиксов",
    )
    bank_account: Optional[str] = Field(None, description="Расчётный счёт / IBAN / р/р (например: 'UA2730033500002600766666666666')")
    address: Optional[str] = Field(None, description="Адрес или город (например: 'м. Чернігів')")


class InvoiceData(BaseModel):
    document_number: Optional[str] = Field(None, description="Номер документа (например: 'INV-2026-0142', 'ВН-000452', '1024')")
    document_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Дата документа строго в формате YYYY-MM-DD (например: '2026-11-05')",
    )
    supplier: Optional[PartyInfo] = Field(None, description="Реквизиты поставщика / продавца / исполнителя")
    buyer: Optional[PartyInfo] = Field(None, description="Реквизиты покупателя / заказчика")
    currency: str = Field(
        "RUB",
        pattern=r"^[A-Z]{3}$",
        description="3-буквенный ISO-код валюты: RUB, UAH, USD, EUR (не символ)",
    )
    total_without_vat: Optional[float] = Field(None, description="Итоговая сумма без НДС/ПДВ/VAT (число)")
    vat_amount: Optional[float] = Field(None, description="Сумма НДС/ПДВ/VAT (число, 0.0 если без НДС)")
    total_amount: Optional[float] = Field(None, description="Итоговая сумма к оплате с НДС (число)")
    items: List[InvoiceItem] = Field(
        default_factory=list,
        description="Все товарные/услуговые позиции из таблицы документа (без итоговых строк)",
    )


class DocumentResult(BaseModel):
    filename: str
    status: DocumentStatus
    data: Optional[InvoiceData] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_files: int


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    total_files: int
    processed_files: int
    results: List[DocumentResult] = Field(default_factory=list)
