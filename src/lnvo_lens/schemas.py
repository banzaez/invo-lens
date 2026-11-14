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
    ERROR = "error"


class InvoiceItem(BaseModel):
    name: str = Field(..., description="Наименование товара или услуги")
    quantity: Optional[float] = Field(None, description="Количество")
    unit: Optional[str] = Field(None, description="Единица измерения (шт, кг, и т.д.)")
    price: Optional[float] = Field(None, description="Цена за единицу")
    amount: Optional[float] = Field(None, description="Общая сумма по позиции")
    vat_rate: Optional[str] = Field(None, description="Ставка НДС (например, 20%, Без НДС)")


class InvoiceData(BaseModel):
    document_number: Optional[str] = Field(None, description="Номер накладной / документа")
    document_date: Optional[str] = Field(None, description="Дата документа в формате YYYY-MM-DD")
    supplier_name: Optional[str] = Field(None, description="Наименование поставщика")
    supplier_inn: Optional[str] = Field(None, description="ИНН поставщика")
    buyer_name: Optional[str] = Field(None, description="Наименование покупателя")
    buyer_inn: Optional[str] = Field(None, description="ИНН покупателя")
    currency: str = Field("RUB", description="Валюта документа (например, RUB, USD, EUR)")
    total_without_vat: Optional[float] = Field(None, description="Итого без НДС")
    vat_amount: Optional[float] = Field(None, description="Сумма НДС")
    total_amount: Optional[float] = Field(None, description="Итоговая сумма к оплате с НДС")
    items: List[InvoiceItem] = Field(default_factory=list, description="Список позиций в накладной")


class DocumentResult(BaseModel):
    filename: str
    status: DocumentStatus
    data: Optional[InvoiceData] = None
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
