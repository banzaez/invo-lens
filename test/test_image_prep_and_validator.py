import io

import pytest
from PIL import Image

from invo_lens.schemas import DocumentStatus, InvoiceData, InvoiceItem
from invo_lens.services.image_prep import _enhance_image, file_bytes_to_base64_images
from invo_lens.services.validator import validate_invoice


def test_enhance_image_upscales_small_image():
    img = Image.new("RGB", (400, 300), color=(120, 120, 120))
    enhanced = _enhance_image(img)
    assert max(enhanced.size) == 1600
    assert enhanced.mode == "RGB"


def test_enhance_image_downscales_large_image():
    img = Image.new("RGB", (4800, 3600), color=(200, 200, 200))
    enhanced = _enhance_image(img)
    assert max(enhanced.size) == 2400
    assert enhanced.mode == "RGB"


def test_enhance_image_converts_rgba_to_rgb():
    img = Image.new("RGBA", (1600, 1200), color=(255, 255, 255, 255))
    enhanced = _enhance_image(img)
    assert enhanced.mode == "RGB"


def test_file_bytes_to_base64_images_valid_png():
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = file_bytes_to_base64_images("invoice.png", buf.getvalue())
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0


def test_file_bytes_to_base64_images_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        file_bytes_to_base64_images("invoice.txt", b"hello")


def test_validate_invoice_valid_with_vat():
    data = InvoiceData(
        document_number="INV-001",
        document_date="2026-11-14",
        total_without_vat=100.0,
        vat_amount=20.0,
        total_amount=120.0,
        items=[
            InvoiceItem(name="Item A", quantity=2.0, price=50.0, amount=100.0, vat_rate="20%"),
        ],
    )
    warnings = validate_invoice(data)
    assert warnings == []
    assert data.items[0].status == DocumentStatus.SUCCESS


def test_validate_invoice_detects_missing_fields_and_mismatch():
    data = InvoiceData(
        document_number=None,
        document_date=None,
        total_without_vat=100.0,
        vat_amount=20.0,
        total_amount=150.0,
        items=[
            InvoiceItem(name="Item A", quantity=2.0, price=50.0, amount=90.0, vat_rate="20%"),
        ],
    )
    warnings = validate_invoice(data)
    assert len(warnings) >= 3
    assert data.items[0].status == DocumentStatus.NEEDS_REVIEW
    assert data.items[0].warning is not None
