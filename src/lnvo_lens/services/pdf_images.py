import base64
import io
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from lnvo_lens.config import settings


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _pil_to_base64_png(image: Image.Image) -> str:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def file_bytes_to_base64_images(filename: str, content: bytes) -> List[str]:
    """Convert PDF pages or an image file into a list of base64-encoded PNG strings."""
    lower = filename.lower()

    if lower.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        images: List[str] = []
        zoom = settings.pdf_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        try:
            page_count = min(len(doc), settings.max_pdf_pages)
            for page_index in range(page_count):
                page = doc[page_index]
                pix = page.get_pixmap(matrix=matrix)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(_pil_to_base64_png(img))
        finally:
            doc.close()
        if not images:
            raise ValueError("PDF contains no pages")
        return images

    if any(lower.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
        img = Image.open(io.BytesIO(content))
        return [_pil_to_base64_png(img)]

    raise ValueError(f"Unsupported file format: {filename}")
