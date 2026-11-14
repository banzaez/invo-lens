import base64
import io
from typing import List

import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageOps

from invo_lens.config import settings


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _enhance_image(image: Image.Image) -> Image.Image:
    """Normalize orientation, size, contrast, and sharpness for better OCR/VLM accuracy."""
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    w, h = image.size
    max_side = max(w, h)
    if max_side < 1400:
        scale = 1600 / max_side
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max_side > 2400:
        scale = 2400 / max_side
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    image = ImageOps.autocontrast(image, cutoff=0.5)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    return image


def _pil_to_base64_png(image: Image.Image) -> str:
    image = _enhance_image(image)
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
