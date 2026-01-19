# SPDX-License-Identifier: MIT
"""OCR processing module for Dyarchy pipeline.

PaddleOCR 기반 직접 YAML 변환 레이어.
Docling wrapper 없이 독립적 사용 가능.
"""

from src.pipeline.ocr.paddle_to_yaml import (
    OCRCell,
    OCRTable,
    PaddleOCRResult,
    PaddleToYAML,
)

__all__ = [
    "PaddleToYAML",
    "PaddleOCRResult",
    "OCRTable",
    "OCRCell",
]
