"""PDF triage and classification module.

PDF 문서를 OCR 필요 여부로 분류하고 배치 처리합니다.
"""

from .pdf_classifier import classify_pdf_batch, print_triage_report

__all__ = ["classify_pdf_batch", "print_triage_report"]
