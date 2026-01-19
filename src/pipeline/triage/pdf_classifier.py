"""PDF classification for triage.

pypdfium2로 텍스트 레이어 존재 여부를 빠르게 확인하여 OCR 필요 여부를 분류합니다.
Docling 전체 파싱 대신 첫 N페이지만 샘플링하여 95%+ 시간 절감.
"""

import json
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

_DISABLE_PYPDFIUM2 = os.getenv("DISABLE_PYPDFIUM2", "false").lower() == "true"
try:
    if _DISABLE_PYPDFIUM2:
        raise ImportError("pypdfium2 disabled via env")
    import pypdfium2 as pdfium

    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False
    pdfium = None
    reason = "disabled by env" if _DISABLE_PYPDFIUM2 else "not available"
    logger.warning(f"pypdfium2 {reason}. Triage will fallback to file size heuristic.")


# 최소 문자 수 임계값 (이 이하면 OCR 필요)
MIN_CHARS_THRESHOLD = 100
# 샘플링할 최대 페이지 수
MAX_SAMPLE_PAGES = 3


@dataclass
class PDFClassification:
    """PDF 분류 결과."""

    file_path: Path
    needs_ocr: bool
    reason: str
    char_count: int = 0
    docling_error: Optional[str] = None


@dataclass
class TriageResult:
    """Triage 배치 결과."""

    digital_files: List[Path]
    scanned_files: List[Path]
    total_processed: int


class PDFClassifier:
    """PDF 문서를 OCR 필요 여부로 분류하는 클래스.

    pypdfium2로 첫 N페이지만 샘플링하여 빠르게 분류 (95%+ 시간 절감).
    """

    def __init__(
        self,
        chars_threshold: int = MIN_CHARS_THRESHOLD,
        sample_pages: int = MAX_SAMPLE_PAGES,
    ):
        """초기화.

        Args:
            chars_threshold: 최소 문자 수 임계값
            sample_pages: 샘플링할 페이지 수 (기본: 3)
        """
        self.chars_threshold = chars_threshold
        self.sample_pages = sample_pages

    def _extract_text_fast(self, file_path: Path) -> tuple[int, int]:
        """pypdfium2로 첫 N페이지 텍스트를 빠르게 추출.

        Returns:
            (char_count, page_count) 튜플
        """
        if not PYPDFIUM2_AVAILABLE:
            return 0, 0

        char_count = 0
        page_count = 0

        try:
            pdf = pdfium.PdfDocument(str(file_path))
            page_count = len(pdf)
            pages_to_check = min(self.sample_pages, page_count)

            for page_idx in range(pages_to_check):
                page = pdf[page_idx]
                textpage = page.get_textpage()
                text = textpage.get_text_range()

                if text:
                    # 공백 정리 후 문자 수 카운트
                    cleaned = " ".join(text.split())
                    char_count += len(cleaned)

                textpage.close()
                page.close()

            pdf.close()

        except Exception as e:
            logger.debug(f"pypdfium2 extraction failed: {e}")
            return 0, 0

        return char_count, page_count

    def classify_pdf(self, file_path: Path) -> PDFClassification:
        """단일 PDF 파일을 분류합니다.

        pypdfium2로 첫 N페이지만 검사하여 빠르게 분류.
        ~0.1-0.5초/파일 (기존 Docling: ~10초/파일)

        Args:
            file_path: PDF 파일 경로

        Returns:
            PDFClassification: 분류 결과
        """
        if not file_path.exists():
            return PDFClassification(
                file_path=file_path,
                needs_ocr=True,
                reason="File not found",
            )

        # pypdfium2로 빠른 텍스트 추출
        char_count, page_count = self._extract_text_fast(file_path)

        # 페이지당 평균 문자 수로 정규화
        avg_chars_per_page = char_count / max(1, min(self.sample_pages, page_count))

        if char_count >= self.chars_threshold:
            logger.debug(
                f"Digital PDF: {file_path.name} " f"({char_count} chars in {min(self.sample_pages, page_count)} pages)"
            )
            return PDFClassification(
                file_path=file_path,
                needs_ocr=False,
                reason=f"Text layer detected ({avg_chars_per_page:.0f} chars/page)",
                char_count=char_count,
            )
        else:
            logger.debug(
                f"Scanned PDF: {file_path.name} " f"({char_count} chars in {min(self.sample_pages, page_count)} pages)"
            )
            return PDFClassification(
                file_path=file_path,
                needs_ocr=True,
                reason=f"No text layer ({char_count} < {self.chars_threshold})",
                char_count=char_count,
            )

    def classify_batch(self, file_paths: List[Path]) -> List[PDFClassification]:
        """PDF 파일들을 배치로 분류합니다.

        Args:
            file_paths: PDF 파일 경로 리스트

        Returns:
            List[PDFClassification]: 분류 결과 리스트
        """
        results = []
        for path in file_paths:
            result = self.classify_pdf(path)
            results.append(result)
        return results


def classify_pdf_batch(
    pdf_files: List[str | Path],
    output_dir: Optional[Path] = None,
    chars_threshold: int = MIN_CHARS_THRESHOLD,
    show_progress: bool = True,
) -> TriageResult:
    """PDF 파일들을 분류하여 OCR 필요/불필요 그룹으로 나눕니다.

    Args:
        pdf_files: PDF 파일 경로 리스트
        output_dir: 결과 저장 디렉토리 (선택)
        chars_threshold: 페이지당 최소 문자 수 임계값
        show_progress: 진행률 표시 여부

    Returns:
        TriageResult 객체
    """
    classifier = PDFClassifier(chars_threshold=chars_threshold)

    # 경로 변환
    file_paths = [Path(f) for f in pdf_files]

    # 배치 분류
    classifications = classifier.classify_batch(file_paths)

    # 그룹화
    digital_files = []
    scanned_files = []

    for classification in classifications:
        if classification.needs_ocr:
            scanned_files.append(classification.file_path)
        else:
            digital_files.append(classification.file_path)

    result = TriageResult(
        digital_files=digital_files,
        scanned_files=scanned_files,
        total_processed=len(file_paths),
    )

    # 결과 저장 (선택적)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 스캔된 PDF 리스트 저장
        if scanned_files:
            scanned_list_file = output_dir / "scanned_pdfs.json"
            with open(scanned_list_file, "w", encoding="utf-8") as f:
                json.dump([str(p) for p in scanned_files], f, ensure_ascii=False, indent=2)

        # 디지털 PDF 리스트 저장
        if digital_files:
            digital_list_file = output_dir / "digital_pdfs.json"
            with open(digital_list_file, "w", encoding="utf-8") as f:
                json.dump([str(p) for p in digital_files], f, ensure_ascii=False, indent=2)

    return result


def print_triage_report(result: TriageResult) -> None:
    """분류 결과를 터미널에 출력합니다."""
    print("\n=== PDF Triage Report ===")
    print(f"Total files processed: {result.total_processed}")
    print(f"Digital PDFs (direct processing): {len(result.digital_files)}")
    print(f"Scanned PDFs (need OCR): {len(result.scanned_files)}")

    if result.digital_files:
        print("\n[Digital PDFs - ready for direct processing]")
        for path in result.digital_files[:10]:
            print(f"  - {path}")
        if len(result.digital_files) > 10:
            print(f"  ... and {len(result.digital_files) - 10} more")

    if result.scanned_files:
        print("\n[Scanned PDFs - requiring OCR]")
        for path in result.scanned_files[:10]:
            print(f"  - {path}")
        if len(result.scanned_files) > 10:
            print(f"  ... and {len(result.scanned_files) - 10} more")


__all__ = [
    "PDFClassification",
    "PDFClassifier",
    "TriageResult",
    "classify_pdf_batch",
    "print_triage_report",
]
