"""
CPU-only PDF Parser for Core Server Stability

Docling의 GPU 메모리 문제를 해결하기 위한 완전한 대체 파서.
pypdfium2 + 구조 분석으로 100% CPU-only 처리.

License: BSD-3-Clause + MIT (no GPL dependencies)
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Try to import pypdfium2 (BSD-3-Clause)
try:
    import pypdfium2 as pdfium

    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False
    pdfium = None
    logger.warning("pypdfium2 not available. CPU-only parsing will be limited.")


@dataclass
class TableCell:
    """테이블 셀 정보."""

    row: int
    col: int
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    row_span: int = 1
    col_span: int = 1


@dataclass
class TableData:
    """테이블 데이터."""

    table_id: str
    page: int
    shape: Dict[str, int]  # {"rows": int, "cols": int}
    cells: List[TableCell] = field(default_factory=list)


@dataclass
class ParagraphData:
    """문단 데이터."""

    text: str
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None


class CPUPDFParser:
    """
    100% CPU-only PDF 파서.

    Docling 대신 pypdfium2를 사용하여 GPU 메모리 문제를 완전히 회피.
    core 서버 안정성을 위해 설계됨.

    특징:
    - GPU 완전 비사용 (CUDA tensors 없음)
    - 메모리 사용량 제한
    - 타임아웃 지원
    - 신호 처리로 강제 종료 가능
    """

    def __init__(
        self,
        max_memory_mb: int = 2048,  # 2GB 메모리 제한
        timeout_seconds: int = 120,  # 2분 타임아웃
        enable_table_extraction: bool = True,
    ) -> None:
        """
        Args:
            max_memory_mb: 최대 메모리 사용량 (MB)
            timeout_seconds: 처리 타임아웃 (초)
            enable_table_extraction: 테이블 추출 활성화
        """
        if not PYPDFIUM2_AVAILABLE:
            raise ImportError("pypdfium2 is required for CPU-only parsing. Install: pip install pypdfium2")

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.timeout_seconds = timeout_seconds
        self.enable_table_extraction = enable_table_extraction

        # 강제 종료 이벤트
        self._force_stop = multiprocessing.Event()

    def _check_memory(self) -> bool:
        """메모리 사용량 체크."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            if mem_info.rss > self.max_memory_bytes:
                logger.warning(f"Memory limit exceeded: {mem_info.rss / 1024 / 1024:.1f}MB > {self.max_memory_mb}MB")
                return False
            return True
        except ImportError:
            return True  # psutil이 없으면 체크 건너뛰기

    def _timeout_handler(self, signum, frame):
        """타임아웃 신호 핸들러."""
        logger.warning(f"PDF processing timed out after {self.timeout_seconds}s")
        self._force_stop.set()
        raise TimeoutError(f"PDF processing timed out after {self.timeout_seconds} seconds")

    def _extract_text_from_page(self, page: Any, page_num: int) -> List[str]:
        """单个 페이지에서 텍스트 추출."""
        paragraphs = []

        try:
            textpage = page.get_textpage()
            text = textpage.get_text_range()

            if text:
                # 빈 줄로 문단 분리
                raw_paragraphs = re.split(r"\n\s*\n+", text)

                for para in raw_paragraphs:
                    # 정리: 여러 공백을 하나로, 양쪽 공백 제거
                    cleaned = " ".join(para.split())
                    if cleaned and len(cleaned) > 2:
                        paragraphs.append(cleaned)

            textpage.close()

        except Exception as e:
            logger.debug(f"Text extraction failed on page {page_num}: {e}")

        return paragraphs

    def _extract_tables_simple(self, page: Any, page_num: int) -> List[TableData]:
        """간단한 테이블 추출 (pypdfium2 기반)."""
        tables = []

        if not self.enable_table_extraction:
            return tables

        try:
            # 페이지의 모든 텍스트 블록에서 테이블 추정
            textpage = page.get_textpage()
            text = textpage.get_text_range()

            # 테이블 패턴 감지 (여러 공백으로 구분된 행)
            lines = text.split("\n")

            table_candidates = []
            in_table = False
            current_table = []

            for line in lines:
                # 테이블 행 패턴: 여러 공백으로 구분된 텍스트
                parts = [p.strip() for p in line.split("  ") if p.strip()]

                if len(parts) >= 2 and all(len(p) > 0 for p in parts):
                    # 테이블일 가능성 높음
                    if not in_table:
                        in_table = True
                        current_table = []
                    current_table.append(parts)
                else:
                    if in_table:
                        if len(current_table) >= 2:  # 최소 2행
                            table_candidates.append(current_table)
                        in_table = False
                        current_table = []

            # 마지막 테이블 처리
            if in_table and len(current_table) >= 2:
                table_candidates.append(current_table)

            # 테이블 데이터 변환
            for idx, table_rows in enumerate(table_candidates):
                num_rows = len(table_rows)
                num_cols = max(len(row) for row in table_rows) if table_rows else 0

                if num_rows < 2 or num_cols < 2:
                    continue  # 유효하지 않은 테이블

                cells = []
                for row_idx, row in enumerate(table_rows):
                    for col_idx, cell_text in enumerate(row):
                        cells.append(TableCell(row=row_idx, col=col_idx, text=cell_text))

                tables.append(
                    TableData(
                        table_id=f"table_{idx + 1}",
                        page=page_num,
                        shape={"rows": num_rows, "cols": num_cols},
                        cells=cells,
                    )
                )

            textpage.close()

        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")

        return tables

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        PDF 파일을 YAML-serializable dict로 변환.

        Args:
            file_path: PDF 파일 경로

        Returns:
            {
                "document": {...},
                "content": {"paragraphs": [...]},
                "tables": [...],
                "assets": {"images": []}
            }
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"CPU-only parsing: {file_path}")

        # 타임아웃 설정
        old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.timeout_seconds)

        try:
            pdf = pdfium.PdfDocument(str(file_path))

            all_paragraphs = []
            all_tables = []
            num_pages = len(pdf)

            for page_idx in range(num_pages):
                # 메모리 체크
                if not self._check_memory():
                    raise MemoryError("Memory limit exceeded during processing")

                page = pdf[page_idx]

                # 문단 추출
                paragraphs = self._extract_text_from_page(page, page_idx + 1)
                for para in paragraphs:
                    all_paragraphs.append(ParagraphData(text=para, page=page_idx + 1))

                # 테이블 추출
                if self.enable_table_extraction:
                    tables = self._extract_tables_simple(page, page_idx + 1)
                    all_tables.extend(tables)

                page.close()

                # 중간 가비지 컬렉션
                if page_idx % 10 == 0:
                    gc.collect()

            pdf.close()

            # 결과 구성
            result = {
                "document": {
                    "source_path": str(file_path),
                    "format": "pdf",
                    "encrypted": False,
                    "parser": "cpu_pdf_parser",  # docling이 아님
                    "text_extractor": "pypdfium2",
                    "page_count": num_pages,
                    "ocr_enabled": False,  # CPU 파서는 OCR 없음
                    "table_extraction": self.enable_table_extraction,
                    "cpu_only_mode": True,  # CPU 전용 모드 표시
                },
                "content": {"paragraphs": [{"text": p.text, "page": p.page} for p in all_paragraphs]},
                "tables": [
                    {
                        "table_id": t.table_id,
                        "page": t.page,
                        "shape": t.shape,
                        "cells": [
                            {
                                "row": c.row,
                                "col": c.col,
                                "text": c.text,
                                "row_span": c.row_span,
                                "col_span": c.col_span,
                            }
                            for c in t.cells
                        ],
                    }
                    for t in all_tables
                ],
                "assets": {"images": []},
            }

            logger.info(
                f"Parsed {file_path.name}: "
                f"{len(all_paragraphs)} paragraphs, "
                f"{len(all_tables)} tables, "
                f"{num_pages} pages (CPU-only)"
            )

            return result

        except TimeoutError:
            raise
        except Exception as e:
            logger.error(f"CPU-only parsing failed: {e}")
            raise ValueError(f"PDF parsing failed: {e}")
        finally:
            # 타임아웃 및 신호 복원
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            # 강제 종료 이벤트 리셋
            self._force_stop.clear()
            gc.collect()


class SafeCPUParser:
    """
    안전한 CPU 파서 래퍼.

    별도 프로세스에서 실행하여 메인 프로세스 안정성 보장.
    프로세스 크래시 시에도 메인 시스템 영향 없음.
    """

    def __init__(
        self,
        max_memory_mb: int = 2048,
        timeout_seconds: int = 120,
    ) -> None:
        self.parser = CPUPDFParser(
            max_memory_mb=max_memory_mb,
            timeout_seconds=timeout_seconds,
        )

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """프로세스 안전하게 PDF 파싱."""

        def _parse_in_process(path: str) -> Dict[str, Any]:
            """별도 프로세스에서 파싱 실행."""
            # 메모리 제한 적용
            try:
                import resource

                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 2048, hard))  # 2GB
            except Exception:
                pass

            return self.parser.parse(Path(path))

        # multiprocessing으로 별도 프로세스에서 실행
        try:
            with multiprocessing.Pool(processes=1) as pool:
                result = pool.apply(_parse_in_process, (str(file_path),))
                return result
        except Exception as e:
            logger.error(f"Safe CPU parsing failed: {e}")
            raise


def create_cpu_only_adapter(ocr_enabled: bool = False) -> "CPUPDFParser":
    """CPU-only 어댑터 팩토리."""
    return CPUPDFParser(
        max_memory_mb=2048,
        timeout_seconds=120,
        enable_table_extraction=True,
    )


# 대체 DoclingYAMLAdapter 구현
class DoclingYAMLAdapterCPUOnly:
    """
    Docling을 완전히 대체하는 CPU-only 어댑터.

    GPU 메모리 문제를 완전히 해결하기 위해 pypdfium2만 사용.
    docling을 사용하지 않으므로 GPU 충돌이 불가능.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        table_extraction: bool = True,
        ocr_engine: str = "paddle",
        use_cpu_fallback: bool = True,  # CPU-only 모드 강제
    ) -> None:
        """Initialize CPU-only adapter."""
        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.ocr_engine = ocr_engine
        self.use_cpu_fallback = use_cpu_fallback

        if use_cpu_fallback:
            # CPU-only 파서 사용
            self._parser = CPUPDFParser(
                max_memory_mb=2048,
                timeout_seconds=120,
                enable_table_extraction=table_extraction,
            )
        else:
            raise NotImplementedError("GPU mode not supported in CPU-only adapter")

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using CPU-only parser (no docling)."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"CPU-only parsing: {file_path}")

        # GPU 완전 비활성화 (추가 안전장치)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["DOCLING_DEVICE"] = "cpu"

        try:
            result = self._parser.parse(file_path)
            result["document"]["parser"] = "cpu_pdf_parser"
            result["document"]["cpu_only_mode"] = True
            return result
        except Exception as e:
            logger.error(f"CPU-only parsing failed: {e}")
            raise


__all__ = [
    "CPUPDFParser",
    "SafeCPUParser",
    "DoclingYAMLAdapterCPUOnly",
    "ParagraphData",
    "TableData",
    "TableCell",
]
