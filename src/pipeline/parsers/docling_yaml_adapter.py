"""Docling-based PDF to YAML adapter.

PDF를 구조화된 YAML로 직접 변환 (Markdown 거치지 않음).
셀 단위 bbox, OCR confidence 보존.

하이브리드 모드:
- pypdfium2: 텍스트 추출 (paragraphs)
- Docling: 구조 추출 (tables, layout)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import os
# threading removed - using process-level isolation for GPU stability

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
    logger.warning(f"pypdfium2 {reason}. Text extraction will use Docling fallback.")

# GPU memory management (license compliant)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logger.debug("torch not available for GPU memory management")

from src.pipeline.quality.table_quality import (
    attach_cell_reliability,
    attach_table_quality,
)


@dataclass(frozen=True)
class OCRLine:
    """OCR 결과 라인."""

    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (l, t, r, b)


class ParagraphExtractor:
    """PDF에서 문단을 추출하는 클래스.

    pypdfium2를 사용하여 텍스트 레이어를 직접 읽음.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def extract(self) -> List[str]:
        """pypdfium2를 사용하여 문단 추출."""
        if not PYPDFIUM2_AVAILABLE:
            logger.warning("pypdfium2 not available, returning empty paragraphs")
            return []

        paragraphs: List[str] = []

        try:
            pdf = pdfium.PdfDocument(str(self.file_path))

            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                textpage = page.get_textpage()
                text = textpage.get_text_range()

                if text:
                    raw_paragraphs = re.split(r"\n\s*\n+", text)

                    for para in raw_paragraphs:
                        cleaned = " ".join(para.split())
                        if cleaned and len(cleaned) > 2:
                            paragraphs.append(cleaned)

                textpage.close()
                page.close()

            pdf.close()
            logger.debug(f"pypdfium2 extracted {len(paragraphs)} paragraphs")

        except Exception as e:
            logger.warning(f"pypdfium2 text extraction failed: {e}")

        return paragraphs


class TableExtractor:
    """Docling 결과에서 테이블을 추출하는 클래스.

    셀 단위 bbox, row/col indices 보존.
    """

    def extract(self, result: Any, table_extraction: bool) -> List[Dict[str, Any]]:
        """테이블 추출 (bbox, row/col indices 포함)."""
        if not table_extraction:
            return []

        tables_out: List[Dict[str, Any]] = []

        try:
            doc = result.document

            # Method 1: doc.tables
            if hasattr(doc, "tables"):
                for idx, table_item in enumerate(doc.tables, start=1):
                    table_dict = self._convert_table_item(table_item, idx)
                    if table_dict:
                        tables_out.append(table_dict)

            # Method 2: iterate pages for tables
            if not tables_out and hasattr(doc, "pages"):
                table_idx = 0
                for page_num, page in enumerate(doc.pages, start=1):
                    page_tables = getattr(page, "tables", None)
                    if isinstance(page_tables, list):
                        for table_item in page_tables:
                            table_idx += 1
                            table_dict = self._convert_table_item(table_item, table_idx, page_num)
                            if table_dict:
                                tables_out.append(table_dict)

        except Exception as e:
            logger.warning(f"Failed to extract tables: {e}")

        return tables_out

    def _convert_table_item(
        self, table_item: Any, idx: int, page_num: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Docling 테이블을 YAML 스키마로 변환."""
        cells_out: List[Dict[str, Any]] = []
        num_rows = 0
        num_cols = 0

        try:
            data = getattr(table_item, "data", None)
            if data is None:
                return None

            num_rows = getattr(data, "num_rows", 0)
            num_cols = getattr(data, "num_cols", 0)

            table_cells = getattr(data, "table_cells", None)
            if not isinstance(table_cells, list):
                return None

            for cell in table_cells:
                cell_dict = self._convert_cell(cell)
                if cell_dict:
                    cells_out.append(cell_dict)

            if page_num is None:
                prov = getattr(table_item, "prov", None)
                if isinstance(prov, list) and prov:
                    first_prov = prov[0]
                    page_num = getattr(first_prov, "page_no", None) or 1

        except Exception as e:
            logger.warning(f"Failed to convert table {idx}: {e}")
            return None

        if not cells_out:
            return None

        if num_rows == 0 or num_cols == 0:
            if cells_out:
                num_rows = max(int(c.get("row", 0)) for c in cells_out) + 1
                num_cols = max(int(c.get("col", 0)) for c in cells_out) + 1

        return {
            "table_id": f"table_{idx}",
            "page": page_num or 1,
            "shape": {"rows": num_rows, "cols": num_cols},
            "cells": cells_out,
        }

    def _convert_cell(self, cell: Any) -> Optional[Dict[str, Any]]:
        """단일 셀을 YAML 스키마로 변환."""
        try:
            row = getattr(cell, "start_row_offset_idx", None)
            col = getattr(cell, "start_col_offset_idx", None)

            if row is None:
                row = getattr(cell, "row", None)
            if col is None:
                col = getattr(cell, "col", None)

            if row is None or col is None:
                return None

            text = getattr(cell, "text", "")
            if text is None:
                text = ""
            text = str(text).strip()

            bbox_list = None
            bbox = getattr(cell, "bbox", None)
            if bbox is not None:
                if hasattr(bbox, "l"):
                    bbox_list = [
                        float(bbox.l),
                        float(bbox.t),
                        float(bbox.r),
                        float(bbox.b),
                    ]
                elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    bbox_list = [float(x) for x in bbox]

            row_span = getattr(cell, "row_span", 1) or 1
            col_span = getattr(cell, "col_span", 1) or 1

            end_row = getattr(cell, "end_row_offset_idx", row)
            end_col = getattr(cell, "end_col_offset_idx", col)

            is_row_header = bool(getattr(cell, "row_header", False))
            is_col_header = bool(getattr(cell, "column_header", False))

            return {
                "row": int(row),
                "col": int(col),
                "text": text,
                "bbox": bbox_list,
                "row_span": int(row_span),
                "col_span": int(col_span),
                "end_row": int(end_row) if end_row is not None else int(row),
                "end_col": int(end_col) if end_col is not None else int(col),
                "is_header": is_row_header or is_col_header,
                "ocr_text": None,
                "ocr_confidence": None,
                "reliability": None,
            }

        except Exception as e:
            logger.debug(f"Failed to convert cell: {e}")
            return None


class ImageExtractor:
    """Docling 결과에서 이미지 에셋을 추출하는 클래스."""

    def extract(self, result: Any) -> List[Dict[str, Any]]:
        """이미지 에셋 추출."""
        images: List[Dict[str, Any]] = []

        try:
            doc = result.document

            if hasattr(doc, "pages"):
                img_idx = 0
                for page_num, page in enumerate(doc.pages, start=1):
                    page_images = getattr(page, "images", None)
                    if isinstance(page_images, list):
                        for img in page_images:
                            img_idx += 1
                            images.append(
                                {
                                    "image_id": f"img_{img_idx}",
                                    "page": page_num,
                                    "ocr_text": None,
                                }
                            )

        except Exception as e:
            logger.debug(f"Failed to extract images: {e}")

        return images


class DoclingYAMLAdapter:
    """PDF -> YAML canonical dict adapter.

    핵심 원칙:
    - Markdown을 중간 표현으로 사용하지 않음
    - Docling의 구조화된 출력을 직접 YAML로 변환
    - 셀 단위 bbox, OCR confidence 보존
    - 품질 게이트 적용

    Output schema:
        {
            "document": {...},
            "content": {"paragraphs": [...]},
            "tables": [...],
            "assets": {"images": [...]}
        }
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        table_extraction: bool = True,
        ocr_engine: str = "paddle",
        overwrite_empty_tables_with_ocr: bool = True,
    ) -> None:
        """Initialize adapter.

        Args:
            ocr_enabled: Enable OCR for scanned PDFs
            table_extraction: Enable table structure extraction
            ocr_engine: OCR engine to use (only 'paddle' supported)
            overwrite_empty_tables_with_ocr: Fill empty table cells with OCR text
        """
        if ocr_engine != "paddle":
            raise ValueError("Only paddle OCR is supported in this pipeline.")

        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.ocr_engine = ocr_engine
        self.overwrite_empty_tables_with_ocr = overwrite_empty_tables_with_ocr
        self._converter = None
        # Removed threading lock to prevent deadlocks in core environment
        # self._converter_lock = threading.Lock()

    def _get_converter(self):
        """Lazy load Docling converter with proper GPU configuration."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.document_converter import PdfFormatOption
                from docling.datamodel.accelerator_options import (
                    AcceleratorOptions,
                    AcceleratorDevice,
                )

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = self.ocr_enabled
                pipeline_options.do_table_structure = self.table_extraction

                # GPU stability configuration (GPU-only architecture 유지)
                pipeline_options.accelerator_options = AcceleratorOptions(
                    num_threads=1,  # 단일 스레드로 GPU 컨텍스트 충돌 방지
                    device=AcceleratorDevice.CUDA,  # 명시적 GPU 사용
                )

                # Disable progress bar to prevent display issues
                if hasattr(pipeline_options, "enable_progress_bar"):
                    pipeline_options.enable_progress_bar = False

                # 배치 사이즈 축소로 GPU 메모리 사용량 감소
                if hasattr(pipeline_options, "ocr_batch_size"):
                    pipeline_options.ocr_batch_size = 1
                if hasattr(pipeline_options, "layout_batch_size"):
                    pipeline_options.layout_batch_size = 1

                self._converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
            except ImportError as e:
                logger.error(f"Failed to import docling: {e}")
                raise ImportError("docling is required. Install with: pip install docling")

        return self._converter

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF to YAML-serializable dict.

        Args:
            file_path: Path to PDF file

        Returns:
            YAML-serializable dict with document, content, tables, assets
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Parsing PDF: {file_path}")

        import os

        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        try:
            # Document size limits to prevent resource exhaustion (GPU stability)
            file_size = file_path.stat().st_size
            max_file_size = 50 * 1024 * 1024  # 50MB 제한

            if file_size > max_file_size:
                logger.warning(
                    f"File too large for stable GPU processing: {file_path.name} ({file_size / 1024 / 1024:.1f}MB > 50MB)"
                )
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB. Max allowed: 50MB")

            # GPU 메모리 정리 (기존 컨텍스트 해제)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            docling_converter = self._get_converter()
            docling_result = docling_converter.convert(
                str(file_path),
                max_num_pages=100,  # 페이지 수 제한
                max_file_size=max_file_size,
            )

            # 하이브리드: pypdfium2(텍스트) + Docling(구조)
            paragraphs = self._extract_paragraphs(file_path, docling_result)

            # TableExtractor 사용
            table_extractor = TableExtractor()
            tables = table_extractor.extract(docling_result, self.table_extraction)

            # Apply quality assessment
            for t in tables:
                attach_cell_reliability(t)
                attach_table_quality(t)

            page_count = getattr(docling_result.document, "pages", None)
            page_count = len(page_count) if page_count else 1

            # ImageExtractor 사용
            image_extractor = ImageExtractor()

            doc = {
                "document": {
                    "source_path": str(file_path),
                    "format": "pdf",
                    "encrypted": False,
                    "parser": "docling",
                    "text_extractor": "pypdfium2" if PYPDFIUM2_AVAILABLE else "docling",
                    "page_count": page_count,
                    "ocr_enabled": self.ocr_enabled,
                    "table_extraction": self.table_extraction,
                },
                "content": {"paragraphs": paragraphs},
                "tables": tables,
                "assets": {
                    "images": image_extractor.extract(docling_result),
                },
            }

            logger.info(
                f"Parsed {file_path.name}: "
                f"{len(paragraphs)} paragraphs, "
                f"{len(tables)} tables, "
                f"{page_count} pages"
            )

            return doc

        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise ValueError(f"PDF parsing failed: {e}")
        finally:
            # GPU memory cleanup for stability
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # Cleanup failures are non-critical

    def _extract_paragraphs_pypdfium2(self, file_path: Path) -> List[str]:
        """Extract paragraphs using pypdfium2 (BSD-3-Clause)."""
        return ParagraphExtractor(file_path).extract()

    def _extract_paragraphs_docling(self, result: Any) -> List[str]:
        """Extract paragraphs from Docling result (fallback).

        Does NOT use markdown - extracts from structured output directly.
        """
        paragraphs: List[str] = []

        try:
            doc = result.document

            # Method 1: main_text blocks
            if hasattr(doc, "main_text"):
                for item in doc.main_text:
                    if hasattr(item, "text") and item.text:
                        text = str(item.text).strip()
                        if text:
                            paragraphs.append(text)

            # Method 2: pages -> text_blocks
            if not paragraphs and hasattr(doc, "pages"):
                for page in doc.pages:
                    blocks = getattr(page, "text_blocks", None) or getattr(page, "blocks", None)
                    if isinstance(blocks, list):
                        for block in blocks:
                            text = getattr(block, "text", None)
                            if isinstance(text, str) and text.strip():
                                paragraphs.append(text.strip())

            # Method 3: body -> items
            if not paragraphs and hasattr(doc, "body"):
                body = doc.body
                if hasattr(body, "items"):
                    for item in body.items:
                        text = getattr(item, "text", None)
                        if isinstance(text, str) and text.strip():
                            paragraphs.append(text.strip())

        except Exception as e:
            logger.warning(f"Failed to extract paragraphs from Docling: {e}")

        return paragraphs

    def _extract_paragraphs(self, file_path: Path, docling_result: Any) -> List[str]:
        """Extract paragraphs - pypdfium2 primary, Docling fallback.

        Args:
            file_path: PDF 파일 경로
            docling_result: Docling 변환 결과 (fallback용)

        Returns:
            문단 리스트
        """
        # Primary: pypdfium2 (더 정확한 텍스트 추출)
        if PYPDFIUM2_AVAILABLE:
            paragraphs = self._extract_paragraphs_pypdfium2(file_path)
            if paragraphs:
                return paragraphs

        # Fallback: Docling 구조에서 추출
        logger.info("Using Docling fallback for paragraph extraction")
        return self._extract_paragraphs_docling(docling_result)


__all__ = [
    "DoclingYAMLAdapter",
    "ParagraphExtractor",
    "TableExtractor",
    "ImageExtractor",
]
