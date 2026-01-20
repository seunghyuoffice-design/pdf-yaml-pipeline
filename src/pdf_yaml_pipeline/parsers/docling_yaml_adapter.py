"""Docling-based PDF to YAML adapter.

PDF를 구조화된 YAML로 직접 변환 (Markdown 거치지 않음).
셀 단위 bbox, OCR confidence 보존.

하이브리드 모드:
- pypdfium2: 텍스트 추출 (paragraphs)
- Docling: 구조 추출 (tables, layout)
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PDF normalization and chunking (MIT license)
try:
    import pikepdf
    PIKEPDF_AVAILABLE = True
except ImportError:
    PIKEPDF_AVAILABLE = False
    pikepdf = None

# Chunking constants
CHUNK_SIZE = 100  # 100페이지 단위로 분할
PAGE_THRESHOLD = 100  # 100페이지 초과 시 분할

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

from src.pipeline.quality.table_quality import (  # noqa: E402
    attach_cell_reliability,
    attach_table_quality,
)


@dataclass(frozen=True)
class ChunkInfo:
    """청크 분할 메타데이터 (무결성 검증용)."""

    chunk_index: int           # 0, 1, 2, ...
    start_page: int            # 원본 기준 시작 페이지 (0-indexed)
    end_page: int              # 원본 기준 끝 페이지 (exclusive)
    total_chunks: int          # 전체 청크 수
    original_page_count: int   # 원본 총 페이지 수


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
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption

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

    def _normalize_pdf(self, file_path: Path) -> Tuple[Path, int, bool]:
        """pikepdf로 PDF 정규화 및 페이지 수 확인.

        Args:
            file_path: 원본 PDF 경로

        Returns:
            (정규화된 PDF 경로, 페이지 수, 임시 파일 여부)
        """
        if not PIKEPDF_AVAILABLE:
            logger.warning("pikepdf not available, skipping normalization")
            # pypdfium2로 페이지 수만 확인
            if PYPDFIUM2_AVAILABLE:
                try:
                    pdf = pdfium.PdfDocument(str(file_path))
                    page_count = len(pdf)
                    pdf.close()
                    return file_path, page_count, False
                except Exception as e:
                    logger.warning(f"Failed to get page count: {e}")
                    return file_path, 0, False
            return file_path, 0, False

        try:
            with pikepdf.open(file_path) as pdf:
                page_count = len(pdf.pages)

                # 정규화 필요 여부 판단 (항상 정규화하여 호환성 보장)
                normalized_path = Path(tempfile.mktemp(suffix=".pdf"))
                pdf.save(normalized_path, linearize=True)
                logger.debug(f"Normalized PDF: {file_path.name} -> {normalized_path.name} ({page_count} pages)")

                return normalized_path, page_count, True

        except Exception as e:
            logger.error(f"pikepdf normalization failed: {e}")
            raise ValueError(f"PDF normalization failed: {e}")

    def _split_pdf(self, normalized_path: Path, page_count: int) -> List[Tuple[ChunkInfo, Path]]:
        """대용량 PDF를 청크로 분할.

        Args:
            normalized_path: 정규화된 PDF 경로
            page_count: 총 페이지 수

        Returns:
            [(ChunkInfo, 청크 PDF 경로), ...]
        """
        if not PIKEPDF_AVAILABLE:
            raise RuntimeError("pikepdf required for PDF splitting")

        chunks: List[Tuple[ChunkInfo, Path]] = []
        total_chunks = (page_count + CHUNK_SIZE - 1) // CHUNK_SIZE  # ceiling division

        with pikepdf.open(normalized_path) as pdf:
            for chunk_idx in range(total_chunks):
                start_page = chunk_idx * CHUNK_SIZE
                end_page = min(start_page + CHUNK_SIZE, page_count)

                chunk_info = ChunkInfo(
                    chunk_index=chunk_idx,
                    start_page=start_page,
                    end_page=end_page,
                    total_chunks=total_chunks,
                    original_page_count=page_count,
                )

                # 청크 PDF 생성 (extend 사용 - append 루프는 segfault 발생)
                chunk_pdf = pikepdf.Pdf.new()
                chunk_pdf.pages.extend(pdf.pages[start_page:end_page])

                chunk_path = Path(tempfile.mktemp(suffix=f"_chunk{chunk_idx}.pdf"))
                chunk_pdf.save(chunk_path)
                chunk_pdf.close()

                chunks.append((chunk_info, chunk_path))
                logger.debug(f"Created chunk {chunk_idx + 1}/{total_chunks}: pages {start_page}-{end_page - 1}")

        return chunks

    def _parse_single_chunk(self, chunk_path: Path, chunk_info: ChunkInfo) -> Dict[str, Any]:
        """단일 청크를 docling으로 파싱.

        Args:
            chunk_path: 청크 PDF 경로
            chunk_info: 청크 메타데이터

        Returns:
            파싱 결과 dict
        """
        logger.info(f"Parsing chunk {chunk_info.chunk_index + 1}/{chunk_info.total_chunks} "
                   f"(pages {chunk_info.start_page}-{chunk_info.end_page - 1})")

        # GPU 메모리 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        docling_converter = self._get_converter()
        docling_result = docling_converter.convert(str(chunk_path))

        # 텍스트 추출
        paragraphs = self._extract_paragraphs(chunk_path, docling_result)

        # 테이블 추출
        table_extractor = TableExtractor()
        tables = table_extractor.extract(docling_result, self.table_extraction)

        # 품질 평가
        for t in tables:
            attach_cell_reliability(t)
            attach_table_quality(t)

        # 이미지 추출
        image_extractor = ImageExtractor()
        images = image_extractor.extract(docling_result)

        chunk_page_count = chunk_info.end_page - chunk_info.start_page

        return {
            "chunk_info": {
                "chunk_index": chunk_info.chunk_index,
                "start_page": chunk_info.start_page,
                "end_page": chunk_info.end_page,
                "total_chunks": chunk_info.total_chunks,
                "original_page_count": chunk_info.original_page_count,
            },
            "page_count": chunk_page_count,
            "paragraphs": paragraphs,
            "tables": tables,
            "images": images,
        }

    def _merge_chunk_results(
        self, file_path: Path, chunks_results: List[Tuple[ChunkInfo, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """청크 결과를 병합하고 무결성 검증.

        Args:
            file_path: 원본 PDF 경로
            chunks_results: [(ChunkInfo, 파싱 결과), ...]

        Returns:
            병합된 최종 결과

        Raises:
            ValueError: 무결성 검증 실패 시
        """
        if not chunks_results:
            raise ValueError("No chunk results to merge")

        # 정렬 (chunk_index 기준)
        chunks_results = sorted(chunks_results, key=lambda x: x[0].chunk_index)

        first_chunk_info = chunks_results[0][0]
        total_chunks = first_chunk_info.total_chunks
        original_page_count = first_chunk_info.original_page_count

        # === 무결성 검증 1: 모든 청크 존재 확인 ===
        chunk_indices = {c.chunk_index for c, _ in chunks_results}
        expected_indices = set(range(total_chunks))
        if chunk_indices != expected_indices:
            missing = expected_indices - chunk_indices
            raise ValueError(f"청크 누락: {missing}")

        # === 무결성 검증 2: 페이지 연속성 확인 ===
        for i in range(1, len(chunks_results)):
            prev_end = chunks_results[i - 1][0].end_page
            curr_start = chunks_results[i][0].start_page
            if prev_end != curr_start:
                raise ValueError(f"페이지 갭 발생: chunk {i - 1} end={prev_end}, chunk {i} start={curr_start}")

        # === 무결성 검증 3: 총 페이지 수 일치 ===
        total_pages = sum(c.end_page - c.start_page for c, _ in chunks_results)
        if total_pages != original_page_count:
            raise ValueError(f"페이지 수 불일치: merged={total_pages}, original={original_page_count}")

        # === 병합 ===
        merged_paragraphs: List[str] = []
        merged_tables: List[Dict[str, Any]] = []
        merged_images: List[Dict[str, Any]] = []
        table_id_counter = 0
        image_id_counter = 0

        for chunk_info, result in chunks_results:
            page_offset = chunk_info.start_page

            # Paragraphs 연결
            merged_paragraphs.extend(result["paragraphs"])

            # Tables 연결 (페이지 번호 보정, ID 재부여)
            for table in result["tables"]:
                table_id_counter += 1
                table["table_id"] = f"table_{table_id_counter}"
                original_page = table.get("page", 1)
                table["page"] = original_page + page_offset  # 원본 기준 절대 페이지
                table["original_page"] = original_page + page_offset
                table["chunk_index"] = chunk_info.chunk_index
                merged_tables.append(table)

            # Images 연결 (페이지 번호 보정, ID 재부여)
            for image in result["images"]:
                image_id_counter += 1
                image["image_id"] = f"img_{image_id_counter}"
                original_page = image.get("page", 1)
                image["page"] = original_page + page_offset
                image["chunk_index"] = chunk_info.chunk_index
                merged_images.append(image)

        logger.info(
            f"Merged {len(chunks_results)} chunks: "
            f"{len(merged_paragraphs)} paragraphs, "
            f"{len(merged_tables)} tables, "
            f"{len(merged_images)} images"
        )

        return {
            "document": {
                "source_path": str(file_path),
                "format": "pdf",
                "encrypted": False,
                "parser": "docling",
                "text_extractor": "pypdfium2" if PYPDFIUM2_AVAILABLE else "docling",
                "page_count": original_page_count,
                "ocr_enabled": self.ocr_enabled,
                "table_extraction": self.table_extraction,
                "chunked": True,
                "chunk_count": total_chunks,
                "chunk_size": CHUNK_SIZE,
            },
            "content": {"paragraphs": merged_paragraphs},
            "tables": merged_tables,
            "assets": {"images": merged_images},
        }

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

        os.environ.get("CUDA_VISIBLE_DEVICES", "")

        try:
            # Pre-validation for common PDF issues
            from .pdf_validator import validate_pdf_integrity, is_pdf_corrupted

            is_valid, validation_error = validate_pdf_integrity(str(file_path))
            if not is_valid:
                logger.error(f"PDF validation failed for {file_path.name}: {validation_error}")
                raise ValueError(f"PDF validation failed: {validation_error}")

            is_corrupted, corruption_reason = is_pdf_corrupted(str(file_path))
            if is_corrupted:
                logger.error(f"PDF corruption detected in {file_path.name}: {corruption_reason}")
                raise ValueError(f"PDF corruption detected: {corruption_reason}")

            # Document size limits to prevent resource exhaustion (GPU stability)
            file_size = file_path.stat().st_size
            max_file_size = 50 * 1024 * 1024  # 50MB 제한

            # 파일 크기 제한 완화 (청크 분할로 대용량 처리 가능)
            # 단, 극단적으로 큰 파일은 여전히 제한 (200MB)
            extreme_max_size = 200 * 1024 * 1024
            if file_size > extreme_max_size:
                logger.warning(
                    f"File extremely large: {file_path.name} ({file_size / 1024 / 1024:.1f}MB > 200MB)"
                )
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB. Max allowed: 200MB")

            # === Step 1: pikepdf로 정규화 + 페이지 수 확인 ===
            normalized_path, page_count, is_temp = self._normalize_pdf(file_path)
            logger.info(f"Normalized PDF: {file_path.name} ({page_count} pages)")

            temp_files: List[Path] = []
            if is_temp:
                temp_files.append(normalized_path)

            try:
                # === Step 2: 페이지 수에 따라 분기 ===
                if page_count > PAGE_THRESHOLD:
                    # 대용량 PDF: 스트리밍 청크 처리 (메모리 효율)
                    logger.info(f"Large PDF detected ({page_count} pages > {PAGE_THRESHOLD}), streaming chunks")

                    total_chunks = (page_count + CHUNK_SIZE - 1) // CHUNK_SIZE
                    chunks_results: List[Tuple[ChunkInfo, Dict[str, Any]]] = []

                    # 청크 하나씩 생성→파싱→삭제 (스트리밍)
                    for chunk_idx in range(total_chunks):
                        start_page = chunk_idx * CHUNK_SIZE
                        end_page = min(start_page + CHUNK_SIZE, page_count)

                        chunk_info = ChunkInfo(
                            chunk_index=chunk_idx,
                            start_page=start_page,
                            end_page=end_page,
                            total_chunks=total_chunks,
                            original_page_count=page_count,
                        )

                        # 청크 PDF 생성
                        chunk_path = Path(tempfile.mktemp(suffix=f"_chunk{chunk_idx}.pdf"))
                        try:
                            with pikepdf.open(normalized_path) as pdf:
                                chunk_pdf = pikepdf.Pdf.new()
                                chunk_pdf.pages.extend(pdf.pages[start_page:end_page])
                                chunk_pdf.save(chunk_path)
                                chunk_pdf.close()
                            logger.debug(f"Created chunk {chunk_idx + 1}/{total_chunks}: pages {start_page}-{end_page - 1}")

                            # 즉시 파싱
                            result = self._parse_single_chunk(chunk_path, chunk_info)
                            chunks_results.append((chunk_info, result))

                        except Exception as e:
                            logger.error(f"Chunk {chunk_info.chunk_index} failed: {e}")
                            raise ValueError(
                                f"Chunk {chunk_info.chunk_index + 1}/{chunk_info.total_chunks} "
                                f"(pages {chunk_info.start_page}-{chunk_info.end_page - 1}) failed: {e}"
                            )
                        finally:
                            # 청크 파일 즉시 삭제
                            try:
                                if chunk_path.exists():
                                    chunk_path.unlink()
                            except Exception:
                                pass

                    # 결과 병합 + 무결성 검증
                    return self._merge_chunk_results(file_path, chunks_results)

                else:
                    # 소형 PDF: 기존 방식 (단일 파싱)
                    # GPU 메모리 정리
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    logger.debug(f"Starting docling conversion for {file_path.name}")
                    docling_converter = self._get_converter()
                    docling_result = docling_converter.convert(str(normalized_path))
                    logger.debug(f"Docling conversion completed for {file_path.name}")

                    # 하이브리드: pypdfium2(텍스트) + Docling(구조)
                    paragraphs = self._extract_paragraphs(normalized_path, docling_result)

                    # TableExtractor 사용
                    table_extractor = TableExtractor()
                    tables = table_extractor.extract(docling_result, self.table_extraction)

                    # Apply quality assessment
                    for t in tables:
                        attach_cell_reliability(t)
                        attach_table_quality(t)

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
                            "chunked": False,
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

            finally:
                # 임시 파일 정리
                for temp_file in temp_files:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass  # 정리 실패는 무시

        except ImportError:
            raise
        except Exception as e:
            # Enhanced error classification for better fallback handling
            error_str = str(e).lower()
            error_type = "unknown"

            if "input document" in error_str and "not valid" in error_str:
                error_type = "invalid_pdf"
                logger.error(f"PDF validation error in pypdfium2/docling: {file_path.name} - {e}")
            elif any(keyword in error_str for keyword in ["corrupt", "damaged", "invalid"]):
                error_type = "corrupted_pdf"
                logger.error(f"PDF corruption detected: {file_path.name} - {e}")
            elif any(keyword in error_str for keyword in ["encrypted", "password", "permission"]):
                error_type = "encrypted_pdf"
                logger.error(f"PDF encryption issue: {file_path.name} - {e}")
            elif any(keyword in error_str for keyword in ["memory", "out of memory", "oom"]):
                error_type = "memory_error"
                logger.error(f"Memory exhaustion during PDF parsing: {file_path.name} - {e}")
            elif any(keyword in error_str for keyword in ["version", "unsupported format"]):
                error_type = "unsupported_version"
                logger.error(f"Unsupported PDF version/format: {file_path.name} - {e}")
            elif any(keyword in error_str for keyword in ["timeout", "time"]):
                error_type = "timeout"
                logger.error(f"Parsing timeout: {file_path.name} - {e}")
            else:
                logger.error(f"Unexpected PDF parsing error: {file_path.name} - {e}")

            # Re-raise with error type information
            raise ValueError(f"PDF parsing failed ({error_type}): {e}") from e
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
