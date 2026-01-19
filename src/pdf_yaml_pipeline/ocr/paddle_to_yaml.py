# SPDX-License-Identifier: MIT
"""PaddleOCR → YAML 정규화 레이어.

PaddleOCR 원시 출력을 Dyarchy YAML 스키마로 직접 변환.
Docling wrapper 없이 독립적 사용 가능.

핵심 원칙:
1. bbox 좌표: image_px [l, t, r, b] (렌더 결과 픽셀 기준)
2. 렌더링 스케일: scale=4 고정 (텍스트/표 bbox 안정성)
3. 셀 병합: rowspan/colspan 풀지 않음 (여러 셀로 복제)
4. OCR 텍스트: 절대 수정하지 않음 (후처리는 상위 레이어)
5. 이미지 파일 저장 금지 (런타임 메모리에서만)

Example:
    >>> from src.pipeline.ocr.paddle_to_yaml import PaddleToYAML
    >>> converter = PaddleToYAML()
    >>> yaml_dict = converter.convert(Path("sample.pdf"))
    >>> with open("output.yaml", "w") as f:
    ...     yaml.safe_dump(yaml_dict, f, allow_unicode=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# PaddleOCR lazy import (Docker 환경 의존)
_OCR_ENGINE = None
_OCR_AVAILABLE: Optional[bool] = None


def _get_ocr_engine():
    """Lazy load PaddleOCR engine."""
    global _OCR_ENGINE, _OCR_AVAILABLE

    if _OCR_AVAILABLE is False:
        return None

    if _OCR_ENGINE is None:
        try:
            from paddleocr import PaddleOCR

            _OCR_ENGINE = PaddleOCR(
                lang="korean",
                use_angle_cls=True,
                show_log=False,
                use_gpu=True,
            )
            _OCR_AVAILABLE = True
            logger.info("PaddleOCR engine initialized with GPU")
        except ImportError:
            logger.warning("PaddleOCR not available")
            _OCR_AVAILABLE = False
            return None
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {e}")
            _OCR_AVAILABLE = False
            return None

    return _OCR_ENGINE


# =============================================================================
# Data Classes (YAML Schema)
# =============================================================================


@dataclass
class PaddleOCRResult:
    """PaddleOCR 원시 결과.

    Attributes:
        text: 인식된 텍스트 (원본 그대로)
        confidence: 신뢰도 (0.0-1.0)
        bbox: [l, t, r, b] 좌표
        polygon: 원본 4점 좌표 (선택적)
    """

    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (l, t, r, b)
    polygon: Optional[List[List[float]]] = None


@dataclass
class OCRCell:
    """테이블 셀 데이터.

    Attributes:
        row: 행 인덱스 (0-based)
        col: 열 인덱스 (0-based)
        bbox: [l, t, r, b] image_px 좌표 (렌더 픽셀 기준)
        text: OCR 추출 텍스트 (수정 불가)
        confidence: OCR 신뢰도 (0.0-1.0)
    """

    row: int
    col: int
    bbox: List[float]  # [l, t, r, b]
    text: str
    confidence: float


@dataclass
class OCRTable:
    """테이블 구조 데이터.

    Attributes:
        table_id: 테이블 식별자
        page: 페이지 번호
        bbox: 테이블 전체 bbox [l, t, r, b]
        coord_system: 좌표계 ("image_px")
        title: 테이블 제목 (OCR 기반 추론)
        shape: {"rows": int, "cols": int}
        quality: {"passed": bool, "confidence": float}
        cells: 셀 목록
    """

    table_id: str
    page: int
    bbox: List[float]
    coord_system: str = "image_px"
    title: Optional[str] = None
    shape: Dict[str, int] = field(default_factory=lambda: {"rows": 0, "cols": 0})
    quality: Dict[str, Any] = field(default_factory=lambda: {"passed": False, "confidence": 0.0})
    cells: List[OCRCell] = field(default_factory=list)


# =============================================================================
# Helper Functions
# =============================================================================


def normalize_bbox(bbox: List[float]) -> List[float]:
    """bbox를 정규화 (변환 없이 float 리스트로만 변환).

    NOTE: 좌표 변환이나 정규화(0~1)는 하지 않음.
    """
    return [float(x) for x in bbox]


# =============================================================================
# Main Converter Class
# =============================================================================


class PaddleToYAML:
    """PaddleOCR → YAML 변환기.

    핵심 원칙:
    1. bbox 좌표: PDF points [l, t, r, b] 유지
    2. 셀 병합: rowspan/colspan 풀지 않음 (여러 셀로 복제)
    3. OCR 텍스트: 절대 수정하지 않음 (후처리는 상위 레이어)
    """

    # Y좌표 허용 오차 (같은 행으로 간주)
    ROW_TOLERANCE = 15.0

    # X좌표 허용 오차 (같은 열로 간주)
    COL_TOLERANCE = 20.0

    # 테이블 최소 셀 수
    MIN_TABLE_CELLS = 4

    def __init__(
        self,
        min_confidence: float = 0.5,
        table_detection: bool = True,
        row_tolerance: float = 15.0,
        col_tolerance: float = 20.0,
    ) -> None:
        """초기화.

        Args:
            min_confidence: 최소 신뢰도 임계값
            table_detection: 테이블 구조 감지 활성화
            row_tolerance: 행 그룹화 Y좌표 허용 오차
            col_tolerance: 열 그룹화 X좌표 허용 오차
        """
        self.min_confidence = min_confidence
        self.table_detection = table_detection
        self.ROW_TOLERANCE = row_tolerance
        self.COL_TOLERANCE = col_tolerance

        self._ocr = None

    def convert(
        self,
        pdf_path: Path,
        product_id: Optional[str] = None,
        version: str = "1.0",
    ) -> Dict[str, Any]:
        """PDF를 YAML dict로 변환.

        Args:
            pdf_path: PDF 파일 경로
            product_id: 상품 ID (기본: 파일명)
            version: 버전 문자열

        Returns:
            Dyarchy YAML 스키마 dict

        Raises:
            FileNotFoundError: 파일이 없을 때
            RuntimeError: OCR 또는 PDF 변환 실패 시
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # OCR 엔진 초기화
        self._ocr = _get_ocr_engine()
        if self._ocr is None:
            raise RuntimeError("PaddleOCR not available")

        # PDF를 페이지별 이미지로 변환 (pypdfium2 - BSD-3-Clause)
        # scale=4 고정 (텍스트/표 bbox 안정성 목적)
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(pdf_path))
            page_images = []
            for page in pdf:
                # scale=4 고정 (288 DPI 상당, bbox 안정성 최적)
                bitmap = page.render(scale=4)
                pil_image = bitmap.to_pil()
                page_images.append(pil_image)
            pdf.close()
        except ImportError:
            raise RuntimeError("pypdfium2 required: pip install pypdfium2")
        except Exception as e:
            raise RuntimeError(f"PDF conversion failed: {e}")

        logger.info(f"Converting PDF: {pdf_path.name} ({len(page_images)} pages)")

        pages_meta: List[Dict[str, Any]] = []
        all_tables: List[OCRTable] = []
        all_paragraphs: List[str] = []
        table_counter = 0

        for page_num, page_img in enumerate(page_images, start=1):
            width, height = page_img.size

            # 페이지 메타데이터
            pages_meta.append(
                {
                    "page": page_num,
                    "width": width,
                    "height": height,
                }
            )

            # OCR 실행
            ocr_results = self._run_ocr_on_page(page_img)

            logger.debug(f"Page {page_num}: {len(ocr_results)} OCR results")

            # 테이블 감지
            page_tables: List[OCRTable] = []
            if self.table_detection and ocr_results:
                page_tables = self._detect_tables(ocr_results, width, height)
                for t in page_tables:
                    table_counter += 1
                    t.table_id = f"table_{table_counter}"
                    t.page = page_num
                    all_tables.append(t)

            # 비테이블 텍스트 → 문단
            table_bboxes = [t.bbox for t in page_tables]
            for ocr in ocr_results:
                if not self._is_inside_any_table(ocr.bbox, table_bboxes):
                    if ocr.text.strip():
                        all_paragraphs.append(ocr.text)  # 원본 유지

        result = self._build_yaml_dict(
            pdf_path=pdf_path,
            pages=pages_meta,
            tables=all_tables,
            paragraphs=all_paragraphs,
            product_id=product_id,
            version=version,
        )

        logger.info(
            f"Converted {pdf_path.name}: "
            f"{len(pages_meta)} pages, "
            f"{len(all_tables)} tables, "
            f"{len(all_paragraphs)} paragraphs"
        )

        return result

    def convert_from_ocr_result(
        self,
        ocr_result: Dict[str, Any],
        source_path: str,
        product_id: str,
        version: str,
    ) -> Dict[str, Any]:
        """PaddleOCR 원시 결과를 YAML dict로 변환.

        이미 OCR 실행된 결과가 있을 때 사용.

        Args:
            ocr_result: PaddleOCR 원시 결과 dict
            source_path: 원본 파일 경로
            product_id: 상품 ID
            version: 버전 문자열

        Returns:
            Dyarchy YAML 스키마 dict
        """
        doc = {
            "document": {
                "source_path": source_path,
                "product_id": product_id,
                "version": version,
                "format": "pdf",
            },
            "pages": [],
            "tables": [],
            "content": {"paragraphs": []},
        }

        table_counter = 0

        for p in ocr_result.get("pages", []):
            page_no = p["page"]
            doc["pages"].append(
                {
                    "page": page_no,
                    "width": p.get("width"),
                    "height": p.get("height"),
                }
            )

            for t in p.get("tables", []):
                table_counter += 1
                doc["tables"].append(self._normalize_table(t, page_no, table_counter))

            for para in p.get("paragraphs", []):
                text = para.get("text", "")
                if text:
                    doc["content"]["paragraphs"].append(text)  # 원본 유지

        return doc

    def _run_ocr_on_page(self, page_image) -> List[PaddleOCRResult]:
        """페이지 이미지에 OCR 실행."""
        import numpy as np

        # PIL Image → numpy array
        img_array = np.array(page_image)

        # PaddleOCR 실행
        result = self._ocr.ocr(img_array, cls=True)

        ocr_results: List[PaddleOCRResult] = []

        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    polygon = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text, confidence = line[1]

                    if confidence < self.min_confidence:
                        continue

                    # 4점 → ltrb 변환
                    bbox = self._polygon_to_ltrb(polygon)

                    ocr_results.append(
                        PaddleOCRResult(
                            text=str(text),  # 원본 유지, 수정 안 함
                            confidence=float(confidence),
                            bbox=bbox,
                            polygon=polygon,
                        )
                    )

        return ocr_results

    def _polygon_to_ltrb(self, polygon: List[List[float]]) -> Tuple[float, float, float, float]:
        """4점 폴리곤을 [l, t, r, b]로 변환.

        Args:
            polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            (left, top, right, bottom)
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (
            float(min(xs)),  # left
            float(min(ys)),  # top
            float(max(xs)),  # right
            float(max(ys)),  # bottom
        )

    def _detect_tables(
        self,
        ocr_results: List[PaddleOCRResult],
        page_width: int,
        page_height: int,
    ) -> List[OCRTable]:
        """OCR 결과에서 테이블 구조 감지.

        휴리스틱 기반:
        1. 수평 정렬된 텍스트 그룹 탐지
        2. 수직 정렬 패턴 확인
        3. 그리드 구조 추론
        """
        if not ocr_results:
            return []

        # Y좌표 기반 행 그룹화
        rows = self._group_by_y_coordinate(ocr_results)

        if len(rows) < 2:
            return []  # 최소 2행 필요

        # 열 감지 (X좌표 패턴 분석)
        col_positions = self._detect_column_positions(rows)

        if len(col_positions) < 2:
            return []  # 최소 2열 필요

        # 테이블로 판단 가능한지 검증
        total_cells = sum(len(row) for row in rows)
        if total_cells < self.MIN_TABLE_CELLS:
            return []

        # 테이블 bbox 계산
        all_bboxes = [r.bbox for r in ocr_results]
        table_bbox = [
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        ]

        # 셀 생성
        cells: List[OCRCell] = []
        for row_idx, row_results in enumerate(rows):
            for ocr in row_results:
                col_idx = self._find_column_index(ocr.bbox, col_positions)
                cells.append(
                    OCRCell(
                        row=row_idx,
                        col=col_idx,
                        bbox=list(ocr.bbox),
                        text=ocr.text,  # 원본 유지
                        confidence=ocr.confidence,
                    )
                )

        # 품질 계산
        avg_conf = sum(c.confidence for c in cells) / max(1, len(cells))
        quality = {
            "passed": avg_conf >= 0.6 and len(cells) >= self.MIN_TABLE_CELLS,
            "confidence": round(avg_conf, 4),
        }

        return [
            OCRTable(
                table_id="table_1",  # 호출자가 재설정
                page=1,  # 호출자가 재설정
                bbox=table_bbox,
                coord_system="image_px",
                title=self._infer_table_title(rows),
                shape={"rows": len(rows), "cols": len(col_positions)},
                quality=quality,
                cells=cells,
            )
        ]

    def _group_by_y_coordinate(self, ocr_results: List[PaddleOCRResult]) -> List[List[PaddleOCRResult]]:
        """OCR 결과를 Y좌표 기준으로 행 그룹화.

        Returns:
            행별로 그룹화된 OCR 결과 리스트
        """
        if not ocr_results:
            return []

        # Y 중심점 기준 정렬
        sorted_results = sorted(ocr_results, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)

        rows: List[List[PaddleOCRResult]] = []
        current_row: List[PaddleOCRResult] = [sorted_results[0]]
        current_y = (sorted_results[0].bbox[1] + sorted_results[0].bbox[3]) / 2

        for ocr in sorted_results[1:]:
            y_center = (ocr.bbox[1] + ocr.bbox[3]) / 2

            if abs(y_center - current_y) <= self.ROW_TOLERANCE:
                # 같은 행
                current_row.append(ocr)
            else:
                # 새 행
                # 현재 행을 X좌표로 정렬하여 저장
                rows.append(sorted(current_row, key=lambda r: r.bbox[0]))
                current_row = [ocr]
                current_y = y_center

        # 마지막 행 저장
        if current_row:
            rows.append(sorted(current_row, key=lambda r: r.bbox[0]))

        return rows

    def _detect_column_positions(self, rows: List[List[PaddleOCRResult]]) -> List[float]:
        """행들에서 열 위치 감지.

        Returns:
            열 시작 X좌표 리스트 (오름차순)
        """
        if not rows:
            return []

        # 모든 셀의 X 시작점 수집
        x_positions: List[float] = []
        for row in rows:
            for ocr in row:
                x_positions.append(ocr.bbox[0])

        if not x_positions:
            return []

        # X좌표 클러스터링
        x_positions.sort()
        columns: List[float] = [x_positions[0]]

        for x in x_positions[1:]:
            # 기존 열과 충분히 떨어져 있으면 새 열
            if all(abs(x - col) > self.COL_TOLERANCE for col in columns):
                columns.append(x)

        return sorted(columns)

    def _find_column_index(self, bbox: Tuple[float, float, float, float], col_positions: List[float]) -> int:
        """셀의 열 인덱스 찾기.

        Args:
            bbox: 셀 bbox (l, t, r, b)
            col_positions: 열 시작 X좌표 리스트

        Returns:
            열 인덱스 (0-based)
        """
        x_start = bbox[0]

        for i, col_x in enumerate(col_positions):
            if abs(x_start - col_x) <= self.COL_TOLERANCE:
                return i

        # 가장 가까운 열 찾기
        min_dist = float("inf")
        min_idx = 0
        for i, col_x in enumerate(col_positions):
            dist = abs(x_start - col_x)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def _infer_table_title(self, rows: List[List[PaddleOCRResult]]) -> Optional[str]:
        """테이블 제목 추론 (첫 행이 헤더인 경우).

        Returns:
            추론된 제목 또는 None
        """
        if not rows or not rows[0]:
            return None

        # 첫 행의 첫 셀을 제목으로 사용 (단순 휴리스틱)
        first_cell = rows[0][0]
        text = first_cell.text.strip()

        # 제목으로 적합한지 간단히 검증
        if len(text) > 2 and len(text) < 100:
            return text

        return None

    def _is_inside_any_table(self, bbox: Tuple[float, float, float, float], table_bboxes: List[List[float]]) -> bool:
        """bbox가 테이블 영역 안에 있는지 확인."""
        for tb in table_bboxes:
            # 간단한 포함 검사 (중심점 기준)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                return True

        return False

    def _normalize_table(
        self,
        raw: Dict[str, Any],
        page: int,
        table_idx: int,
    ) -> Dict[str, Any]:
        """원시 테이블 데이터를 YAML 스키마로 정규화."""
        cells = []
        for c in raw.get("cells", []):
            cells.append(
                {
                    "row": int(c["row"]),
                    "col": int(c["col"]),
                    "bbox": normalize_bbox(c["bbox"]),
                    "text": c.get("text", ""),  # 원본 유지
                    "confidence": float(c.get("confidence", 0.0)),
                }
            )

        return {
            "table_id": f"table_{table_idx}",
            "page": page,
            "bbox": normalize_bbox(raw["bbox"]),
            "coord_system": "image_px",
            "title": raw.get("title", ""),
            "shape": {
                "rows": raw.get("rows", 0),
                "cols": raw.get("cols", 0),
            },
            "quality": {
                "passed": raw.get("confidence", 0) >= 0.6,
                "confidence": float(raw.get("confidence", 0)),
            },
            "cells": cells,
        }

    def _build_yaml_dict(
        self,
        pdf_path: Path,
        pages: List[Dict[str, Any]],
        tables: List[OCRTable],
        paragraphs: List[str],
        product_id: Optional[str] = None,
        version: str = "1.0",
    ) -> Dict[str, Any]:
        """최종 YAML dict 구성.

        Dyarchy 표준 스키마 준수.
        """
        # 테이블 dict 변환
        tables_list = []
        for t in tables:
            tables_list.append(
                {
                    "table_id": t.table_id,
                    "page": t.page,
                    "bbox": t.bbox,
                    "coord_system": t.coord_system,
                    "title": t.title,
                    "shape": t.shape,
                    "quality": t.quality,
                    "cells": [
                        {
                            "row": c.row,
                            "col": c.col,
                            "bbox": c.bbox,
                            "text": c.text,  # 원본 유지
                            "confidence": c.confidence,
                        }
                        for c in t.cells
                    ],
                }
            )

        return {
            "document": {
                "source_path": str(pdf_path),
                "product_id": product_id or pdf_path.stem,
                "version": version,
                "format": "pdf",
            },
            "pages": pages,
            "tables": tables_list,
            "content": {
                "paragraphs": paragraphs,
            },
        }


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """YAML 파일 저장 유틸리티.

    Args:
        path: 저장 경로
        data: YAML 데이터
    """
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


__all__ = [
    "PaddleToYAML",
    "PaddleOCRResult",
    "OCRTable",
    "OCRCell",
    "write_yaml",
]
