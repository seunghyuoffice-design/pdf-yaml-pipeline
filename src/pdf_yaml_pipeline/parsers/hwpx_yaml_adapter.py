"""HWPX to YAML adapter.

HWPX 파일을 구조화된 YAML로 직접 변환.
레거시 파서 의존 없이 zipfile + XML로 직접 파싱.

HWPX 구조:
- ZIP 압축 파일
- Contents/SECTION_*/BodyText.xml: 섹션별 본문
- <Para><Text>: 문단
- <Table><Row><Cell>: 테이블
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from pdf_yaml_pipeline.quality.table_quality import (
    attach_cell_reliability,
    attach_table_quality,
)


class HWPXYAMLAdapter:
    """HWPX -> YAML canonical dict adapter.

    직접 HWPX ZIP 파일을 파싱하여 YAML 스키마로 변환.
    Markdown 중간 변환 없이 구조 보존.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        table_extraction: bool = True,
        ocr_engine: str = "paddle",
        overwrite_empty_tables_with_ocr: bool = True,
        **_: Any,
    ) -> None:
        if ocr_engine != "paddle":
            raise ValueError("Only paddle OCR is supported.")

        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.ocr_engine = ocr_engine
        self.overwrite_empty_tables_with_ocr = overwrite_empty_tables_with_ocr

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse HWPX file to YAML-serializable dict.

        Args:
            file_path: Path to HWPX file

        Returns:
            YAML-serializable dict with document, content, tables, assets
        """
        if not file_path.exists():
            raise FileNotFoundError(f"HWPX file not found: {file_path}")

        logger.info(f"Parsing HWPX: {file_path}")

        # 직접 HWPX 파싱
        paragraphs, tables_raw = self._parse_hwpx_direct(file_path)

        # 테이블을 YAML 스키마로 변환
        tables_yaml: List[Dict[str, Any]] = []
        if self.table_extraction:
            for i, table_raw in enumerate(tables_raw, start=1):
                table_yaml = self._table_to_yaml(table_raw, i)
                if table_yaml:
                    tables_yaml.append(table_yaml)

        # 품질 평가 적용
        for t in tables_yaml:
            attach_cell_reliability(t)
            attach_table_quality(t)

        logger.info(f"Parsed {file_path.name}: " f"{len(paragraphs)} paragraphs, " f"{len(tables_yaml)} tables")

        return {
            "document": {
                "source_path": str(file_path),
                "format": "hwpx",
                "encrypted": False,
                "parser": "hwpx_direct",
            },
            "content": {"paragraphs": [str(p).strip() for p in paragraphs if str(p).strip()]},
            "tables": tables_yaml,
            "assets": {"images": []},
        }

    def _parse_hwpx_direct(self, file_path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
        """HWPX ZIP 파일을 직접 파싱.

        Args:
            file_path: HWPX 파일 경로

        Returns:
            (paragraphs, tables_raw) 튜플
        """
        paragraphs: List[str] = []
        tables_raw: List[Dict[str, Any]] = []

        try:
            with zipfile.ZipFile(file_path, "r") as hwpx_zip:
                file_list = hwpx_zip.namelist()

                # 섹션 파일들 찾기 (BodyText.xml 또는 section*.xml)
                section_files = []
                for filename in file_list:
                    if "BodyText.xml" in filename:
                        section_files.append(filename)
                    elif filename.startswith("Contents/section") and filename.endswith(".xml"):
                        section_files.append(filename)

                # 각 섹션 파싱
                for section_file in sorted(section_files):
                    try:
                        xml_data = hwpx_zip.read(section_file).decode("utf-8", errors="ignore")
                        root = ET.fromstring(xml_data)

                        # 문단 추출
                        section_paragraphs = self._extract_paragraphs_from_xml(root)
                        paragraphs.extend(section_paragraphs)

                        # 테이블 추출
                        if self.table_extraction:
                            section_tables = self._extract_tables_from_xml(root)
                            tables_raw.extend(section_tables)

                    except Exception as e:
                        logger.debug(f"Failed to parse section {section_file}: {e}")

                # content.hwpml에서도 추출 시도
                for filename in file_list:
                    if "content.hwpml" in filename:
                        try:
                            xml_data = hwpx_zip.read(filename).decode("utf-8", errors="ignore")
                            root = ET.fromstring(xml_data)
                            extra_paragraphs = self._extract_paragraphs_from_xml(root)
                            # 중복 제거
                            for p in extra_paragraphs:
                                if p not in paragraphs:
                                    paragraphs.append(p)
                        except Exception as e:
                            logger.debug(f"Failed to parse content.hwpml: {e}")

        except zipfile.BadZipFile:
            logger.error(f"Invalid HWPX file (not a valid ZIP): {file_path}")
            raise ValueError(f"Invalid HWPX file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to parse HWPX: {e}")
            raise ValueError(f"HWPX parsing failed: {e}")

        return paragraphs, tables_raw

    def _extract_paragraphs_from_xml(self, root: ET.Element) -> List[str]:
        """XML에서 문단 추출.

        HWPX XML 구조:
        - <Para><Text>텍스트</Text></Para>
        - 또는 일반 텍스트 노드
        """
        paragraphs: List[str] = []

        # 방법 1: <Para><Text> 구조
        for para in root.iter():
            if para.tag.endswith("Para") or para.tag == "Para":
                para_text = ""
                for text_elem in para.iter():
                    if text_elem.tag.endswith("Text") or text_elem.tag == "Text":
                        if text_elem.text:
                            para_text += text_elem.text
                if para_text.strip():
                    paragraphs.append(para_text.strip())

        # 방법 2: 일반 텍스트 노드 (폴백)
        if not paragraphs:
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text = elem.text.strip()
                    if len(text) > 3 and text not in paragraphs:
                        paragraphs.append(text)
                if elem.tail and elem.tail.strip():
                    text = elem.tail.strip()
                    if len(text) > 3 and text not in paragraphs:
                        paragraphs.append(text)

        return paragraphs

    def _extract_tables_from_xml(self, root: ET.Element) -> List[Dict[str, Any]]:
        """XML에서 테이블 추출.

        HWPX XML 구조:
        - <Table><Row><Cell><Text>텍스트</Text></Cell></Row></Table>
        """
        tables: List[Dict[str, Any]] = []

        for table in root.iter():
            if not (table.tag.endswith("Table") or table.tag == "Table"):
                continue

            table_data: Dict[str, Any] = {
                "headers": [],
                "rows": [],
            }

            is_first_row = True
            for row in table.iter():
                if not (row.tag.endswith("Row") or row.tag == "Row"):
                    continue

                row_data: List[str] = []
                for cell in row.iter():
                    if not (cell.tag.endswith("Cell") or cell.tag == "Cell"):
                        continue

                    cell_text = ""
                    for text_elem in cell.iter():
                        if text_elem.tag.endswith("Text") or text_elem.tag == "Text":
                            if text_elem.text:
                                cell_text += text_elem.text
                    row_data.append(cell_text.strip())

                if row_data:
                    if is_first_row:
                        table_data["headers"] = row_data
                        is_first_row = False
                    else:
                        table_data["rows"].append(row_data)

            # 유효한 테이블만 추가
            if table_data["headers"] or table_data["rows"]:
                tables.append(table_data)

        return tables

    def _table_to_yaml(self, table_raw: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """원시 테이블 데이터를 YAML 스키마로 변환.

        Args:
            table_raw: {"headers": [...], "rows": [[...]]}
            idx: 테이블 인덱스

        Returns:
            YAML 스키마 테이블 또는 None
        """
        headers = table_raw.get("headers", [])
        rows = table_raw.get("rows", [])

        # 그리드 구성
        grid: List[List[str]] = []

        if headers:
            grid.append([str(h or "").strip() for h in headers])

        for row in rows:
            if isinstance(row, list):
                grid.append([str(x or "").strip() for x in row])

        n_rows = len(grid)
        n_cols = max((len(r) for r in grid), default=0)

        if n_rows == 0 or n_cols == 0:
            return None

        # 행 길이 정규화
        for r in range(n_rows):
            if len(grid[r]) < n_cols:
                grid[r] = grid[r] + [""] * (n_cols - len(grid[r]))

        # cells 스키마로 변환
        cells: List[Dict[str, Any]] = []
        for r in range(n_rows):
            for c in range(n_cols):
                cells.append(
                    {
                        "row": r,
                        "col": c,
                        "text": grid[r][c],
                        "bbox": None,
                        "coord_origin": None,
                        "row_span": 1,
                        "col_span": 1,
                        "is_header": (r == 0),
                        "ocr_text": None,
                        "ocr_confidence": None,
                        "reliability": None,
                    }
                )

        return {
            "table_id": f"table_{idx}",
            "page": 1,
            "shape": {"rows": n_rows, "cols": n_cols},
            "cells": cells,
        }


__all__ = ["HWPXYAMLAdapter"]
