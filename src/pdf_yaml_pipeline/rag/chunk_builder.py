# SPDX-License-Identifier: MIT
"""RAG chunk builder with clause title tracking.

조항명(clause_title)과 계층 경로(heading_path)를 추적하여
RAG 청크에 컨텍스트 정보를 추가.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf_yaml_pipeline.rag.doc_classifier import DocumentRole, classify_document_role

# =============================================================================
# Clause Detection Patterns
# =============================================================================

# 조항 패턴: 제N조, 제N장, 제N절, 별표N 등
CLAUSE_PATTERNS = [
    # 제N조 (보험금 지급) 형태
    re.compile(r"^(제\s*\d+\s*조)\s*[\(\(]?([^\)\)]*?)[\)\)]?\s*$"),
    # 제N조 보험금 지급 형태 (괄호 없음)
    re.compile(r"^(제\s*\d+\s*조)\s+(.+)$"),
    # 제N장 형태
    re.compile(r"^(제\s*\d+\s*장)\s*[\(\(]?([^\)\)]*?)[\)\)]?\s*$"),
    re.compile(r"^(제\s*\d+\s*장)\s+(.+)$"),
    # 제N절 형태
    re.compile(r"^(제\s*\d+\s*절)\s*[\(\(]?([^\)\)]*?)[\)\)]?\s*$"),
    re.compile(r"^(제\s*\d+\s*절)\s+(.+)$"),
    # 별표N 형태
    re.compile(r"^(별표\s*\d+)\s*[\(\(]?([^\)\)]*?)[\)\)]?\s*$"),
    re.compile(r"^(별표\s*\d+)\s+(.+)$"),
]

# 상위 구조 (장, 편)
CHAPTER_PATTERN = re.compile(r"^제\s*\d+\s*장")
SECTION_PATTERN = re.compile(r"^제\s*\d+\s*절")
ARTICLE_PATTERN = re.compile(r"^제\s*\d+\s*조")
APPENDIX_PATTERN = re.compile(r"^별표\s*\d+")


# =============================================================================
# Clause Tracker
# =============================================================================


class ClauseTracker:
    """조항 컨텍스트 추적기.

    문서를 순차적으로 처리하면서 현재 조항 제목과
    계층 경로를 추적합니다.
    """

    def __init__(self) -> None:
        self._chapter: Optional[str] = None  # 현재 장
        self._section: Optional[str] = None  # 현재 절
        self._article: Optional[str] = None  # 현재 조
        self._appendix: Optional[str] = None  # 현재 별표

    def update(self, text: str) -> Tuple[Optional[str], List[str]]:
        """텍스트를 분석하여 조항 컨텍스트 업데이트.

        Args:
            text: 분석할 텍스트 (paragraph)

        Returns:
            (clause_title, heading_path) 튜플
            - clause_title: 현재 조항 제목 (가장 구체적인 것)
            - heading_path: 계층 경로 리스트
        """
        text = text.strip()

        # 조항 패턴 매칭
        for pattern in CLAUSE_PATTERNS:
            match = pattern.match(text)
            if match:
                prefix = match.group(1).replace(" ", "")  # 제1조, 제1장 등
                title = match.group(2).strip() if match.lastindex >= 2 else ""
                full_title = f"{prefix} ({title})" if title else prefix

                # 유형별 업데이트
                if CHAPTER_PATTERN.match(text):
                    self._chapter = full_title
                    self._section = None
                    self._article = None
                elif SECTION_PATTERN.match(text):
                    self._section = full_title
                    self._article = None
                elif ARTICLE_PATTERN.match(text):
                    self._article = full_title
                elif APPENDIX_PATTERN.match(text):
                    self._appendix = full_title
                    # 별표는 독립적이므로 장/절/조 초기화
                    self._chapter = None
                    self._section = None
                    self._article = None

                break

        return self._get_current_context()

    def _get_current_context(self) -> Tuple[Optional[str], List[str]]:
        """현재 컨텍스트 반환."""
        # heading_path 구성
        path: List[str] = []
        if self._chapter:
            path.append(self._chapter)
        if self._section:
            path.append(self._section)
        if self._article:
            path.append(self._article)
        if self._appendix:
            path.append(self._appendix)

        # clause_title: 가장 구체적인 것 (조 > 절 > 장 > 별표)
        clause_title = self._article or self._section or self._chapter or self._appendix

        return clause_title, path

    def get_current(self) -> Tuple[Optional[str], List[str]]:
        """현재 컨텍스트 조회 (업데이트 없이)."""
        return self._get_current_context()


# =============================================================================
# Helper Functions
# =============================================================================


def _cell_text(c: Dict[str, Any]) -> str:
    """셀 텍스트 추출."""
    return (c.get("text") or c.get("ocr_text") or "").strip()


def table_rows_to_sentences(table: Dict[str, Any], min_reliability: float = 0.6) -> List[str]:
    """테이블 행을 문장으로 변환."""
    shape = table.get("shape") or {}
    rows, cols = int(shape.get("rows", 0)), int(shape.get("cols", 0))
    cells = table.get("cells") or []
    if rows <= 1 or cols <= 0:
        return []

    grid = [["" for _ in range(cols)] for _ in range(rows)]
    rel = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for c in cells:
        r, k = int(c.get("row", -1)), int(c.get("col", -1))
        if 0 <= r < rows and 0 <= k < cols:
            grid[r][k] = _cell_text(c)
            rel[r][k] = float(c.get("reliability") or 0.0)

    headers = [h or f"col_{i}" for i, h in enumerate(grid[0])]
    out: List[str] = []

    for r in range(1, rows):
        parts = [
            f"{headers[c]}: {grid[r][c]}"
            for c in range(cols)
            if grid[r][c] and rel[r][c] >= min_reliability
        ]
        if parts:
            out.append(" | ".join(parts))
    return out


# =============================================================================
# Main Builder
# =============================================================================


def build_rag_chunks(doc: Dict[str, Any], min_table_reliability: float = 0.6) -> List[Dict[str, Any]]:
    """YAML 문서에서 RAG 청크 생성.

    Args:
        doc: YAML 파싱된 문서 dict

    Returns:
        RAG 청크 리스트. 각 청크는 다음 구조:
        {
            "text": str,
            "meta": {
                "source_path": str,
                "format": str,
                "role": str,
                "product_id": str,
                "version": str,
                "type": "paragraph" | "table_row",
                "idx": int,
                "page": int,
                "clause_title": str,      # 상위 조항 제목
                "heading_path": List[str] # 계층 경로
            }
        }
    """
    d = doc.get("document") or {}
    role = classify_document_role(d.get("source_path", ""))

    source_path = d.get("source_path") or ""
    product_id = d.get("product_id")
    if not product_id:
        path = Path(source_path)
        # Prefer parent directory (pipeline convention), but fall back safely for bare filenames.
        product_id = path.parent.name if path.parent and str(path.parent) not in (".", "") else path.stem

    meta_base = {
        "source_path": source_path,
        "format": d.get("format"),
        "role": role.value,
        "product_id": product_id,
        "version": d.get("version") or "unknown",
    }

    chunks: List[Dict[str, Any]] = []
    tracker = ClauseTracker()

    # paragraph별 테이블 매핑 (페이지 기반)
    tables_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for t in doc.get("tables") or []:
        page = t.get("page", 1)
        if page not in tables_by_page:
            tables_by_page[page] = []
        tables_by_page[page].append(t)

    # Paragraph 처리
    paragraphs = (doc.get("content") or {}).get("paragraphs") or []
    for i, p in enumerate(paragraphs):
        p = (p or "").strip()
        if not p:
            continue

        # 조항 컨텍스트 업데이트
        clause_title, heading_path = tracker.update(p)

        chunks.append(
            {
                "text": p,
                "meta": {
                    **meta_base,
                    "type": "paragraph",
                    "idx": i,
                    "clause_title": clause_title,
                    "heading_path": heading_path,
                },
            }
        )

    # Table 처리 (canonical 문서만)
    if role == DocumentRole.CANONICAL:
        # 현재 조항 컨텍스트를 테이블에 적용
        current_clause, current_path = tracker.get_current()

        for t in doc.get("tables") or []:
            if (t.get("quality") or {}).get("passed") is not True:
                continue

            # 테이블 제목이 있으면 조항명으로 사용
            table_title = t.get("title")
            if table_title and _is_clause_like(table_title):
                clause_for_table = table_title
            else:
                clause_for_table = current_clause

            for s in table_rows_to_sentences(t, min_reliability=min_table_reliability):
                chunks.append(
                    {
                        "text": s,
                        "meta": {
                            **meta_base,
                            "type": "table_row",
                            "table_id": t.get("table_id"),
                            "page": t.get("page"),
                            "clause_title": clause_for_table,
                            "heading_path": current_path,
                        },
                    }
                )

    return chunks


def _is_clause_like(text: str) -> bool:
    """텍스트가 조항 제목 형태인지 확인."""
    text = text.strip()
    for pattern in CLAUSE_PATTERNS:
        if pattern.match(text):
            return True
    return False


__all__ = [
    "build_rag_chunks",
    "table_rows_to_sentences",
    "ClauseTracker",
]
