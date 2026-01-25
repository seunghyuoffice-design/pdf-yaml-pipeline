"""Table structure validation module.

테이블 구조 검증 및 헤더 감지.
"""

from __future__ import annotations

from typing import Any

from .models import HeaderInfo, StructureValidation


class TableStructureValidator:
    """테이블 구조 검증.

    헤더 자동 감지, 병합 셀 처리, 구조적 일관성 검증.
    """

    # 헤더 감지 임계값
    HEADER_MIN_FILL_RATE = 0.5
    HEADER_MIN_AVG_RELIABILITY = 0.75
    HEADER_MAX_NUMERIC_RATIO = 0.3  # 숫자 비율 30% 이하

    def validate(self, table: dict[str, Any]) -> StructureValidation:
        """전체 구조 검증.

        Args:
            table: 테이블 딕셔너리

        Returns:
            StructureValidation: 검증 결과
        """
        cells = table.get("cells", []) or []

        if not cells:
            return StructureValidation(
                is_valid=False,
                header_info=None,
                has_merged_cells=False,
                issues=["empty_table"],
            )

        issues = []

        # 헤더 감지
        header_info = self.detect_header(cells)
        if not header_info.row_indices:
            issues.append("header_detection_failed")

        # 병합 셀 확인
        has_merged = self._has_merged_cells(cells)

        # 구조 일관성 검사
        structure_issues = self._check_structure_consistency(cells)
        issues.extend(structure_issues)

        is_valid = len(issues) == 0 or (
            len(issues) == 1 and issues[0] == "header_detection_failed" and header_info.confidence > 0.5
        )

        return StructureValidation(
            is_valid=is_valid,
            header_info=header_info,
            has_merged_cells=has_merged,
            issues=issues,
        )

    def detect_header(self, cells: list[dict[str, Any]]) -> HeaderInfo:
        """헤더 행 자동 감지.

        휴리스틱 기반으로 헤더 행을 감지한다:
        1. Row 0 우선 검사
        2. 숫자 비율이 낮은 행
        3. 텍스트 길이가 짧은 행
        4. 신뢰도가 높은 행

        Args:
            cells: 셀 리스트

        Returns:
            HeaderInfo: 헤더 정보
        """
        if not cells:
            return HeaderInfo(row_indices=[], confidence=0.0, is_multi_level=False)

        # 행별로 그룹화
        rows: dict[int, list[dict[str, Any]]] = {}
        for cell in cells:
            row_idx = int(cell.get("row", 0))
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(cell)

        if not rows:
            return HeaderInfo(row_indices=[], confidence=0.0, is_multi_level=False)

        # 각 행의 헤더 점수 계산
        header_scores: dict[int, float] = {}
        for row_idx, row_cells in sorted(rows.items()):
            score = self._calculate_header_score(row_cells)
            header_scores[row_idx] = score

        # Row 0 우선
        if 0 in header_scores and header_scores[0] >= 0.5:
            # 다중 헤더 확인 (Row 1도 헤더인지)
            is_multi = 1 in header_scores and header_scores[1] >= 0.6
            return HeaderInfo(
                row_indices=[0, 1] if is_multi else [0],
                confidence=header_scores[0],
                is_multi_level=is_multi,
            )

        # Row 0이 헤더가 아니면 가장 높은 점수 행
        if header_scores:
            best_row = max(header_scores, key=lambda x: header_scores[x])
            if header_scores[best_row] >= 0.5:
                return HeaderInfo(
                    row_indices=[best_row],
                    confidence=header_scores[best_row],
                    is_multi_level=False,
                )

        # 헤더 감지 실패 - Row 0을 기본값으로
        return HeaderInfo(
            row_indices=[0] if 0 in rows else [],
            confidence=0.3,
            is_multi_level=False,
        )

    def normalize_merged_cells(
        self,
        cells: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """병합 셀 정규화 (확장).

        colspan/rowspan이 있는 셀을 개별 셀로 확장한다.

        Args:
            cells: 원본 셀 리스트

        Returns:
            정규화된 셀 리스트
        """
        normalized = []

        for cell in cells:
            row = int(cell.get("row", 0))
            col = int(cell.get("col", 0))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # 병합 셀 확장
            for r in range(rowspan):
                for c in range(colspan):
                    new_cell = cell.copy()
                    new_cell["row"] = row + r
                    new_cell["col"] = col + c
                    new_cell["rowspan"] = 1
                    new_cell["colspan"] = 1

                    # 원본 셀 참조 (첫 번째 셀만)
                    if r == 0 and c == 0:
                        new_cell["is_merge_origin"] = True
                    else:
                        new_cell["is_merge_origin"] = False
                        new_cell["merge_origin"] = {"row": row, "col": col}

                    normalized.append(new_cell)

        return normalized

    def _calculate_header_score(self, row_cells: list[dict[str, Any]]) -> float:
        """단일 행의 헤더 점수 계산.

        Args:
            row_cells: 행의 셀 리스트

        Returns:
            헤더 점수 (0.0 ~ 1.0)
        """
        if not row_cells:
            return 0.0

        # 채움률
        filled = sum(1 for c in row_cells if (c.get("text") or "").strip() or (c.get("ocr_text") or "").strip())
        fill_rate = filled / len(row_cells)

        # 평균 신뢰도
        reliabilities = [float(c.get("reliability") or 0.0) for c in row_cells]
        avg_rel = sum(reliabilities) / len(reliabilities)

        # 숫자 비율 (낮을수록 헤더)
        numeric_count = 0
        for cell in row_cells:
            text = (cell.get("text") or cell.get("ocr_text") or "").strip()
            if text and self._is_numeric(text):
                numeric_count += 1
        numeric_ratio = numeric_count / max(1, filled)

        # 평균 텍스트 길이 (짧을수록 헤더)
        total_len = 0
        for cell in row_cells:
            text = (cell.get("text") or cell.get("ocr_text") or "").strip()
            total_len += len(text)
        avg_len = total_len / max(1, filled)
        len_score = 1.0 if avg_len < 20 else (0.5 if avg_len < 50 else 0.2)

        # 종합 점수
        score = fill_rate * 0.25 + avg_rel * 0.25 + (1 - numeric_ratio) * 0.30 + len_score * 0.20

        return score

    def _is_numeric(self, text: str) -> bool:
        """텍스트가 숫자인지 판별.

        Args:
            text: 텍스트

        Returns:
            숫자 여부
        """
        # 숫자, 쉼표, 소수점, 단위 제거 후 판별
        cleaned = text.replace(",", "").replace(".", "").replace("원", "")
        cleaned = cleaned.replace("만", "").replace("천", "").replace("억", "")
        cleaned = cleaned.replace("%", "").replace("세", "").replace("년", "")
        return cleaned.isdigit()

    def _has_merged_cells(self, cells: list[dict[str, Any]]) -> bool:
        """병합 셀 존재 여부.

        Args:
            cells: 셀 리스트

        Returns:
            병합 셀 존재 여부
        """
        for cell in cells:
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            if rowspan > 1 or colspan > 1:
                return True
        return False

    def _check_structure_consistency(
        self,
        cells: list[dict[str, Any]],
    ) -> list[str]:
        """구조 일관성 검사.

        Args:
            cells: 셀 리스트

        Returns:
            문제 리스트
        """
        issues = []

        # 행별 열 개수 확인
        rows: dict[int, set[int]] = {}
        for cell in cells:
            row_idx = int(cell.get("row", 0))
            col_idx = int(cell.get("col", 0))
            colspan = int(cell.get("colspan", 1))

            if row_idx not in rows:
                rows[row_idx] = set()
            for c in range(colspan):
                rows[row_idx].add(col_idx + c)

        if rows:
            col_counts = [len(cols) for cols in rows.values()]
            if max(col_counts) != min(col_counts):
                issues.append("inconsistent_column_count")

        # 중복 셀 위치 확인
        positions: set[tuple[int, int]] = set()
        for cell in cells:
            row = int(cell.get("row", 0))
            col = int(cell.get("col", 0))
            pos = (row, col)
            if pos in positions:
                issues.append("duplicate_cell_position")
                break
            positions.add(pos)

        return issues


__all__ = ["TableStructureValidator"]
