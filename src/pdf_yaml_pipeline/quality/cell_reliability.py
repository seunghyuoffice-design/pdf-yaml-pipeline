"""Cell reliability calculation module.

셀 단위 복합 신뢰도 계산.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .models import ReliabilityResult


class CellReliabilityCalculator:
    """셀 신뢰도 복합 계산기.

    세 가지 요소를 결합하여 신뢰도 점수 산출:
    - OCR confidence (50%)
    - 문자 유효성 (30%)
    - 패턴 매칭 (20%)
    """

    WEIGHTS = {
        "ocr_conf": 0.5,
        "char_validity": 0.3,
        "pattern_match": 0.2,
    }

    # 숫자 패턴 (보험 도메인 특화)
    PATTERNS = {
        "amount": re.compile(r"^[\d,]+(\.\d+)?원?$"),
        "amount_unit": re.compile(r"^[\d,]+(\.\d+)?\s*(원|만원|천원|억)$"),
        "date_ymd": re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$"),
        "date_ym": re.compile(r"^\d{4}[-./]\d{1,2}$"),
        "percentage": re.compile(r"^\d+(\.\d+)?%$"),
        "age": re.compile(r"^\d{1,3}세?$"),
        "period": re.compile(r"^\d+\s*(년|개월|일|세)$"),
        "phone": re.compile(r"^\d{2,3}-\d{3,4}-\d{4}$"),
    }

    # 패턴 타임아웃 (ReDoS 방지)
    PATTERN_TIMEOUT = 0.1  # 100ms

    def calculate(self, cell: dict[str, Any]) -> ReliabilityResult:
        """복합 신뢰도 계산.

        Args:
            cell: 셀 딕셔너리 (text, ocr_text, ocr_confidence 등)

        Returns:
            ReliabilityResult: 신뢰도 결과
        """
        text = (cell.get("text") or "").strip()
        ocr_text = (cell.get("ocr_text") or "").strip()
        content = text or ocr_text

        # 빈 셀 처리
        if not content:
            return ReliabilityResult(
                score=0.0,
                components={"ocr_conf": 0.0, "char_validity": 0.0, "pattern_match": 0.0},
                flags=["empty_cell"],
            )

        # 각 요소 계산
        ocr_score = self._calc_ocr_confidence(cell)
        char_score = self._calc_char_validity(content)
        pattern_score = self._calc_pattern_match(content)

        # 가중 평균
        total = (
            self.WEIGHTS["ocr_conf"] * ocr_score
            + self.WEIGHTS["char_validity"] * char_score
            + self.WEIGHTS["pattern_match"] * pattern_score
        )

        # 플래그 수집
        flags = []
        if len(content) <= 1:
            flags.append("single_char")
            total -= 0.15  # 단일 문자 페널티
        if ocr_score < 0.5:
            flags.append("low_ocr_conf")
        if char_score < 0.5:
            flags.append("low_char_validity")

        return ReliabilityResult(
            score=max(0.0, min(1.0, total)),
            components={
                "ocr_conf": round(ocr_score, 4),
                "char_validity": round(char_score, 4),
                "pattern_match": round(pattern_score, 4),
            },
            flags=flags,
        )

    def _calc_ocr_confidence(self, cell: dict[str, Any]) -> float:
        """OCR confidence 정규화.

        text 필드가 있으면 기본 0.90 (구조적 텍스트),
        ocr_text만 있으면 ocr_confidence 사용.
        """
        text = (cell.get("text") or "").strip()
        ocr_conf = cell.get("ocr_confidence")

        if text:
            # 구조적 텍스트는 높은 신뢰도
            return 0.90

        if ocr_conf is not None:
            # OCR confidence는 0-1 범위로 정규화
            return max(0.0, min(1.0, float(ocr_conf)))

        # 기본값
        return 0.50

    def _calc_char_validity(self, text: str) -> float:
        """문자 유효성 계산.

        한글, 영문, 숫자의 비율에 따른 점수.
        특수문자 과다 시 페널티.
        """
        if not text:
            return 0.0

        total = len(text)
        valid_count = 0
        special_count = 0

        for char in text:
            cat = unicodedata.category(char)
            if cat.startswith("L"):  # Letter (한글, 영문 등)
                valid_count += 1
            elif cat.startswith("N"):  # Number
                valid_count += 1
            elif char in " \t\n":  # 공백
                valid_count += 0.5
            elif cat.startswith("P") or cat.startswith("S"):  # Punctuation, Symbol
                special_count += 1

        valid_ratio = valid_count / total
        special_ratio = special_count / total

        # 특수문자 30% 초과 시 페널티
        penalty = max(0, (special_ratio - 0.3) * 0.5)

        return max(0.0, min(1.0, valid_ratio - penalty))

    def _calc_pattern_match(self, text: str) -> float:
        """숫자 패턴 매칭 보너스.

        보험 도메인 특화 패턴 (금액, 날짜, 비율 등) 매칭 시 보너스.
        """
        if not text:
            return 0.5  # 기본값

        text = text.strip()

        # 패턴 매칭 시도
        for pattern_name, pattern in self.PATTERNS.items():
            try:
                if pattern.match(text):
                    return 1.0  # 완전 매칭
            except Exception:
                continue

        # 부분 숫자 포함 시 약간의 보너스
        if any(c.isdigit() for c in text):
            return 0.6

        # 일반 텍스트
        return 0.5


def calculate_cell_reliability(cell: dict[str, Any]) -> float:
    """셀 신뢰도 계산 (하위 호환 함수).

    기존 API 호환을 위한 래퍼 함수.

    Args:
        cell: 셀 딕셔너리

    Returns:
        신뢰도 점수 (0.0 ~ 1.0)
    """
    calc = CellReliabilityCalculator()
    result = calc.calculate(cell)
    return result.score


def attach_cell_reliability(table: dict[str, Any]) -> None:
    """테이블 내 모든 셀에 신뢰도 부착.

    Args:
        table: 테이블 딕셔너리 (cells 리스트 포함)
    """
    calc = CellReliabilityCalculator()
    for cell in table.get("cells", []) or []:
        if cell.get("reliability") is None:
            result = calc.calculate(cell)
            cell["reliability"] = round(result.score, 4)
            cell["reliability_detail"] = result.components


__all__ = [
    "CellReliabilityCalculator",
    "calculate_cell_reliability",
    "attach_cell_reliability",
]
