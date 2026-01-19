"""Table quality assessment module.

테이블 품질 통합 평가 모듈.
셀 신뢰도 계산, 구조 검증, 품질 게이트, 재처리 전략을 통합 제공한다.

v2.0: 복합 신뢰도, 동적 임계값, 헤더 자동 감지, 재처리 전략 추가
"""

from __future__ import annotations

from typing import Any, Optional

from .cell_reliability import CellReliabilityCalculator, attach_cell_reliability
from .models import (
    GateResult,
    HeaderInfo,
    QualityGrade,
    ReliabilityResult,
    ReprocessDecision,
    ReprocessHistory,
    ReprocessLevel,
    ReprocessReason,
    StructureValidation,
    ThresholdConfig,
)
from .quality_gate import QualityGate
from .reprocess_strategy import ReprocessStrategy, get_reprocess_reason_description
from .table_structure import TableStructureValidator


class TableQualityAssessor:
    """테이블 품질 통합 평가.

    전체 품질 평가 파이프라인을 단일 인터페이스로 제공한다.

    Example:
        >>> assessor = TableQualityAssessor()
        >>> result = assessor.assess(table)
        >>> if result["quality"]["passed"]:
        ...     print("품질 통과!")
    """

    def __init__(
        self,
        threshold_config: Optional[ThresholdConfig] = None,
        use_dynamic_threshold: bool = True,
    ):
        """초기화.

        Args:
            threshold_config: 품질 게이트 임계값 설정 (None이면 기본값)
            use_dynamic_threshold: 테이블 크기에 따른 동적 임계값 사용 여부
        """
        self.reliability_calc = CellReliabilityCalculator()
        self.structure_validator = TableStructureValidator()
        self.quality_gate = QualityGate(threshold_config)
        self.reprocess_strategy = ReprocessStrategy()
        self.use_dynamic_threshold = use_dynamic_threshold

    def assess(
        self,
        table: dict[str, Any],
        history: Optional[ReprocessHistory] = None,
    ) -> dict[str, Any]:
        """전체 품질 평가 파이프라인.

        1. 셀 신뢰도 계산 (복합 점수)
        2. 테이블 구조 검증 (헤더 감지, 병합 셀)
        3. 품질 게이트 판정 (등급 산출)
        4. 재처리 전략 결정

        Args:
            table: 테이블 딕셔너리 (cells 리스트 포함)
            history: 이전 재처리 이력 (있으면)

        Returns:
            통합 평가 결과 딕셔너리
        """
        # 1. 셀 신뢰도 계산
        for cell in table.get("cells", []) or []:
            if cell.get("reliability") is None:
                result = self.reliability_calc.calculate(cell)
                cell["reliability"] = round(result.score, 4)
                cell["reliability_detail"] = result.components

        # 2. 구조 검증
        structure = self.structure_validator.validate(table)

        # 3. 품질 게이트
        gate_result = self.quality_gate.evaluate(table, use_dynamic_threshold=self.use_dynamic_threshold)

        # 4. 재처리 전략
        reprocess = self.reprocess_strategy.decide(gate_result, history)

        return {
            "quality": {
                "passed": gate_result.passed,
                "grade": gate_result.grade.value,
                "action": gate_result.action,
                "metrics": gate_result.metrics,
                "reasons": gate_result.reasons,
            },
            "structure": {
                "valid": structure.is_valid,
                "header_rows": (structure.header_info.row_indices if structure.header_info else [0]),
                "header_confidence": (structure.header_info.confidence if structure.header_info else 0.0),
                "is_multi_level_header": (structure.header_info.is_multi_level if structure.header_info else False),
                "has_merged_cells": structure.has_merged_cells,
                "issues": structure.issues,
            },
            "reprocess": {
                "needed": reprocess.should_reprocess,
                "level": reprocess.level.value,
                "reasons": [r.value for r in reprocess.reasons],
                "priority": reprocess.priority,
            },
        }

    def assess_and_attach(
        self,
        table: dict[str, Any],
        history: Optional[ReprocessHistory] = None,
    ) -> dict[str, Any]:
        """평가 후 테이블에 결과 부착.

        Args:
            table: 테이블 딕셔너리
            history: 이전 재처리 이력

        Returns:
            테이블 (quality, structure, reprocess 필드 추가됨)
        """
        result = self.assess(table, history)
        table["quality"] = result["quality"]
        table["structure"] = result["structure"]
        table["reprocess"] = result["reprocess"]
        return table


# =============================================================================
# 하위 호환 함수 (기존 API 유지)
# =============================================================================


def calculate_cell_reliability(cell: dict[str, Any]) -> float:
    """셀 신뢰도 계산 (하위 호환).

    기존 단순 알고리즘 대신 복합 점수 사용.

    Args:
        cell: 셀 딕셔너리

    Returns:
        신뢰도 점수 (0.0 ~ 1.0)
    """
    calc = CellReliabilityCalculator()
    result = calc.calculate(cell)
    return result.score


def attach_table_quality(table: dict[str, Any]) -> None:
    """테이블 품질 평가 및 부착 (하위 호환).

    기존 API와 동일한 출력 형식 유지.

    Args:
        table: 테이블 딕셔너리
    """
    # 셀 신뢰도 먼저 계산
    attach_cell_reliability(table)

    # 통합 평가
    assessor = TableQualityAssessor()
    result = assessor.assess(table)

    # 기존 형식으로 변환 (호환성)
    table["quality"] = {
        "passed": result["quality"]["passed"],
        "action": result["quality"]["action"],
        "metrics": {
            "fill_rate": result["quality"]["metrics"].get("fill_rate", 0.0),
            "avg_reliability": result["quality"]["metrics"].get("avg_reliability", 0.0),
            "low_conf_ratio": result["quality"]["metrics"].get("low_conf_ratio", 1.0),
            "header_ok": result["quality"]["metrics"].get("header_ok", False),
        },
        # 추가 필드 (v2.0)
        "grade": result["quality"]["grade"],
        "reasons": result["quality"]["reasons"],
    }


__all__ = [
    # 통합 클래스
    "TableQualityAssessor",
    # 하위 호환 함수
    "calculate_cell_reliability",
    "attach_cell_reliability",
    "attach_table_quality",
    # 컴포넌트
    "CellReliabilityCalculator",
    "QualityGate",
    "TableStructureValidator",
    "ReprocessStrategy",
    # 모델
    "QualityGrade",
    "ReprocessReason",
    "ReprocessLevel",
    "ReliabilityResult",
    "ThresholdConfig",
    "GateResult",
    "HeaderInfo",
    "StructureValidation",
    "ReprocessDecision",
    "ReprocessHistory",
    # 유틸리티
    "get_reprocess_reason_description",
]
