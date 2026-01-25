"""Quality assessment models.

품질 평가 결과 모델 정의.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReliabilityResult(BaseModel):
    """셀 신뢰도 계산 결과.

    Attributes:
        score: 최종 신뢰도 점수 (0.0 ~ 1.0)
        components: 각 요소별 점수 (ocr_conf, char_validity, pattern_match)
        flags: 경고/특이사항 플래그 목록
    """

    model_config = ConfigDict(frozen=False)

    score: float = Field(ge=0.0, le=1.0, description="최종 신뢰도 점수")
    components: dict[str, float] = Field(default_factory=dict, description="각 요소별 점수")
    flags: list[str] = Field(default_factory=list, description="경고/특이사항 플래그")


class QualityGrade(str, Enum):
    """품질 등급."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class ReprocessLevel(str, Enum):
    """재처리 레벨."""

    FULL = "full"
    STRUCTURE_ONLY = "structure_only"
    OCR_ONLY = "ocr_only"
    MANUAL = "manual"


class ReprocessReason(str, Enum):
    """재처리 사유."""

    R001 = "low_fill_rate"
    R002 = "low_avg_reliability"
    R003 = "header_detection_failed"
    R004 = "structure_validation_failed"
    R005 = "ocr_bbox_missing"
    R006 = "merged_cell_error"
    R007 = "pattern_validation_failed"
    R008 = "excessive_low_conf_cells"


@dataclass
class ThresholdConfig:
    """품질 게이트 임계값 설정."""

    min_fill_rate: float = 0.60
    min_avg_reliability: float = 0.70
    max_low_conf_ratio: float = 0.25
    header_required: bool = True
    low_conf_threshold: float = 0.60


# Backward-compat alias for quality gate configuration.
GateThresholdConfig = ThresholdConfig


@dataclass
class GateResult:
    """품질 게이트 평가 결과."""

    passed: bool
    grade: QualityGrade
    action: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class HeaderInfo:
    """헤더 정보."""

    row_indices: list[int] = field(default_factory=lambda: [0])
    confidence: float = 0.0
    is_multi_level: bool = False


@dataclass
class StructureValidation:
    """구조 검증 결과."""

    is_valid: bool = True
    header_info: Optional[HeaderInfo] = None
    has_merged_cells: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class ReprocessHistory:
    """재처리 이력."""

    file_id: str = ""
    attempts: int = 0
    reasons: list[ReprocessReason] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    last_attempt: Optional[Any] = None  # datetime


@dataclass
class ReprocessDecision:
    """재처리 결정."""

    should_reprocess: bool = False
    level: ReprocessLevel = ReprocessLevel.FULL
    reasons: list[ReprocessReason] = field(default_factory=list)
    priority: int = 0
    max_attempts: int = 3


__all__ = [
    "ReliabilityResult",
    "QualityGrade",
    "ReprocessLevel",
    "ReprocessReason",
    "ThresholdConfig",
    "GateThresholdConfig",
    "GateResult",
    "HeaderInfo",
    "StructureValidation",
    "ReprocessHistory",
    "ReprocessDecision",
]
