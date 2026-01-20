"""Quality assessment models.

품질 평가 결과 모델 정의.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReliabilityResult(BaseModel):
    """셀 신뢰도 계산 결과.

    Attributes:
        score: 최종 신뢰도 점수 (0.0 ~ 1.0)
        components: 각 요소별 점수 (ocr_conf, char_validity, pattern_match)
        flags: 경고/특이사항 플래그 목록
    """

    score: float = Field(ge=0.0, le=1.0, description="최종 신뢰도 점수")
    components: dict[str, float] = Field(default_factory=dict, description="각 요소별 점수")
    flags: list[str] = Field(default_factory=list, description="경고/특이사항 플래그")

    class Config:
        frozen = False


__all__ = ["ReliabilityResult"]
