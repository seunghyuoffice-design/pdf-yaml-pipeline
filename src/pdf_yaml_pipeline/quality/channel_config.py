# SPDX-License-Identifier: MIT
"""채널 분류 설정 로더 및 스키마 정의

이 모듈은 config/channel_rules.yaml을 로드하고 검증합니다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ChannelDefinition(BaseModel):
    """개별 채널 정의"""

    keywords: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    priority: int = Field(default=99)

    # 컴파일된 패턴 (런타임)
    _compiled_patterns: list[re.Pattern] | None = None

    def get_compiled_patterns(self) -> list[re.Pattern]:
        """정규표현식 패턴을 컴파일하여 반환"""
        if self._compiled_patterns is None:
            self._compiled_patterns = []
            for pattern in self.patterns:
                try:
                    self._compiled_patterns.append(re.compile(pattern))
                except re.error:
                    pass  # 잘못된 패턴 무시
        return self._compiled_patterns

    def get_all_keywords(self) -> set[str]:
        """모든 키워드 (keywords + synonyms) 반환"""
        return set(self.keywords) | set(self.synonyms)


class ExtractionConfig(BaseModel):
    """텍스트 추출 설정"""

    header_chars: int = Field(default=1000)
    footer_chars: int = Field(default=300)
    metadata_fields: list[str] = Field(default_factory=lambda: ["channel", "sales_channel", "판매채널"])


class ConfidenceConfig(BaseModel):
    """신뢰도 점수 설정"""

    metadata: float = Field(default=1.0)
    header: float = Field(default=0.9)
    body: float = Field(default=0.7)
    footer: float = Field(default=0.6)
    filename: float = Field(default=0.95)

    @field_validator("metadata", "header", "body", "footer", "filename")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("신뢰도는 0.0 ~ 1.0 사이여야 합니다")
        return v


class MixedChannelConfig(BaseModel):
    """혼합 채널 처리 설정"""

    strategy: str = Field(default="primary_by_frequency")
    threshold: float = Field(default=0.3)
    send_to_review: bool = Field(default=True)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = ["primary_by_frequency", "primary_by_priority", "manual_review"]
        if v not in allowed:
            raise ValueError(f"strategy는 {allowed} 중 하나여야 합니다")
        return v


class AutoEnqueueCondition(BaseModel):
    """자동 검토 큐 전송 조건"""

    confidence_below: float | None = None
    mixed_channels: bool | None = None
    no_match: bool | None = None


class ReviewQueueConfig(BaseModel):
    """검토 큐 설정"""

    max_size: int = Field(default=1000)
    ttl_hours: int = Field(default=168)
    auto_enqueue_when: list[dict[str, Any]] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
    """모니터링 설정"""

    rolling_window: int = Field(default=1000)
    alert_threshold: float = Field(default=0.2)
    expected_distribution: dict[str, float] = Field(default_factory=dict)


class ChannelRulesConfig(BaseModel):
    """채널 분류 전체 설정"""

    version: str = Field(default="1.0")
    channels: dict[str, ChannelDefinition] = Field(default_factory=dict)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    mixed_channel: MixedChannelConfig = Field(default_factory=MixedChannelConfig)
    review_queue: ReviewQueueConfig = Field(default_factory=ReviewQueueConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    error_codes: dict[str, str] = Field(default_factory=dict)

    def get_channel_by_priority(self) -> list[tuple[str, ChannelDefinition]]:
        """우선순위 순으로 정렬된 채널 목록 반환"""
        return sorted(self.channels.items(), key=lambda x: x[1].priority)


def load_channel_config(config_path: Path | str | None = None) -> ChannelRulesConfig:
    """채널 분류 설정 로드

    Args:
        config_path: 설정 파일 경로. None이면 기본 경로 사용.

    Returns:
        ChannelRulesConfig: 검증된 설정 객체

    Raises:
        FileNotFoundError: 설정 파일이 없는 경우
        ValueError: 설정 형식이 잘못된 경우
    """
    if config_path is None:
        # 기본 경로: 프로젝트 루트/config/channel_rules.yaml
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "channel_rules.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    # channels 딕셔너리를 ChannelDefinition으로 변환
    if "channels" in raw_config:
        raw_config["channels"] = {name: ChannelDefinition(**defn) for name, defn in raw_config["channels"].items()}

    return ChannelRulesConfig(**raw_config)


# 싱글톤 패턴으로 설정 캐싱
_cached_config: ChannelRulesConfig | None = None


def get_channel_config(reload: bool = False) -> ChannelRulesConfig:
    """캐시된 채널 설정 반환

    Args:
        reload: True이면 설정 파일을 다시 로드

    Returns:
        ChannelRulesConfig: 채널 분류 설정
    """
    global _cached_config  # noqa: PLW0603

    if _cached_config is None or reload:
        _cached_config = load_channel_config()

    return _cached_config
