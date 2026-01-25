# SPDX-License-Identifier: MIT
"""채널 분류기

보험약관 YAML 파일에서 판매 채널을 자동으로 분류합니다.

사용법:
    from pipeline.quality.channel_classifier import ChannelClassifier

    classifier = ChannelClassifier()
    result = classifier.classify(yaml_data)
    print(result.channel, result.confidence)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .channel_config import ChannelRulesConfig, get_channel_config

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ChannelMatch:
    """채널 매칭 결과"""

    channel: str
    source: str  # metadata | header | body | footer | filename
    confidence: float
    matched_text: str


@dataclass
class ChannelResult:
    """채널 분류 최종 결과"""

    channel: str
    confidence: float
    source: str
    mixed: bool
    candidates: list[str]
    matches: list[ChannelMatch]

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "channel": self.channel,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "mixed": self.mixed,
            "candidates": self.candidates,
        }


class ChannelClassifier:
    """채널 분류기

    보험약관 문서에서 판매 채널을 추출합니다.
    """

    def __init__(self, config: ChannelRulesConfig | None = None) -> None:
        """초기화

        Args:
            config: 채널 분류 설정. None이면 기본 설정 로드.
        """
        self.config = config or get_channel_config()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """채널 키워드 및 패턴을 정규표현식으로 컴파일"""
        self._keyword_patterns: dict[str, re.Pattern] = {}
        self._filename_patterns: dict[str, re.Pattern] = {}

        for channel_name, channel_def in self.config.channels.items():
            # 키워드 + 동의어 패턴
            all_keywords = channel_def.get_all_keywords()
            if all_keywords:
                pattern = "|".join(re.escape(kw) for kw in all_keywords)
                self._keyword_patterns[channel_name] = re.compile(pattern)

            # 파일명 패턴 (괄호 포함)
            if channel_def.patterns:
                pattern = "|".join(channel_def.patterns)
                try:
                    self._filename_patterns[channel_name] = re.compile(pattern)
                except re.error:
                    pass

    def classify(
        self,
        yaml_data: dict[str, Any],
        filename: str | None = None,
    ) -> ChannelResult:
        """YAML 데이터에서 채널 분류

        Args:
            yaml_data: 파싱된 YAML 데이터
            filename: 원본 파일명 (선택)

        Returns:
            ChannelResult: 분류 결과
        """
        matches: list[ChannelMatch] = []

        # 1. 메타데이터에서 추출 (최우선)
        meta_match = self._extract_from_metadata(yaml_data)
        if meta_match:
            matches.append(meta_match)

        # 2. 파일명에서 추출
        if filename:
            filename_matches = self._extract_from_filename(filename)
            matches.extend(filename_matches)

        # 3. 본문에서 추출
        content = self._extract_text_content(yaml_data)
        if content:
            content_matches = self._extract_from_content(content)
            matches.extend(content_matches)

        # 4. 결과 종합
        return self._aggregate_results(matches)

    def _extract_from_metadata(self, yaml_data: dict[str, Any]) -> ChannelMatch | None:
        """메타데이터에서 채널 추출"""
        metadata = yaml_data.get("metadata", {})
        if not isinstance(metadata, dict):
            return None

        for field in self.config.extraction.metadata_fields:
            if field in metadata:
                value = str(metadata[field]).strip()
                if value:
                    # 채널명과 직접 매칭
                    for channel_name in self.config.channels:
                        if channel_name in value or value == channel_name:
                            return ChannelMatch(
                                channel=channel_name,
                                source="metadata",
                                confidence=self.config.confidence.metadata,
                                matched_text=value,
                            )
        return None

    def _extract_from_filename(self, filename: str) -> list[ChannelMatch]:
        """파일명에서 채널 추출"""
        matches: list[ChannelMatch] = []

        for channel_name, pattern in self._filename_patterns.items():
            match = pattern.search(filename)
            if match:
                matches.append(
                    ChannelMatch(
                        channel=channel_name,
                        source="filename",
                        confidence=self.config.confidence.filename,
                        matched_text=match.group(),
                    )
                )

        return matches

    def _extract_text_content(self, yaml_data: dict[str, Any]) -> str:
        """YAML에서 텍스트 본문 추출"""
        # 여러 가능한 구조 시도
        content = ""

        # 구조 1: content.paragraphs
        if "content" in yaml_data and isinstance(yaml_data["content"], dict):
            paragraphs = yaml_data["content"].get("paragraphs", [])
            if isinstance(paragraphs, list):
                content = " ".join(str(p) for p in paragraphs)

        # 구조 2: document.text
        if not content and "document" in yaml_data:
            doc = yaml_data["document"]
            if isinstance(doc, dict) and "text" in doc:
                content = str(doc["text"])

        # 구조 3: text 필드
        if not content and "text" in yaml_data:
            content = str(yaml_data["text"])

        # 구조 4: sections
        if not content and "sections" in yaml_data:
            sections = yaml_data["sections"]
            if isinstance(sections, list):
                texts = []
                for section in sections:
                    if isinstance(section, dict) and "content" in section:
                        texts.append(str(section["content"]))
                content = " ".join(texts)

        return content

    def _extract_from_content(self, content: str) -> list[ChannelMatch]:
        """본문에서 채널 추출"""
        matches: list[ChannelMatch] = []
        content_len = len(content)

        header_chars = self.config.extraction.header_chars
        footer_chars = self.config.extraction.footer_chars

        # 영역 분리
        header = content[:header_chars] if content_len > header_chars else content
        footer = content[-footer_chars:] if content_len > footer_chars else ""
        body = content[header_chars:-footer_chars] if content_len > header_chars + footer_chars else ""

        # 각 영역에서 검색
        for channel_name, pattern in self._keyword_patterns.items():
            channel_def = self.config.channels.get(channel_name)

            # exclude 키워드가 있으면 제외
            if channel_def and channel_def.exclude:
                has_exclude = any(exc in content for exc in channel_def.exclude)
                if has_exclude:
                    continue

            # 헤더에서 검색
            header_match = pattern.search(header)
            if header_match:
                matches.append(
                    ChannelMatch(
                        channel=channel_name,
                        source="header",
                        confidence=self.config.confidence.header,
                        matched_text=header_match.group(),
                    )
                )
                continue  # 같은 채널 중복 방지

            # 본문에서 검색
            body_match = pattern.search(body)
            if body_match:
                matches.append(
                    ChannelMatch(
                        channel=channel_name,
                        source="body",
                        confidence=self.config.confidence.body,
                        matched_text=body_match.group(),
                    )
                )
                continue

            # 푸터에서 검색
            footer_match = pattern.search(footer)
            if footer_match:
                matches.append(
                    ChannelMatch(
                        channel=channel_name,
                        source="footer",
                        confidence=self.config.confidence.footer,
                        matched_text=footer_match.group(),
                    )
                )

        return matches

    def _aggregate_results(self, matches: list[ChannelMatch]) -> ChannelResult:
        """매칭 결과 종합"""
        if not matches:
            # 매칭 없음 → 일반 채널
            return ChannelResult(
                channel="일반",
                confidence=0.0,
                source="none",
                mixed=False,
                candidates=[],
                matches=[],
            )

        # 채널별 그룹핑 및 최고 신뢰도 선택
        channel_scores: dict[str, float] = {}
        channel_sources: dict[str, str] = {}

        for match in matches:
            current = channel_scores.get(match.channel, 0.0)
            if match.confidence > current:
                channel_scores[match.channel] = match.confidence
                channel_sources[match.channel] = match.source

        # 정렬 (신뢰도 높은 순)
        sorted_channels = sorted(
            channel_scores.items(),
            key=lambda x: (-x[1], self.config.channels.get(x[0], type("", (), {"priority": 99})).priority),
        )

        if len(sorted_channels) == 1:
            # 단일 채널
            top_channel, top_score = sorted_channels[0]
            return ChannelResult(
                channel=top_channel,
                confidence=top_score,
                source=channel_sources[top_channel],
                mixed=False,
                candidates=[top_channel],
                matches=matches,
            )

        # 혼합 채널 판정
        top_channel, top_score = sorted_channels[0]
        second_channel, second_score = sorted_channels[1]

        threshold = self.config.mixed_channel.threshold
        is_mixed = (second_score / top_score) >= threshold if top_score > 0 else False

        # 전략에 따른 최종 채널 결정
        strategy = self.config.mixed_channel.strategy

        if strategy == "primary_by_frequency":
            # 빈도 기반 (이미 신뢰도로 정렬됨)
            final_channel = top_channel
        elif strategy == "primary_by_priority":
            # 우선순위 기반
            candidates = [ch for ch, _ in sorted_channels]
            sorted_by_priority = sorted(
                candidates,
                key=lambda ch: self.config.channels.get(ch, type("", (), {"priority": 99})).priority,
            )
            final_channel = sorted_by_priority[0]
        else:
            # manual_review
            final_channel = top_channel

        return ChannelResult(
            channel=final_channel,
            confidence=top_score,
            source=channel_sources[top_channel],
            mixed=is_mixed,
            candidates=[ch for ch, _ in sorted_channels],
            matches=matches,
        )

    def classify_file(self, file_path: Path | str) -> ChannelResult:
        """YAML 파일에서 채널 분류

        Args:
            file_path: YAML 파일 경로

        Returns:
            ChannelResult: 분류 결과
        """
        from pathlib import Path

        import yaml

        file_path = Path(file_path)

        with open(file_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            yaml_data = {}

        return self.classify(yaml_data, filename=file_path.name)
