# SPDX-License-Identifier: MIT
"""본문 분석기

YAML 본문에서 채널 키워드를 탐지하고 신뢰도를 계산한다.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .channel_keywords import ChannelKeywords


@dataclass
class ContentAnalysisResult:
    """본문 분석 결과"""

    channel: str
    confidence: float  # 0.0 - 1.0
    location: str  # "header" | "body" | "footer"
    keyword_count: int
    matched_keywords: List[str]


class ContentAnalyzer:
    """본문 분석기

    YAML 본문에서 채널 키워드를 탐지하고
    위치 기반 신뢰도를 계산한다.

    Academy 근거: Boolean Retrieval Model (Salton & McGill, 1983)
    """

    # 헤더 영역 문자 수
    HEADER_SIZE = 1000
    # 푸터 영역 문자 수
    FOOTER_SIZE = 500

    # 기본 신뢰도 설정
    DEFAULT_CONFIDENCE_CONFIG = {
        "base": 0.3,
        "location": {"header": 0.35, "body": 0.1, "footer": 0.05},
        "frequency": {"coefficient": 0.06, "max": 0.3},
        "penalty": {"negative": 0.35},
    }

    def __init__(
        self,
        keywords: Optional[ChannelKeywords] = None,
        confidence_config: Optional[Dict[str, Any]] = None,
    ):
        """초기화

        Args:
            keywords: 키워드 사전 (None이면 기본값 사용)
            confidence_config: 신뢰도 계산 설정 (None이면 기본값 사용)
        """
        self.keywords = keywords or ChannelKeywords()
        self.conf = confidence_config or self.DEFAULT_CONFIDENCE_CONFIG
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """채널별 정규표현식 패턴 컴파일"""
        self._patterns: Dict[str, re.Pattern] = {}
        self._negative_patterns: Dict[str, re.Pattern] = {}

        for channel in self.keywords.get_all_channels():
            # 긍정 패턴
            all_kw = self.keywords.get_all_keywords(channel)
            if all_kw:
                pattern = "|".join(re.escape(kw) for kw in all_kw)
                self._patterns[channel] = re.compile(pattern, re.IGNORECASE)

            # 부정 패턴
            neg_kw = self.keywords.get_negative_keywords(channel)
            if neg_kw:
                neg_pattern = "|".join(re.escape(kw) for kw in neg_kw)
                self._negative_patterns[channel] = re.compile(neg_pattern, re.IGNORECASE)

    def analyze(self, yaml_data: Dict[str, Any]) -> List[ContentAnalysisResult]:
        """본문 분석하여 채널 후보 반환

        Args:
            yaml_data: YAML 파일 내용

        Returns:
            채널 분석 결과 리스트 (신뢰도 순)
        """
        text = self._extract_text(yaml_data)
        if not text:
            return []

        results = []
        for channel in self.keywords.get_all_channels():
            result = self._analyze_channel(text, channel)
            if result:
                results.append(result)

        # 신뢰도 내림차순 정렬
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _extract_text(self, yaml_data: Dict[str, Any]) -> str:
        """YAML에서 본문 텍스트 추출

        Args:
            yaml_data: YAML 파일 내용

        Returns:
            본문 텍스트
        """
        text_parts = []

        # content.paragraphs 형식
        if "content" in yaml_data:
            content = yaml_data["content"]
            if isinstance(content, dict) and "paragraphs" in content:
                paragraphs = content["paragraphs"]
                if isinstance(paragraphs, list):
                    text_parts.extend(str(p) for p in paragraphs)

        # document.text 형식
        if "document" in yaml_data:
            doc = yaml_data["document"]
            if isinstance(doc, dict):
                if "text" in doc:
                    text_parts.append(str(doc["text"]))
                if "title" in doc:
                    text_parts.insert(0, str(doc["title"]))

        # messages 형식 (학습 데이터)
        if "messages" in yaml_data:
            messages = yaml_data["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        text_parts.append(str(msg["content"]))

        return " ".join(text_parts)

    def _analyze_channel(self, text: str, channel: str) -> Optional[ContentAnalysisResult]:
        """특정 채널에 대한 분석 수행

        Args:
            text: 본문 텍스트
            channel: 채널명

        Returns:
            분석 결과 (매칭 없으면 None)
        """
        if channel not in self._patterns:
            return None

        pattern = self._patterns[channel]
        matches = pattern.findall(text)

        if not matches:
            return None

        # 부정 키워드 확인
        has_negative = False
        if channel in self._negative_patterns:
            neg_matches = self._negative_patterns[channel].findall(text)
            has_negative = len(neg_matches) > 0

        # 위치 판단
        location = self._determine_location(text, pattern)

        # 신뢰도 계산
        confidence = self._calculate_confidence(
            keyword_count=len(matches),
            location=location,
            has_negative=has_negative,
        )

        return ContentAnalysisResult(
            channel=channel,
            confidence=confidence,
            location=location,
            keyword_count=len(matches),
            matched_keywords=list(set(matches)),
        )

    def _determine_location(self, text: str, pattern: re.Pattern) -> str:
        """키워드 위치 판단

        Args:
            text: 본문 텍스트
            pattern: 검색 패턴

        Returns:
            "header" | "body" | "footer"
        """
        header = text[: self.HEADER_SIZE]
        footer = text[-self.FOOTER_SIZE :] if len(text) > self.FOOTER_SIZE else ""

        if pattern.search(header):
            return "header"
        elif footer and pattern.search(footer):
            return "footer"
        else:
            return "body"

    def _calculate_confidence(
        self,
        keyword_count: int,
        location: str,
        has_negative: bool,
    ) -> float:
        """신뢰도 점수 계산

        Args:
            keyword_count: 키워드 출현 횟수
            location: 키워드 위치
            has_negative: 부정 키워드 존재 여부

        Returns:
            신뢰도 점수 (0.0 - 1.0)
        """
        # 설정에서 값 로드
        base = self.conf.get("base", 0.3)
        loc_weights = self.conf.get("location", {})
        freq_config = self.conf.get("frequency", {})
        penalty_config = self.conf.get("penalty", {})

        # 기본 점수
        score = base

        # 위치 가중치
        location_weights = {
            "header": loc_weights.get("header", 0.35),
            "body": loc_weights.get("body", 0.1),
            "footer": loc_weights.get("footer", 0.05),
        }
        score += location_weights.get(location, 0.1)

        # 빈도 가중치
        freq_coef = freq_config.get("coefficient", 0.06)
        freq_max = freq_config.get("max", 0.3)
        frequency_score = min(keyword_count * freq_coef, freq_max)
        score += frequency_score

        # 부정 키워드 페널티
        if has_negative:
            neg_penalty = penalty_config.get("negative", 0.35)
            score -= neg_penalty

        # 0.0 - 1.0 범위로 클램핑
        return max(0.0, min(1.0, score))
