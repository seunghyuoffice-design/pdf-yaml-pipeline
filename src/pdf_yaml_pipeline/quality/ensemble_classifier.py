# SPDX-License-Identifier: MIT
"""앙상블 분류기

본문, 메타데이터, 파일명 분석 결과를 통합하여 최종 채널을 결정한다.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .channel_keywords import ChannelKeywords
from .content_analyzer import ContentAnalysisResult, ContentAnalyzer
from .metadata_extractor import MetadataExtractor, MetadataResult

logger = logging.getLogger(__name__)

# 기본 설정 파일 경로
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "thresholds.yaml"


@dataclass
class ClassificationResult:
    """분류 결과"""

    channel: Optional[str]
    confidence: float
    route: str  # "auto" | "review" | "unclassified"
    sources: Dict[str, Any] = field(default_factory=dict)


class ConfigValidationError(ValueError):
    """설정 검증 오류"""

    pass


@dataclass
class ThresholdConfig:
    """임계값 설정"""

    auto: float = 0.6
    review: float = 0.35
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "content": 0.6,
            "metadata": 0.3,
            "filename": 0.1,
        }
    )
    confidence: Dict[str, Any] = field(
        default_factory=lambda: {
            "base": 0.3,
            "location": {"header": 0.35, "body": 0.1, "footer": 0.05},
            "frequency": {"coefficient": 0.06, "max": 0.3},
            "penalty": {"negative": 0.35},
        }
    )
    channels: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        """초기화 후 검증"""
        errors = self.validate()
        if errors:
            logger.warning(f"설정 검증 경고: {errors}")

    def validate(self) -> List[str]:
        """설정 검증

        Returns:
            오류 메시지 리스트 (빈 리스트면 유효)
        """
        errors = []

        # 임계값 범위 검증
        if not 0.0 <= self.auto <= 1.0:
            errors.append(f"auto 임계값 범위 오류: {self.auto} (0.0-1.0)")
        if not 0.0 <= self.review <= 1.0:
            errors.append(f"review 임계값 범위 오류: {self.review} (0.0-1.0)")
        if self.auto <= self.review:
            errors.append(f"auto({self.auto}) <= review({self.review}): auto가 더 커야 함")

        # 가중치 검증
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"가중치 합계 오류: {weight_sum:.3f} (1.0이어야 함)")

        required_weights = {"content", "metadata", "filename"}
        missing_weights = required_weights - set(self.weights.keys())
        if missing_weights:
            errors.append(f"필수 가중치 누락: {missing_weights}")

        # 신뢰도 설정 검증
        if "base" not in self.confidence:
            errors.append("confidence.base 누락")
        elif not 0.0 <= self.confidence["base"] <= 1.0:
            errors.append(f"base 범위 오류: {self.confidence['base']}")

        if "location" not in self.confidence:
            errors.append("confidence.location 누락")
        else:
            for loc in ["header", "body", "footer"]:
                if loc not in self.confidence["location"]:
                    errors.append(f"confidence.location.{loc} 누락")

        if "frequency" not in self.confidence:
            errors.append("confidence.frequency 누락")
        else:
            if "coefficient" not in self.confidence["frequency"]:
                errors.append("confidence.frequency.coefficient 누락")
            if "max" not in self.confidence["frequency"]:
                errors.append("confidence.frequency.max 누락")

        if "penalty" not in self.confidence:
            errors.append("confidence.penalty 누락")
        elif "negative" not in self.confidence["penalty"]:
            errors.append("confidence.penalty.negative 누락")

        # 채널별 오버라이드 검증
        for channel, ch_config in self.channels.items():
            if "auto" in ch_config and not 0.0 <= ch_config["auto"] <= 1.0:
                errors.append(f"channels.{channel}.auto 범위 오류")
            if "review" in ch_config and not 0.0 <= ch_config["review"] <= 1.0:
                errors.append(f"channels.{channel}.review 범위 오류")

        return errors

    def validate_strict(self) -> None:
        """엄격한 검증 (오류 시 예외 발생)

        Raises:
            ConfigValidationError: 검증 실패 시
        """
        errors = self.validate()
        if errors:
            raise ConfigValidationError(f"설정 검증 실패: {errors}")

    @classmethod
    def from_yaml(cls, path: Path, strict: bool = False) -> "ThresholdConfig":
        """YAML 파일에서 설정 로드

        Args:
            path: 설정 파일 경로
            strict: True이면 검증 실패 시 예외 발생

        Returns:
            ThresholdConfig 인스턴스

        Raises:
            ConfigValidationError: strict=True이고 검증 실패 시
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            routing = data.get("routing", {})
            config = cls(
                auto=routing.get("auto", 0.6),
                review=routing.get("review", 0.35),
                weights=data.get("weights", cls.__dataclass_fields__["weights"].default_factory()),
                confidence=data.get("confidence", cls.__dataclass_fields__["confidence"].default_factory()),
                channels=data.get("channels", {}),
            )

            if strict:
                config.validate_strict()

            return config
        except ConfigValidationError:
            raise
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본값 사용: {e}")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환

        Returns:
            설정 딕셔너리
        """
        return {
            "routing": {
                "auto": self.auto,
                "review": self.review,
            },
            "weights": self.weights,
            "confidence": self.confidence,
            "channels": self.channels,
        }

    def save_yaml(self, path: Path) -> None:
        """YAML 파일로 저장

        Args:
            path: 저장 경로
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)


class EnsembleClassifier:
    """앙상블 분류기

    다중 소스(본문, 메타데이터, 파일명)를 통합하여
    최종 채널을 결정한다.

    Academy 근거: Ensemble Methods (Dietterich, 2000)

    Note:
        RULES: RUNTIME - Docker only
        이 분류기는 sage-pipeline 컨테이너 내에서 실행되어야 한다.
    """

    # 보안: 파일 크기 제한 (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024

    def __init__(
        self,
        content_analyzer: Optional[ContentAnalyzer] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        keywords: Optional[ChannelKeywords] = None,
        allowed_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        """초기화

        Args:
            content_analyzer: 본문 분석기
            metadata_extractor: 메타데이터 추출기
            keywords: 키워드 사전
            allowed_root: 허용된 파일 시스템 루트 (보안)
            config_path: 임계값 설정 파일 경로 (None이면 기본 경로)
        """
        # 설정 로드
        config_file = config_path or DEFAULT_CONFIG_PATH
        if config_file.exists():
            self.config = ThresholdConfig.from_yaml(config_file)
            logger.info(f"설정 로드: {config_file}")
        else:
            self.config = ThresholdConfig()
            logger.info("기본 설정 사용")

        self.keywords = keywords or ChannelKeywords()
        self.content = content_analyzer or ContentAnalyzer(
            self.keywords,
            confidence_config=self.config.confidence,
        )
        self.metadata = metadata_extractor or MetadataExtractor()
        self.allowed_root = allowed_root

    def classify(self, yaml_path: Path) -> ClassificationResult:
        """YAML 파일 분류

        Args:
            yaml_path: YAML 파일 경로

        Returns:
            분류 결과
        """
        # 보안: 경로 검증
        if not self._validate_path(yaml_path):
            logger.warning(f"경로 검증 실패: {yaml_path}")
            return ClassificationResult(
                channel=None,
                confidence=0.0,
                route="unclassified",
                sources={"error": "path_validation_failed"},
            )

        # 보안: 파일 크기 검증
        if not self._validate_file_size(yaml_path):
            logger.warning(f"파일 크기 초과: {yaml_path}")
            return ClassificationResult(
                channel=None,
                confidence=0.0,
                route="unclassified",
                sources={"error": "file_too_large"},
            )

        # YAML 로드
        yaml_data = self._load_yaml(yaml_path)
        if not yaml_data:
            return ClassificationResult(
                channel=None,
                confidence=0.0,
                route="unclassified",
                sources={"error": "yaml_load_failed"},
            )

        # 각 소스 분석
        content_results = self.content.analyze(yaml_data)
        metadata_result = self.metadata.extract(yaml_data)
        filename_channel = self._analyze_filename(yaml_path.name)

        # 결과 병합
        channel, confidence = self._merge_results(content_results, metadata_result, filename_channel)

        # 라우팅 결정 (채널별 임계값 지원)
        route = self._route(confidence, channel)

        return ClassificationResult(
            channel=channel,
            confidence=confidence,
            route=route,
            sources={
                "content": [
                    {
                        "channel": r.channel,
                        "confidence": r.confidence,
                        "location": r.location,
                    }
                    for r in content_results
                ],
                "metadata": {
                    "channel_hint": metadata_result.channel_hint,
                    "quality": metadata_result.quality_score,
                },
                "filename": filename_channel,
            },
        )

    def _validate_path(self, path: Path) -> bool:
        """경로 검증 (경로 조작 방지)

        Args:
            path: 검증할 경로

        Returns:
            유효 여부
        """
        if self.allowed_root is None:
            return True

        try:
            resolved = path.resolve()
            allowed_resolved = self.allowed_root.resolve()
            return resolved.is_relative_to(allowed_resolved)
        except (ValueError, RuntimeError):
            return False

    def _validate_file_size(self, path: Path) -> bool:
        """파일 크기 검증

        Args:
            path: 검증할 파일 경로

        Returns:
            크기 제한 이내 여부
        """
        try:
            return path.stat().st_size <= self.MAX_FILE_SIZE
        except OSError:
            return False

    def _load_yaml(self, yaml_path: Path) -> Optional[Dict[str, Any]]:
        """YAML 파일 로드

        Args:
            yaml_path: YAML 파일 경로

        Returns:
            YAML 데이터 또는 None
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"YAML 로드 실패: {yaml_path} - {e}")
            return None

    def _analyze_filename(self, filename: str) -> Optional[str]:
        """파일명에서 채널 분석

        Args:
            filename: 파일명

        Returns:
            채널명 또는 None
        """
        import re

        for channel in self.keywords.get_all_channels():
            primary = self.keywords.get_primary_keywords(channel)
            for kw in primary:
                # 괄호 내 키워드 검색: (제휴), （제휴）
                pattern = rf"[\(（]{re.escape(kw)}[\)）]"
                if re.search(pattern, filename, re.IGNORECASE):
                    return channel
        return None

    def _merge_results(
        self,
        content_results: List[ContentAnalysisResult],
        metadata_result: MetadataResult,
        filename_channel: Optional[str],
    ) -> Tuple[Optional[str], float]:
        """결과 병합 및 최종 채널 결정

        Args:
            content_results: 본문 분석 결과
            metadata_result: 메타데이터 결과
            filename_channel: 파일명 채널

        Returns:
            (채널명, 신뢰도)
        """
        candidates: Dict[str, float] = {}

        weights = self.config.weights

        # 본문 분석 결과 반영
        for result in content_results:
            channel = result.channel
            score = result.confidence * weights["content"]
            candidates[channel] = candidates.get(channel, 0.0) + score

        # 메타데이터 결과 반영
        if metadata_result.channel_hint:
            channel = metadata_result.channel_hint
            score = metadata_result.quality_score * weights["metadata"]
            candidates[channel] = candidates.get(channel, 0.0) + score

        # 파일명 결과 반영
        if filename_channel:
            score = 1.0 * weights["filename"]  # 파일명 매칭은 high confidence
            candidates[filename_channel] = candidates.get(filename_channel, 0.0) + score

        if not candidates:
            return None, 0.0

        # 충돌 해결
        return self._resolve_conflict(list(candidates.items()))

    def _resolve_conflict(self, candidates: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
        """채널 충돌 해결 (우선순위 기반)

        동점인 경우 우선순위가 높은 채널 선택

        Args:
            candidates: (채널명, 점수) 리스트

        Returns:
            (채널명, 신뢰도)
        """
        if not candidates:
            return None, 0.0

        # 점수 내림차순, 우선순위 오름차순 정렬
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (-x[1], self.keywords.get_priority(x[0])),
        )

        best_channel, best_score = sorted_candidates[0]
        return best_channel, min(best_score, 1.0)

    def _route(self, confidence: float, channel: Optional[str] = None) -> str:
        """신뢰도 기반 라우팅 결정

        Args:
            confidence: 신뢰도 점수
            channel: 채널명 (채널별 임계값 적용용)

        Returns:
            "auto" | "review" | "unclassified"
        """
        # 채널별 임계값 오버라이드 확인
        auto_threshold = self.config.auto
        review_threshold = self.config.review

        if channel and channel in self.config.channels:
            ch_config = self.config.channels[channel]
            auto_threshold = ch_config.get("auto", auto_threshold)
            review_threshold = ch_config.get("review", review_threshold)

        if confidence >= auto_threshold:
            return "auto"
        elif confidence >= review_threshold:
            return "review"
        else:
            return "unclassified"

    def classify_batch(
        self,
        yaml_files: List[Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, List[ClassificationResult]]:
        """배치 분류

        여러 YAML 파일을 분류하고 라우트별로 그룹화한다.

        Args:
            yaml_files: 분류할 YAML 파일 목록
            output_dir: 결과 저장 디렉토리 (None이면 저장 안함)
            progress_callback: 진행 상황 콜백 (current, total) -> None

        Returns:
            라우트별 분류 결과 {"auto": [...], "review": [...], "unclassified": [...]}
        """
        results: Dict[str, List[ClassificationResult]] = {
            "auto": [],
            "review": [],
            "unclassified": [],
        }

        total = len(yaml_files)
        for i, yaml_path in enumerate(yaml_files):
            try:
                result = self.classify(yaml_path)
                result.sources["path"] = str(yaml_path)
                results[result.route].append(result)
            except Exception as e:
                logger.error(f"분류 실패: {yaml_path} - {e}")
                results["unclassified"].append(
                    ClassificationResult(
                        channel=None,
                        confidence=0.0,
                        route="unclassified",
                        sources={"path": str(yaml_path), "error": str(e)},
                    )
                )

            if progress_callback:
                progress_callback(i + 1, total)

        # 결과 저장
        if output_dir:
            self._save_batch_results(results, output_dir)

        return results

    def _save_batch_results(self, results: Dict[str, List[ClassificationResult]], output_dir: Path) -> None:
        """배치 결과 저장

        Args:
            results: 라우트별 분류 결과
            output_dir: 출력 디렉토리
        """
        import json
        from datetime import datetime

        output_dir.mkdir(parents=True, exist_ok=True)

        # 요약 저장
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "auto_threshold": self.config.auto,
                "review_threshold": self.config.review,
            },
            "counts": {route: len(items) for route, items in results.items()},
            "total": sum(len(items) for items in results.values()),
        }

        summary_path = output_dir / "classification_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 라우트별 상세 결과 저장
        for route, items in results.items():
            if not items:
                continue

            route_data = [
                {
                    "path": r.sources.get("path"),
                    "channel": r.channel,
                    "confidence": r.confidence,
                    "sources": r.sources,
                }
                for r in items
            ]

            route_path = output_dir / f"classified_{route}.json"
            with open(route_path, "w", encoding="utf-8") as f:
                json.dump(route_data, f, ensure_ascii=False, indent=2)

        logger.info(f"배치 결과 저장: {output_dir}")

    def get_statistics(self, results: Dict[str, List[ClassificationResult]]) -> Dict[str, Any]:
        """분류 통계 계산

        Args:
            results: classify_batch 결과

        Returns:
            통계 딕셔너리
        """
        from collections import Counter

        total = sum(len(items) for items in results.values())
        if total == 0:
            return {"error": "결과 없음"}

        # 라우트별 비율
        route_stats = {
            route: {
                "count": len(items),
                "percentage": len(items) / total * 100,
            }
            for route, items in results.items()
        }

        # 채널별 분포 (auto + review만)
        channel_counter: Counter = Counter()
        confidence_sum: Dict[str, float] = {}
        confidence_count: Dict[str, int] = {}

        for route in ["auto", "review"]:
            for result in results[route]:
                if result.channel:
                    channel_counter[result.channel] += 1
                    confidence_sum[result.channel] = confidence_sum.get(result.channel, 0.0) + result.confidence
                    confidence_count[result.channel] = confidence_count.get(result.channel, 0) + 1

        channel_stats = {
            channel: {
                "count": count,
                "percentage": count / total * 100,
                "avg_confidence": (confidence_sum.get(channel, 0) / confidence_count.get(channel, 1)),
            }
            for channel, count in channel_counter.most_common()
        }

        return {
            "total": total,
            "routes": route_stats,
            "channels": channel_stats,
            "config": {
                "auto_threshold": self.config.auto,
                "review_threshold": self.config.review,
            },
        }
