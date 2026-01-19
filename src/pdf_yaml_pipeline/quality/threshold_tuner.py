# SPDX-License-Identifier: MIT
"""임계값 튜닝 및 분류 성능 평가 도구

Academy 근거:
- Optimal Threshold Selection (Lachiche & Flach, 2003)
- ROC Analysis (Fawcett, 2006)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .ensemble_classifier import EnsembleClassifier, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """평가 메트릭"""

    precision: float
    recall: float
    f1: float
    support: int  # 샘플 수


@dataclass
class ConfusionMatrix:
    """혼동 행렬"""

    labels: List[str]
    matrix: List[List[int]]  # [actual][predicted]

    def get_count(self, actual: str, predicted: str) -> int:
        """특정 셀 값 조회"""
        try:
            i = self.labels.index(actual)
            j = self.labels.index(predicted)
            return self.matrix[i][j]
        except (ValueError, IndexError):
            return 0


@dataclass
class EvaluationReport:
    """분류 평가 리포트"""

    overall: EvaluationMetrics
    per_channel: Dict[str, EvaluationMetrics]
    confusion: ConfusionMatrix
    misclassified: List[Dict[str, Any]] = field(default_factory=list)
    threshold_used: float = 0.0


@dataclass
class TuningResult:
    """튜닝 결과"""

    best_threshold: float
    best_f1: float
    threshold_scores: Dict[float, float]  # threshold -> f1
    recommendation: str


class ClassificationEvaluator:
    """분류 성능 평가기

    레이블 데이터와 예측 결과를 비교하여
    precision, recall, F1 스코어를 계산한다.
    """

    def evaluate(
        self,
        predictions: List[Tuple[Path, ClassificationResult]],
        ground_truth: Dict[Path, str],
        threshold: float = 0.6,
    ) -> EvaluationReport:
        """분류 성능 평가

        Args:
            predictions: [(파일경로, 분류결과)] 리스트
            ground_truth: {파일경로: 정답채널} 딕셔너리
            threshold: auto 분류 임계값

        Returns:
            평가 리포트
        """
        # 채널별 TP, FP, FN 집계
        tp: Dict[str, int] = defaultdict(int)
        fp: Dict[str, int] = defaultdict(int)
        fn: Dict[str, int] = defaultdict(int)

        # 혼동 행렬용 데이터
        all_channels = set(ground_truth.values())
        all_channels.add("unclassified")
        labels = sorted(all_channels)

        confusion_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        misclassified = []

        for path, result in predictions:
            if path not in ground_truth:
                continue

            actual = ground_truth[path]
            # 임계값 적용하여 예측 결정
            if result.confidence >= threshold and result.channel:
                predicted = result.channel
            else:
                predicted = "unclassified"

            confusion_counts[(actual, predicted)] += 1

            if actual == predicted:
                tp[actual] += 1
            else:
                fp[predicted] += 1
                fn[actual] += 1
                misclassified.append(
                    {
                        "path": str(path),
                        "actual": actual,
                        "predicted": predicted,
                        "confidence": result.confidence,
                    }
                )

        # 채널별 메트릭 계산
        per_channel: Dict[str, EvaluationMetrics] = {}
        for channel in labels:
            if channel == "unclassified":
                continue
            precision = tp[channel] / (tp[channel] + fp[channel]) if (tp[channel] + fp[channel]) > 0 else 0.0
            recall = tp[channel] / (tp[channel] + fn[channel]) if (tp[channel] + fn[channel]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp[channel] + fn[channel]

            per_channel[channel] = EvaluationMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )

        # 전체 메트릭 (macro average)
        if per_channel:
            avg_precision = sum(m.precision for m in per_channel.values()) / len(per_channel)
            avg_recall = sum(m.recall for m in per_channel.values()) / len(per_channel)
            avg_f1 = sum(m.f1 for m in per_channel.values()) / len(per_channel)
            total_support = sum(m.support for m in per_channel.values())
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
            total_support = 0

        overall = EvaluationMetrics(
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
            support=total_support,
        )

        # 혼동 행렬 생성
        matrix = [[0] * len(labels) for _ in labels]
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                matrix[i][j] = confusion_counts[(actual, predicted)]

        confusion = ConfusionMatrix(labels=labels, matrix=matrix)

        return EvaluationReport(
            overall=overall,
            per_channel=per_channel,
            confusion=confusion,
            misclassified=misclassified[:50],  # 최대 50개
            threshold_used=threshold,
        )


class ThresholdTuner:
    """임계값 튜닝 도구

    다양한 임계값에서 F1 스코어를 측정하여
    최적 임계값을 찾는다.
    """

    def __init__(self, classifier: Optional[EnsembleClassifier] = None):
        """초기화

        Args:
            classifier: 앙상블 분류기 (None이면 기본 생성)
        """
        self.classifier = classifier or EnsembleClassifier()
        self.evaluator = ClassificationEvaluator()

    def tune(
        self,
        yaml_files: List[Path],
        labels: Dict[Path, str],
        threshold_range: Tuple[float, float] = (0.4, 0.8),
        step: float = 0.05,
    ) -> TuningResult:
        """최적 임계값 탐색

        Args:
            yaml_files: 테스트 YAML 파일 목록
            labels: {파일경로: 정답채널} 딕셔너리
            threshold_range: 탐색할 임계값 범위 (min, max)
            step: 탐색 단위

        Returns:
            튜닝 결과
        """
        # 모든 파일 분류
        predictions: List[Tuple[Path, ClassificationResult]] = []
        for path in yaml_files:
            if path in labels:
                result = self.classifier.classify(path)
                predictions.append((path, result))

        logger.info(f"분류 완료: {len(predictions)}개 파일")

        # 각 임계값에서 F1 측정
        threshold_scores: Dict[float, float] = {}
        current = threshold_range[0]

        while current <= threshold_range[1]:
            report = self.evaluator.evaluate(predictions, labels, threshold=current)
            threshold_scores[round(current, 2)] = report.overall.f1
            current += step

        # 최적 임계값 찾기
        best_threshold = max(threshold_scores, key=threshold_scores.get)  # type: ignore
        best_f1 = threshold_scores[best_threshold]

        # 권장사항 생성
        if best_f1 >= 0.8:
            recommendation = f"권장: {best_threshold} (F1={best_f1:.3f}, 우수)"
        elif best_f1 >= 0.6:
            recommendation = f"권장: {best_threshold} (F1={best_f1:.3f}, 양호)"
        else:
            recommendation = f"권장: {best_threshold} (F1={best_f1:.3f}, 개선 필요 - 키워드 확장 검토)"

        return TuningResult(
            best_threshold=best_threshold,
            best_f1=best_f1,
            threshold_scores=threshold_scores,
            recommendation=recommendation,
        )

    def compare_thresholds(
        self,
        yaml_files: List[Path],
        labels: Dict[Path, str],
        old_threshold: float,
        new_threshold: float,
    ) -> Dict[str, Any]:
        """이전/이후 임계값 비교

        Args:
            yaml_files: 테스트 YAML 파일 목록
            labels: 정답 레이블
            old_threshold: 이전 임계값
            new_threshold: 새 임계값

        Returns:
            비교 결과
        """
        predictions = [(p, self.classifier.classify(p)) for p in yaml_files if p in labels]

        old_report = self.evaluator.evaluate(predictions, labels, threshold=old_threshold)
        new_report = self.evaluator.evaluate(predictions, labels, threshold=new_threshold)

        return {
            "old": {
                "threshold": old_threshold,
                "f1": old_report.overall.f1,
                "precision": old_report.overall.precision,
                "recall": old_report.overall.recall,
            },
            "new": {
                "threshold": new_threshold,
                "f1": new_report.overall.f1,
                "precision": new_report.overall.precision,
                "recall": new_report.overall.recall,
            },
            "improvement": {
                "f1": new_report.overall.f1 - old_report.overall.f1,
                "precision": new_report.overall.precision - old_report.overall.precision,
                "recall": new_report.overall.recall - old_report.overall.recall,
            },
        }
