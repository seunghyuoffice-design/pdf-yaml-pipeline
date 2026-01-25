"""Reprocess strategy module.

재처리 전략 결정 및 이력 관리.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .models import (
    GateResult,
    ReprocessDecision,
    ReprocessHistory,
    ReprocessLevel,
    ReprocessReason,
)


class ReprocessStrategy:
    """재처리 전략 결정.

    품질 게이트 실패 시 재처리 수준, 우선순위, 에스컬레이션 결정.
    """

    MAX_ATTEMPTS = 3

    # 사유별 재처리 수준 매핑
    REASON_TO_LEVEL: dict[ReprocessReason, ReprocessLevel] = {
        ReprocessReason.R001: ReprocessLevel.FULL,  # low_fill_rate → 전체 재처리
        ReprocessReason.R002: ReprocessLevel.OCR_ONLY,  # low_avg_reliability → OCR만
        ReprocessReason.R003: ReprocessLevel.STRUCTURE_ONLY,  # header_detection_failed
        ReprocessReason.R004: ReprocessLevel.STRUCTURE_ONLY,  # structure_validation_failed
        ReprocessReason.R005: ReprocessLevel.OCR_ONLY,  # ocr_bbox_missing
        ReprocessReason.R006: ReprocessLevel.STRUCTURE_ONLY,  # merged_cell_error
        ReprocessReason.R007: ReprocessLevel.OCR_ONLY,  # pattern_validation_failed
        ReprocessReason.R008: ReprocessLevel.FULL,  # excessive_low_conf_cells
    }

    # 사유별 우선순위 (1: 최고, 5: 최저)
    REASON_PRIORITY: dict[ReprocessReason, int] = {
        ReprocessReason.R001: 2,  # low_fill_rate
        ReprocessReason.R002: 3,  # low_avg_reliability
        ReprocessReason.R003: 2,  # header_detection_failed
        ReprocessReason.R004: 1,  # structure_validation_failed (심각)
        ReprocessReason.R005: 3,  # ocr_bbox_missing
        ReprocessReason.R006: 2,  # merged_cell_error
        ReprocessReason.R007: 4,  # pattern_validation_failed (경미)
        ReprocessReason.R008: 1,  # excessive_low_conf_cells (심각)
    }

    def decide(
        self,
        gate_result: GateResult,
        history: Optional[ReprocessHistory] = None,
    ) -> ReprocessDecision:
        """재처리 여부 및 수준 결정.

        Args:
            gate_result: 품질 게이트 결과
            history: 이전 재처리 이력 (있으면)

        Returns:
            ReprocessDecision: 재처리 결정
        """
        # 이미 통과한 경우
        if gate_result.passed:
            return ReprocessDecision(
                should_reprocess=False,
                level=ReprocessLevel.FULL,
                reasons=[],
                priority=5,
            )

        # 이력 확인 - 최대 시도 횟수 초과
        if history and self.should_escalate_to_manual(history):
            return ReprocessDecision(
                should_reprocess=True,
                level=ReprocessLevel.MANUAL,
                reasons=self._parse_reasons(gate_result.reasons),
                priority=1,
                max_attempts=0,  # 더 이상 자동 재처리 안 함
            )

        # 사유 파싱
        reasons = self._parse_reasons(gate_result.reasons)

        if not reasons:
            # 알 수 없는 사유
            return ReprocessDecision(
                should_reprocess=True,
                level=ReprocessLevel.FULL,
                reasons=[],
                priority=3,
            )

        # 재처리 수준 결정 (가장 높은 수준 선택)
        level = self._determine_level(reasons)

        # 우선순위 결정 (가장 높은 우선순위 선택)
        priority = self._determine_priority(reasons)

        return ReprocessDecision(
            should_reprocess=True,
            level=level,
            reasons=reasons,
            priority=priority,
            max_attempts=self.MAX_ATTEMPTS,
        )

    def should_escalate_to_manual(self, history: ReprocessHistory) -> bool:
        """수동 검토 에스컬레이션 판단.

        Args:
            history: 재처리 이력

        Returns:
            에스컬레이션 필요 여부
        """
        return history.attempts >= self.MAX_ATTEMPTS

    def update_history(
        self,
        history: Optional[ReprocessHistory],
        file_id: str,
        decision: ReprocessDecision,
        outcome: str,
    ) -> ReprocessHistory:
        """재처리 이력 업데이트.

        Args:
            history: 기존 이력 (없으면 새로 생성)
            file_id: 파일 ID
            decision: 재처리 결정
            outcome: 결과 (success, failed, skipped 등)

        Returns:
            업데이트된 이력
        """
        if history is None:
            history = ReprocessHistory(
                file_id=file_id,
                attempts=0,
                reasons=[],
                outcomes=[],
            )

        history.attempts += 1
        history.last_attempt = datetime.now()
        history.reasons = decision.reasons
        history.outcomes.append(outcome)

        # 이력 크기 제한 (최근 10개만)
        if len(history.outcomes) > 10:
            history.outcomes = history.outcomes[-10:]

        return history

    def _parse_reasons(self, reason_strings: list[str]) -> list[ReprocessReason]:
        """문자열 사유를 Enum으로 파싱.

        Args:
            reason_strings: 사유 문자열 리스트

        Returns:
            ReprocessReason 리스트
        """
        reasons = []
        for s in reason_strings:
            # "low_fill_rate (0.45 < 0.60)" 형태에서 키워드 추출
            for reason in ReprocessReason:
                if reason.value in s.lower():
                    reasons.append(reason)
                    break
        return reasons

    def _determine_level(self, reasons: list[ReprocessReason]) -> ReprocessLevel:
        """재처리 수준 결정.

        여러 사유가 있으면 가장 높은 수준 선택.
        FULL > STRUCTURE_ONLY > OCR_ONLY

        Args:
            reasons: 사유 리스트

        Returns:
            ReprocessLevel: 재처리 수준
        """
        level_order = [
            ReprocessLevel.FULL,
            ReprocessLevel.STRUCTURE_ONLY,
            ReprocessLevel.OCR_ONLY,
        ]

        best_level = ReprocessLevel.OCR_ONLY

        for reason in reasons:
            level = self.REASON_TO_LEVEL.get(reason, ReprocessLevel.FULL)
            if level_order.index(level) < level_order.index(best_level):
                best_level = level

        return best_level

    def _determine_priority(self, reasons: list[ReprocessReason]) -> int:
        """우선순위 결정.

        여러 사유가 있으면 가장 높은 우선순위 (낮은 숫자) 선택.

        Args:
            reasons: 사유 리스트

        Returns:
            우선순위 (1-5)
        """
        if not reasons:
            return 3  # 기본 중간 우선순위

        priorities = [self.REASON_PRIORITY.get(r, 3) for r in reasons]
        return min(priorities)


def get_reprocess_reason_description(reason: ReprocessReason) -> str:
    """재처리 사유 한글 설명.

    Args:
        reason: 재처리 사유

    Returns:
        한글 설명
    """
    descriptions = {
        ReprocessReason.R001: "채움률 미달",
        ReprocessReason.R002: "평균 신뢰도 미달",
        ReprocessReason.R003: "헤더 감지 실패",
        ReprocessReason.R004: "구조 검증 실패",
        ReprocessReason.R005: "OCR 박스 좌표 누락",
        ReprocessReason.R006: "병합 셀 오류",
        ReprocessReason.R007: "패턴 검증 실패",
        ReprocessReason.R008: "저신뢰 셀 과다",
    }
    return descriptions.get(reason, "알 수 없는 사유")


__all__ = [
    "ReprocessStrategy",
    "get_reprocess_reason_description",
]
