# SPDX-License-Identifier: MIT
"""채널 분류 모니터링

채널 분포를 추적하고 이상 감지 알림을 생성합니다.

Redis 키 구조:
- channel:monitor:distribution (Sorted Set): 채널별 카운트
- channel:monitor:history (List): 최근 분류 이력
- channel:monitor:alerts (List): 이상 감지 알림

사용법:
    from dashboard.services.channel_monitor import ChannelMonitor

    monitor = ChannelMonitor()
    monitor.record_classification("제휴", "header")
    distribution = monitor.get_distribution()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)

# Redis 키 접두사
KEY_PREFIX = "channel:monitor"
KEY_DISTRIBUTION = f"{KEY_PREFIX}:distribution"
KEY_HISTORY = f"{KEY_PREFIX}:history"
KEY_ALERTS = f"{KEY_PREFIX}:alerts"


@dataclass
class AnomalyAlert:
    """이상 감지 알림"""

    channel: str
    expected_ratio: float
    actual_ratio: float
    deviation: float
    timestamp: str
    severity: str  # info | warning | critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "expected_ratio": round(self.expected_ratio, 4),
            "actual_ratio": round(self.actual_ratio, 4),
            "deviation": round(self.deviation, 4),
            "timestamp": self.timestamp,
            "severity": self.severity,
        }


class ChannelMonitor:
    """채널 분류 모니터링"""

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        rolling_window: int = 1000,
        alert_threshold: float = 0.2,
        expected_distribution: dict[str, float] | None = None,
    ) -> None:
        """초기화

        Args:
            redis_client: Redis 클라이언트
            rolling_window: 최근 N건 기준 분포 계산
            alert_threshold: 이상 감지 임계값 (비율)
            expected_distribution: 기대 분포
        """
        self._redis: redis.Redis | None = redis_client
        self.rolling_window = rolling_window
        self.alert_threshold = alert_threshold
        self.expected_distribution = expected_distribution or {
            "제휴": 0.15,
            "TM": 0.10,
            "홈쇼핑": 0.05,
            "콜센터": 0.35,
            "방송": 0.10,
            "일반": 0.25,
        }

    @property
    def redis(self) -> redis.Redis:
        """Redis 클라이언트 반환 (지연 초기화)"""
        if self._redis is None:
            import os

            import redis

            host = os.environ.get("REDIS_HOST", "localhost")
            port = int(os.environ.get("REDIS_PORT", "6379"))
            db = int(os.environ.get("REDIS_DB", "0"))

            self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)

        return self._redis

    def record_classification(
        self,
        channel: str,
        source: str,
        file_path: str | None = None,
    ) -> None:
        """분류 결과 기록

        Args:
            channel: 분류된 채널
            source: 분류 소스 (metadata, header, body, footer, filename)
            file_path: 파일 경로 (선택)
        """
        try:
            timestamp = datetime.now().isoformat()

            # 분포 카운트 증가
            self.redis.zincrby(KEY_DISTRIBUTION, 1, channel)

            # 이력 추가
            history_entry = {
                "channel": channel,
                "source": source,
                "file_path": file_path,
                "timestamp": timestamp,
            }
            self.redis.lpush(KEY_HISTORY, json.dumps(history_entry, ensure_ascii=False))

            # 이력 크기 제한
            self.redis.ltrim(KEY_HISTORY, 0, self.rolling_window - 1)

            # 이상 감지
            self._check_anomaly()

        except Exception as e:
            logger.exception(f"분류 기록 실패: {e}")

    def get_distribution(self) -> dict[str, float]:
        """채널 분포 조회 (비율)

        Returns:
            채널별 비율 딕셔너리
        """
        try:
            # 전체 카운트 조회
            raw = self.redis.zrangebyscore(KEY_DISTRIBUTION, 0, "+inf", withscores=True)

            if not raw:
                return {}

            total = sum(score for _, score in raw)
            if total == 0:
                return {}

            return {channel: count / total for channel, count in raw}

        except Exception as e:
            logger.exception(f"분포 조회 실패: {e}")
            return {}

    def get_rolling_distribution(self) -> dict[str, float]:
        """최근 N건 기준 분포 조회

        Returns:
            채널별 비율 딕셔너리
        """
        try:
            history = self.redis.lrange(KEY_HISTORY, 0, self.rolling_window - 1)

            if not history:
                return {}

            channel_counts: dict[str, int] = {}
            for raw in history:
                entry = json.loads(raw)
                channel = entry.get("channel", "일반")
                channel_counts[channel] = channel_counts.get(channel, 0) + 1

            total = sum(channel_counts.values())
            if total == 0:
                return {}

            return {channel: count / total for channel, count in channel_counts.items()}

        except Exception as e:
            logger.exception(f"롤링 분포 조회 실패: {e}")
            return {}

    def _check_anomaly(self) -> AnomalyAlert | None:
        """이상 감지 수행

        Returns:
            이상 감지 시 AnomalyAlert, 아니면 None
        """
        try:
            current = self.get_rolling_distribution()

            if not current:
                return None

            for channel, expected_ratio in self.expected_distribution.items():
                actual_ratio = current.get(channel, 0.0)
                deviation = abs(actual_ratio - expected_ratio)

                if deviation > self.alert_threshold:
                    severity = "critical" if deviation > self.alert_threshold * 2 else "warning"

                    alert = AnomalyAlert(
                        channel=channel,
                        expected_ratio=expected_ratio,
                        actual_ratio=actual_ratio,
                        deviation=deviation,
                        timestamp=datetime.now().isoformat(),
                        severity=severity,
                    )

                    # 알림 저장
                    self.redis.lpush(KEY_ALERTS, json.dumps(alert.to_dict(), ensure_ascii=False))
                    self.redis.ltrim(KEY_ALERTS, 0, 99)  # 최근 100개만 유지

                    logger.warning(
                        f"채널 이상 감지: {channel} (expected={expected_ratio:.2%}, "
                        f"actual={actual_ratio:.2%}, deviation={deviation:.2%})"
                    )

                    return alert

            return None

        except Exception as e:
            logger.exception(f"이상 감지 실패: {e}")
            return None

    def detect_anomaly(self) -> AnomalyAlert | None:
        """수동 이상 감지 호출"""
        return self._check_anomaly()

    def get_alerts(self, limit: int = 10) -> list[AnomalyAlert]:
        """최근 알림 조회

        Args:
            limit: 조회할 최대 개수

        Returns:
            AnomalyAlert 목록
        """
        try:
            raw_alerts = self.redis.lrange(KEY_ALERTS, 0, limit - 1)

            alerts = []
            for raw in raw_alerts:
                data = json.loads(raw)
                alerts.append(
                    AnomalyAlert(
                        channel=data["channel"],
                        expected_ratio=data["expected_ratio"],
                        actual_ratio=data["actual_ratio"],
                        deviation=data["deviation"],
                        timestamp=data["timestamp"],
                        severity=data.get("severity", "warning"),
                    )
                )

            return alerts

        except Exception as e:
            logger.exception(f"알림 조회 실패: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """모니터링 통계 조회"""
        try:
            total_count = sum(
                int(score) for _, score in self.redis.zrangebyscore(KEY_DISTRIBUTION, 0, "+inf", withscores=True)
            )

            history_count = self.redis.llen(KEY_HISTORY)
            alert_count = self.redis.llen(KEY_ALERTS)

            return {
                "total_classified": total_count,
                "history_size": history_count,
                "pending_alerts": alert_count,
                "rolling_window": self.rolling_window,
                "alert_threshold": self.alert_threshold,
                "distribution": self.get_distribution(),
                "rolling_distribution": self.get_rolling_distribution(),
            }

        except Exception as e:
            logger.exception(f"통계 조회 실패: {e}")
            return {}

    def reset(self) -> None:
        """모니터링 데이터 초기화 (주의: 모든 데이터 삭제)"""
        try:
            self.redis.delete(KEY_DISTRIBUTION)
            self.redis.delete(KEY_HISTORY)
            self.redis.delete(KEY_ALERTS)
            logger.info("채널 모니터링 데이터 초기화")

        except Exception as e:
            logger.exception(f"초기화 실패: {e}")
