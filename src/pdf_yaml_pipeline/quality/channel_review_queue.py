# SPDX-License-Identifier: MIT
"""채널 분류 수동 검토 큐

혼합 채널이나 저신뢰도 분류 결과를 수동 검토 큐에 저장합니다.

Redis 키 구조:
- channel:review:queue (List): 검토 대기 항목
- channel:review:resolved (Hash): 검토 완료 항목
- channel:review:stats (Hash): 통계

사용법:
    from pipeline.quality.channel_review_queue import ChannelReviewQueue

    queue = ChannelReviewQueue()
    queue.enqueue("file.yaml", result)
    items = queue.dequeue(limit=10)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis

    from .channel_classifier import ChannelResult

logger = logging.getLogger(__name__)

# Redis 키 접두사
KEY_PREFIX = "channel:review"
KEY_QUEUE = f"{KEY_PREFIX}:queue"
KEY_QUEUE_SET = f"{KEY_PREFIX}:queue:set"
KEY_RESOLVED = f"{KEY_PREFIX}:resolved"
KEY_STATS = f"{KEY_PREFIX}:stats"


@dataclass
class ReviewItem:
    """검토 대기 항목"""

    file_path: str
    channel: str
    confidence: float
    candidates: list[str]
    mixed: bool
    enqueued_at: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "channel": self.channel,
            "confidence": self.confidence,
            "candidates": self.candidates,
            "mixed": self.mixed,
            "enqueued_at": self.enqueued_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewItem:
        return cls(
            file_path=data["file_path"],
            channel=data["channel"],
            confidence=data["confidence"],
            candidates=data.get("candidates", []),
            mixed=data.get("mixed", False),
            enqueued_at=data.get("enqueued_at", ""),
            source=data.get("source", "unknown"),
        )


@dataclass
class QueueStats:
    """큐 통계"""

    pending: int
    resolved: int
    total_enqueued: int
    total_resolved: int
    avg_resolution_time_hours: float | None


class ChannelReviewQueue:
    """채널 분류 수동 검토 큐"""

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        max_size: int = 1000,
        ttl_hours: int = 168,
    ) -> None:
        """초기화

        Args:
            redis_client: Redis 클라이언트. None이면 새로 생성.
            max_size: 최대 큐 크기
            ttl_hours: 항목 만료 시간 (시간)
        """
        self._redis: redis.Redis | None = redis_client
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600

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

    def enqueue(
        self,
        file_path: str | Path,
        result: ChannelResult,
    ) -> bool:
        """검토 큐에 항목 추가

        Args:
            file_path: 파일 경로
            result: 채널 분류 결과

        Returns:
            True: 큐에 추가됨
            False: 큐가 가득 차거나 이미 존재
        """
        try:
            file_path_str = str(file_path)

            # 중복 체크
            if self.redis.sismember(KEY_QUEUE_SET, file_path_str):
                logger.debug(f"이미 큐에 존재: {file_path_str}")
                return False

            # 큐 크기 체크
            current_size = self.redis.llen(KEY_QUEUE)
            if current_size >= self.max_size:
                logger.warning(f"큐가 가득 참: {current_size}/{self.max_size}")
                return False

            item = ReviewItem(
                file_path=file_path_str,
                channel=result.channel,
                confidence=result.confidence,
                candidates=result.candidates,
                mixed=result.mixed,
                enqueued_at=datetime.now().isoformat(),
                source=result.source,
            )

            # 트랜잭션으로 추가
            pipe = self.redis.pipeline()
            pipe.rpush(KEY_QUEUE, json.dumps(item.to_dict(), ensure_ascii=False))
            pipe.sadd(KEY_QUEUE_SET, file_path_str)
            pipe.hincrby(KEY_STATS, "total_enqueued", 1)
            pipe.execute()

            logger.info(f"검토 큐 추가: {file_path_str} (channel={result.channel}, confidence={result.confidence})")
            return True

        except Exception as e:
            logger.exception(f"검토 큐 추가 실패: {e}")
            return False

    def dequeue(self, limit: int = 10) -> list[ReviewItem]:
        """검토 큐에서 항목 가져오기 (FIFO)

        Args:
            limit: 가져올 최대 개수

        Returns:
            ReviewItem 목록
        """
        try:
            items: list[ReviewItem] = []

            for _ in range(limit):
                raw = self.redis.lpop(KEY_QUEUE)
                if raw is None:
                    break

                data = json.loads(raw)
                item = ReviewItem.from_dict(data)

                # 중복 세트에서도 제거
                self.redis.srem(KEY_QUEUE_SET, item.file_path)

                items.append(item)

            return items

        except Exception as e:
            logger.exception(f"검토 큐 읽기 실패: {e}")
            return []

    def peek(self, limit: int = 10) -> list[ReviewItem]:
        """검토 큐 미리보기 (제거하지 않음)

        Args:
            limit: 가져올 최대 개수

        Returns:
            ReviewItem 목록
        """
        try:
            raw_items = self.redis.lrange(KEY_QUEUE, 0, limit - 1)
            return [ReviewItem.from_dict(json.loads(raw)) for raw in raw_items]

        except Exception as e:
            logger.exception(f"검토 큐 미리보기 실패: {e}")
            return []

    def resolve(
        self,
        file_path: str | Path,
        channel: str,
        reviewer: str,
        notes: str = "",
    ) -> bool:
        """검토 완료 처리

        Args:
            file_path: 파일 경로
            channel: 최종 결정된 채널
            reviewer: 검토자 ID
            notes: 메모

        Returns:
            True: 성공
        """
        try:
            file_path_str = str(file_path)

            resolution = {
                "channel": channel,
                "reviewer": reviewer,
                "resolved_at": datetime.now().isoformat(),
                "notes": notes,
            }

            pipe = self.redis.pipeline()
            pipe.hset(KEY_RESOLVED, file_path_str, json.dumps(resolution, ensure_ascii=False))
            pipe.hincrby(KEY_STATS, "total_resolved", 1)
            # 큐에서 제거 (이미 dequeue로 제거되었을 수 있음)
            pipe.srem(KEY_QUEUE_SET, file_path_str)
            pipe.execute()

            logger.info(f"검토 완료: {file_path_str} → {channel} (by {reviewer})")
            return True

        except Exception as e:
            logger.exception(f"검토 완료 처리 실패: {e}")
            return False

    def get_resolution(self, file_path: str | Path) -> dict[str, Any] | None:
        """검토 완료 결과 조회"""
        try:
            raw = self.redis.hget(KEY_RESOLVED, str(file_path))
            if raw:
                return json.loads(raw)
            return None

        except Exception:
            return None

    def get_stats(self) -> QueueStats:
        """큐 통계 조회"""
        try:
            pending = self.redis.llen(KEY_QUEUE)
            resolved = self.redis.hlen(KEY_RESOLVED)

            stats = self.redis.hgetall(KEY_STATS)
            total_enqueued = int(stats.get("total_enqueued", 0))
            total_resolved = int(stats.get("total_resolved", 0))

            return QueueStats(
                pending=pending,
                resolved=resolved,
                total_enqueued=total_enqueued,
                total_resolved=total_resolved,
                avg_resolution_time_hours=None,  # TODO: 계산 구현
            )

        except Exception as e:
            logger.exception(f"통계 조회 실패: {e}")
            return QueueStats(
                pending=0,
                resolved=0,
                total_enqueued=0,
                total_resolved=0,
                avg_resolution_time_hours=None,
            )

    def export_for_labeling(self, output_path: Path | str) -> int:
        """라벨링용 데이터 내보내기

        Args:
            output_path: 출력 파일 경로 (JSON)

        Returns:
            내보낸 항목 수
        """
        try:
            items = self.peek(limit=self.max_size)

            export_data = [item.to_dict() for item in items]

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"라벨링 데이터 내보내기: {len(items)}건 → {output_path}")
            return len(items)

        except Exception as e:
            logger.exception(f"라벨링 데이터 내보내기 실패: {e}")
            return 0

    def should_enqueue(self, result: ChannelResult, config: dict[str, Any] | None = None) -> bool:
        """자동 검토 큐 전송 여부 판단

        Args:
            result: 채널 분류 결과
            config: 자동 전송 조건 설정

        Returns:
            True: 검토 큐로 전송 필요
        """
        if config is None:
            # 기본 조건
            return result.mixed or result.confidence < 0.5 or result.channel == "일반"

        for condition in config.get("auto_enqueue_when", []):
            if "confidence_below" in condition:
                if result.confidence < condition["confidence_below"]:
                    return True
            if "mixed_channels" in condition:
                if result.mixed and condition["mixed_channels"]:
                    return True
            if "no_match" in condition:
                if result.channel == "일반" and condition["no_match"]:
                    return True

        return False
