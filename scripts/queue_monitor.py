"""Queue Monitor Script.

Redis 큐 상태를 실시간으로 모니터링합니다.

사용법:
    # Docker 내부
    docker-compose -f docker-compose.pipeline.yml --profile monitor run queue-monitor

    # 직접 실행
    python -m scripts.queue_monitor
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

try:
    import redis
except ImportError:
    print("ERROR: redis 패키지가 필요합니다. pip install redis")
    sys.exit(1)

# Redis 키
KEY_QUEUE = "file:queue"
KEY_PROCESSING = "file:processing"
KEY_DONE = "file:done"
KEY_FAILED = "file:failed"


def format_duration(seconds: float) -> str:
    """초를 읽기 좋은 형식으로 변환."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def estimate_remaining(done: int, total: int, elapsed: float) -> str:
    """남은 시간 추정."""
    if done == 0 or elapsed == 0:
        return "N/A"

    rate = done / elapsed  # files per second
    remaining = total - done
    eta_seconds = remaining / rate

    return format_duration(eta_seconds)


def main():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    print(f"Connecting to Redis at {redis_host}:{redis_port}")
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    try:
        r.ping()
        print("Redis connected\n")
    except redis.ConnectionError as e:
        print(f"ERROR: Failed to connect to Redis: {e}")
        sys.exit(1)

    start_time = time.time()
    initial_done = r.scard(KEY_DONE)

    print("=" * 60)
    print("Pipeline Queue Monitor")
    print("=" * 60)
    print("Press Ctrl+C to exit\n")

    prev_done = initial_done

    try:
        while True:
            queue_len = r.llen(KEY_QUEUE)
            processing_len = r.llen(KEY_PROCESSING)
            done_count = r.scard(KEY_DONE)
            failed_count = r.scard(KEY_FAILED)

            total = queue_len + processing_len + done_count + failed_count
            elapsed = time.time() - start_time
            processed_in_session = done_count - initial_done

            # 처리 속도 계산
            rate = processed_in_session / elapsed if elapsed > 0 else 0
            rate_per_min = rate * 60

            # 남은 시간 추정
            remaining = queue_len + processing_len
            eta = estimate_remaining(processed_in_session, remaining + processed_in_session, elapsed)

            # 진행률
            progress = (done_count / total * 100) if total > 0 else 0

            # 최근 처리량
            delta = done_count - prev_done
            prev_done = done_count

            # 출력
            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"\r[{now}] "
                f"Queue: {queue_len:,} | "
                f"Processing: {processing_len} | "
                f"Done: {done_count:,} | "
                f"Failed: {failed_count} | "
                f"Progress: {progress:.1f}% | "
                f"Rate: {rate_per_min:.2f}/min | "
                f"ETA: {eta} | "
                f"+{delta}",
                end="",
                flush=True,
            )

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped")

        # 최종 통계
        elapsed = time.time() - start_time
        processed = r.scard(KEY_DONE) - initial_done

        print("\n=== Session Summary ===")
        print(f"Duration: {format_duration(elapsed)}")
        print(f"Processed: {processed:,} files")
        if elapsed > 0:
            print(f"Average rate: {processed / elapsed * 60:.2f} files/min")


if __name__ == "__main__":
    main()
