"""File Queue Worker for Parallel Pipeline Processing.

파일 단위 병렬 처리를 위한 Redis 기반 워커.

병렬화 원칙:
  - 파일 간 병렬화: ✅ N 워커 분담
  - 페이지 단위 병렬화: ❌ 금지
  - 파일 내부 처리: 단일 워커 + 특약/담보 단위 청크

Redis 키 구조:
  - file:queue       → 처리 대기 파일
  - file:processing  → 현재 처리 중
  - file:done        → 완료
  - file:failed      → 실패 (DLQ)
  - file:lock:{hash} → 파일별 락 (idempotency)
"""

from __future__ import annotations

# ============================================================
# 스레드 수 고정 설정 (import 전에 설정해야 함)
# 12 워커 × 2 스레드 = 24 스레드 = 24 코어
# ============================================================
import os

_THREADS_PER_WORKER = 2  # 워커당 2코어

os.environ["OMP_NUM_THREADS"] = str(_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(_THREADS_PER_WORKER)
os.environ["OPENBLAS_NUM_THREADS"] = str(_THREADS_PER_WORKER)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_THREADS_PER_WORKER)
os.environ["NUMEXPR_NUM_THREADS"] = str(_THREADS_PER_WORKER)


# PyTorch 스레드 설정 (torch import 후 적용)
def _set_torch_threads():
    try:
        import torch

        torch.set_num_threads(_THREADS_PER_WORKER)
        torch.set_num_interop_threads(_THREADS_PER_WORKER)
    except ImportError:
        pass


_set_torch_threads()
# ============================================================

import atexit
import hashlib
import os
import signal
import sys
import threading
import time
import gc
from pathlib import Path
from typing import Optional, Dict

import yaml

try:
    import redis
except ImportError:
    print("ERROR: redis 패키지가 필요합니다. pip install redis")
    sys.exit(1)

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

# Discord 알림 (optional)
try:
    from src.pipeline.notifications.discord_notifier import (
        notify_error as dc_notify_error,
        notify_dlq as dc_notify_dlq,
        notify_milestone as dc_notify_milestone,
        notify_worker_crash as dc_notify_worker_crash,
    )

    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False

    def dc_notify_error(*args, **kwargs):
        pass

    def dc_notify_dlq(*args, **kwargs):
        pass

    def dc_notify_milestone(*args, **kwargs):
        pass

    def dc_notify_worker_crash(*args, **kwargs):
        pass


# 상수
# TTL-TIMEOUT 관계 규정:
#   LOCK_TTL >= TIMEOUT + LOCK_TTL_MARGIN (최소 300초 여유)
#   이 관계가 깨지면 처리 중 락 만료 위험
LOCK_TTL_MARGIN = 300  # 락 여유 시간 (초)
TIMEOUT = int(os.getenv("TIMEOUT", "600"))
LOCK_TTL = max(900, TIMEOUT + LOCK_TTL_MARGIN)  # 락 만료 시간 (최소 900초)
IDLE_LOG_INTERVAL = 300  # 유휴 상태 로그 간격 (5분)
REDIS_CONNECT_MAX_RETRIES = 5  # Redis 연결 최대 재시도 횟수
REDIS_CONNECT_RETRY_DELAY = 5  # Redis 연결 재시도 간격 (초)
MAX_REDIS_ERRORS = 10  # 연속 Redis 에러 허용 횟수
STALE_RECOVERY_INTERVAL = int(os.getenv("STALE_RECOVERY_INTERVAL", "300"))  # processing 복구 주기 (초)

# 대용량 파일 제한 (VRAM 부족 GPU용)
LARGE_FILE_THRESHOLD_MB = 10  # 10MB 이상은 대용량
LOW_VRAM_WORKER_PREFIXES = ("gpu2-", "forge-")  # RTX 3060, RTX 4070 Laptop

# 환경변수
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")  # None if not set
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
SAFE_MODE = os.getenv("SAFE_MODE", "true").lower() == "true"
# TIMEOUT은 위에서 LOCK_TTL 계산용으로 이미 정의됨
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
WORKER_ID = os.getenv("HOSTNAME", f"worker-{os.getpid()}")
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
AUTO_TRIAGE = os.getenv("AUTO_TRIAGE", "true").lower() == "true"

# 파서 캐시 (워커 프로세스 단위, thread-safe, LRU)
# 캐시 상한: 메모리 절약을 위해 최대 4개 파서 인스턴스로 제한
MAX_PARSER_CACHE = 4
_PARSER_CACHE: Dict[str, object] = {}
_PARSER_CACHE_ORDER: list[str] = []  # LRU 순서 추적 (가장 오래된 것이 앞)
_PARSER_CACHE_LOCK = threading.Lock()


def shutdown_parsers() -> None:
    """Close cached parsers to ensure subprocesses are reaped."""
    with _PARSER_CACHE_LOCK:
        parsers = list(_PARSER_CACHE.values())
        _PARSER_CACHE.clear()

    for parser in parsers:
        close = getattr(parser, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:
                logger.warning(f"Failed to close parser cleanly: {e}")


# Redis 키
KEY_QUEUE = "file:queue"
KEY_QUEUE_SET = "file:queue:set"
KEY_PROCESSING = "file:processing"
KEY_PROCESSING_SET = "file:processing:set"
KEY_DONE = "file:done"
KEY_FAILED = "file:failed"
KEY_PROBATION_FAILED = "probation:failed"
KEY_LOCK_PREFIX = "file:lock:"
KEY_RETRY_PREFIX = "file:retry:"
KEY_WORKER_STATUS_PREFIX = "worker:status:"
KEY_FAILED_META_PREFIX = "file:failed:meta:"  # Hash: error_code, error_detail, worker_id, timestamp
KEY_METRICS_TIMES = "metrics:processing_times"  # List: 최근 처리 시간 (초)
KEY_METRICS_SUMMARY = "metrics:summary"  # Hash: 집계 지표

# 메트릭 상수
METRICS_WINDOW_SIZE = 1000  # 최근 N개 처리 시간만 유지
METRICS_SUMMARY_INTERVAL = 60  # 집계 주기 (초)

# 에러 코드 정의
ERROR_CODES = {
    "FILE_NOT_FOUND": "파일이 존재하지 않음",
    "PERMISSION_DENIED": "파일 접근 권한 없음",
    "IO_ERROR": "I/O 오류",
    "PARSE_ERROR": "파싱 실패",
    "TIMEOUT": "처리 시간 초과",
    "LOCK_LOST": "락 만료/연장 실패",
    "MAX_RETRIES": "최대 재시도 횟수 초과",
    "PROBATION_CRASH": "SafeParser 격리 실패",
    "YAML_ERROR": "YAML 직렬화 오류",
    "UNKNOWN": "알 수 없는 오류",
}


# 종료 핸들러
class GracefulShutdownHandler:
    """Graceful shutdown 신호 처리 클래스."""

    def __init__(self):
        self._shutdown = False

    def signal_handler(self, signum, frame):
        """시그널 핸들러."""
        logger.warning(f"Received signal {signum}, shutting down gracefully...")
        self._shutdown = True

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    def register(self):
        """SIGTERM, SIGINT 핸들러 등록."""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)


shutdown_handler = GracefulShutdownHandler()
atexit.register(shutdown_parsers)


def generate_idempotency_key(file_path: Path) -> str:
    """파일 경로 기반 MD5 해시 생성 (idempotency/lock 키용)."""
    return hashlib.md5(str(file_path).encode()).hexdigest()[:12]


def get_lock_key(file_name: str) -> str:
    """파일명으로 Redis 락 키 생성."""
    return f"{KEY_LOCK_PREFIX}{generate_idempotency_key(Path(file_name))}"


def is_already_processed(r: redis.Redis, file_name: str) -> bool:
    """이미 처리 완료된 파일인지 확인."""
    return r.sismember(KEY_DONE, file_name)


def acquire_lock(r: redis.Redis, file_name: str, ttl: int = LOCK_TTL) -> bool:
    """파일 처리 락 획득 (중복 처리 방지)."""
    return r.set(get_lock_key(file_name), WORKER_ID, nx=True, ex=ttl)


def extend_lock(r: redis.Redis, file_name: str, ttl: int = LOCK_TTL) -> bool:
    """파일 처리 락 TTL 연장 (장시간 작업용).

    Args:
        r: Redis 연결
        file_name: 파일명
        ttl: 새로운 TTL (초)

    Returns:
        연장 성공 여부
    """
    try:
        lock_key = get_lock_key(file_name)
        lua_script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('EXPIRE', KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        return r.eval(lua_script, 1, lock_key, WORKER_ID, ttl) > 0
    except redis.RedisError as e:
        logger.warning(f"Failed to extend lock for {file_name}: {e}")
        return False


def release_lock(r: redis.Redis, file_name: str) -> bool:
    """파일 처리 락 해제.

    Returns:
        락 삭제 성공 여부
    """
    try:
        lock_key = get_lock_key(file_name)
        lua_script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """
        return r.eval(lua_script, 1, lock_key, WORKER_ID) > 0
    except redis.RedisError as e:
        logger.warning(f"Failed to release lock for {file_name}: {e}")
        return False


def recover_stale_processing(r: redis.Redis) -> int:
    """processing 리스트에 남은 stale 항목을 큐로 복귀."""
    try:
        processing_items = r.lrange(KEY_PROCESSING, 0, -1)
        recovered = 0
        for file_name in processing_items:
            # 락이 없으면 워커가 죽었다고 판단하고 큐로 복귀
            if not r.exists(get_lock_key(file_name)):
                r.lrem(KEY_PROCESSING, 0, file_name)
                r.srem(KEY_PROCESSING_SET, file_name)
                r.sadd(KEY_QUEUE_SET, file_name)
                r.lpush(KEY_QUEUE, file_name)
                recovered += 1
        if recovered:
            logger.warning(f"Recovered {recovered} stale items from processing")
        return recovered
    except redis.RedisError as e:
        logger.warning(f"Failed stale recovery: {e}")
        return 0


def start_lock_extender(
    r: redis.Redis,
    file_name: str,
    stop_event: threading.Event,
    interval_s: Optional[int] = None,
    max_failures: int = 3,
) -> tuple[threading.Thread, threading.Event]:
    """파일 처리 중 락 TTL을 주기적으로 연장.

    Args:
        r: Redis 연결
        file_name: 파일명
        stop_event: 종료 이벤트
        interval_s: 연장 주기 (기본: LOCK_TTL // 3)
        max_failures: 연속 실패 허용 횟수

    Returns:
        (스레드, 실패 이벤트) - 실패 이벤트가 set되면 재큐잉 필요
    """
    if interval_s is None:
        interval_s = max(1, LOCK_TTL // 3)

    lock_failed_event = threading.Event()
    failure_count = 0

    def _extend_loop():
        nonlocal failure_count
        while not stop_event.wait(interval_s):
            if extend_lock(r, file_name):
                failure_count = 0  # 성공 시 리셋
            else:
                failure_count += 1
                logger.warning(f"Lock extension failed ({failure_count}/{max_failures}): {file_name}")
                if failure_count >= max_failures:
                    logger.error(f"Lock extension max failures reached, signaling requeue: {file_name}")
                    lock_failed_event.set()
                    break

    thread = threading.Thread(
        target=_extend_loop,
        name=f"lock-extender-{WORKER_ID}",
        daemon=True,
    )
    thread.start()
    return thread, lock_failed_event


def store_failure_meta(
    r: redis.Redis,
    file_name: str,
    error_code: str,
    error_detail: str = "",
) -> None:
    """실패 메타데이터 저장.

    Args:
        r: Redis 연결
        file_name: 파일명
        error_code: 에러 코드 (ERROR_CODES 키)
        error_detail: 상세 에러 메시지 (1줄)
    """
    meta_key = f"{KEY_FAILED_META_PREFIX}{file_name}"
    meta = {
        "error_code": error_code,
        "error_detail": error_detail[:500] if error_detail else "",  # 500자 제한
        "worker_id": WORKER_ID,
        "timestamp": int(time.time()),
    }
    try:
        r.hset(meta_key, mapping=meta)
        # 7일 후 자동 삭제 (메타 데이터 정리)
        r.expire(meta_key, 7 * 24 * 3600)
    except redis.RedisError as e:
        logger.warning(f"Failed to store failure meta for {file_name}: {e}")


def handle_failure_atomic(
    r: redis.Redis,
    file_name: str,
    max_retries: int = MAX_RETRIES,
    error_code: str = "UNKNOWN",
    error_detail: str = "",
) -> None:
    """Atomic 재시도/DLQ 처리.

    Lua 스크립트로 race condition 없이 한 번에 처리합니다.

    Args:
        r: Redis 연결
        file_name: 파일명
        max_retries: 최대 재시도 횟수
        error_code: 에러 코드 (ERROR_CODES 키)
        error_detail: 상세 에러 메시지
    """
    file_path = INPUT_DIR / file_name
    if r.sismember(KEY_PROBATION_FAILED, str(file_path.resolve())):
        logger.warning(f"PROBATION failed, direct DLQ: {file_name}")
        r.sadd(KEY_FAILED, file_name)
        r.srem(KEY_QUEUE_SET, file_name)
        store_failure_meta(r, file_name, "PROBATION_CRASH", "SafeParser probation failure")
        return
    retry_key = f"{KEY_RETRY_PREFIX}{file_name}"

    # Lua 스크립트: atomic 증가 + 조건부 이동
    lua_script = """
    local retry_count = redis.call('INCR', KEYS[1])
    if tonumber(retry_count) < tonumber(ARGV[1]) then
        redis.call('SADD', KEYS[2], ARGV[2])
        redis.call('LPUSH', KEYS[3], ARGV[2])
        return 'retry'
    else
        redis.call('SADD', KEYS[4], ARGV[2])
        redis.call('SREM', KEYS[2], ARGV[2])
        return 'failed'
    end
    """

    try:
        result = r.eval(
            lua_script,
            4,
            retry_key,
            KEY_QUEUE_SET,
            KEY_QUEUE,
            KEY_FAILED,
            str(max_retries),
            file_name,
        )
        if result == b"retry" or result == "retry":
            retry_count = get_retry_count(r, file_name)
            logger.warning(f"Retry {retry_count}/{max_retries}: {file_name}")
        else:
            logger.error(f"Max retries exceeded, moving to DLQ: {file_name}")
            store_failure_meta(r, file_name, error_code, error_detail)
            dc_notify_dlq(WORKER_ID, file_name, error_code)
    except redis.RedisError as e:
        # Lua 스크립트 실패 시 폴백
        logger.warning(f"Lua script failed, using fallback: {e}")
        retry_count = increment_retry(r, file_name)
        if retry_count < max_retries:
            logger.warning(f"Retry {retry_count}/{max_retries}: {file_name}")
            r.sadd(KEY_QUEUE_SET, file_name)
            r.lpush(KEY_QUEUE, file_name)
        else:
            logger.error(f"Max retries exceeded, moving to DLQ: {file_name}")
            r.sadd(KEY_FAILED, file_name)
            r.srem(KEY_QUEUE_SET, file_name)
            store_failure_meta(r, file_name, error_code, error_detail)
            dc_notify_dlq(WORKER_ID, file_name, error_code)


def get_retry_count(r: redis.Redis, file_name: str) -> int:
    """재시도 횟수 조회."""
    retry_key = f"{KEY_RETRY_PREFIX}{file_name}"
    count = r.get(retry_key)
    return int(count) if count else 0


def increment_retry(r: redis.Redis, file_name: str) -> int:
    """재시도 횟수 증가."""
    retry_key = f"{KEY_RETRY_PREFIX}{file_name}"
    return r.incr(retry_key)


def report_worker_status(r: redis.Redis, status: str, **kwargs) -> None:
    """워커 상태를 Redis에 보고 (HSET)."""
    key = f"{KEY_WORKER_STATUS_PREFIX}{WORKER_ID}"
    data = {"status": status, "last_heartbeat": int(time.time()), **kwargs}
    # None 값 제거 및 문자열 변환
    data = {k: str(v) for k, v in data.items() if v is not None}

    try:
        r.hset(key, mapping=data)
        # 키 만료 설정 (워커가 죽으면 상태도 결국 사라지게 1시간 TTL)
        r.expire(key, 3600)
    except Exception as e:
        logger.warning(f"Failed to report status: {e}")


def increment_worker_metric(r: redis.Redis, metric: str) -> None:
    """워커 메트릭(에러/재시작) 증가 (HINCRBY)."""
    key = f"{KEY_WORKER_STATUS_PREFIX}{WORKER_ID}"
    try:
        r.hincrby(key, metric, 1)
        r.expire(key, 3600)
    except Exception as e:
        logger.warning(f"Failed to increment metric {metric}: {e}")


def record_processing_time(r: redis.Redis, elapsed_seconds: float) -> None:
    """처리 시간 기록 (메트릭 수집).

    Args:
        r: Redis 연결
        elapsed_seconds: 처리 시간 (초)
    """
    try:
        # 처리 시간 추가 (LPUSH + LTRIM으로 윈도우 유지)
        r.lpush(KEY_METRICS_TIMES, f"{elapsed_seconds:.2f}")
        r.ltrim(KEY_METRICS_TIMES, 0, METRICS_WINDOW_SIZE - 1)
    except redis.RedisError as e:
        logger.debug(f"Failed to record processing time: {e}")


def update_metrics_summary(r: redis.Redis) -> None:
    """메트릭 집계 업데이트.

    최근 처리 시간으로 평균, 95p, 최대값 등 계산.
    """
    try:
        # 최근 처리 시간 가져오기
        times_str = r.lrange(KEY_METRICS_TIMES, 0, METRICS_WINDOW_SIZE - 1)
        if not times_str:
            return

        times = [float(t) for t in times_str]
        times.sort()

        count = len(times)
        avg = sum(times) / count
        p95_idx = int(count * 0.95)
        p95 = times[p95_idx] if p95_idx < count else times[-1]
        max_time = times[-1]
        min_time = times[0]

        # 큐/완료/실패 현황
        queue_len = r.llen(KEY_QUEUE)
        done_count = r.scard(KEY_DONE)
        failed_count = r.scard(KEY_FAILED)

        # 실패율 계산
        total_processed = done_count + failed_count
        fail_rate = (failed_count / total_processed * 100) if total_processed > 0 else 0

        summary = {
            "sample_count": count,
            "avg_seconds": f"{avg:.2f}",
            "p95_seconds": f"{p95:.2f}",
            "max_seconds": f"{max_time:.2f}",
            "min_seconds": f"{min_time:.2f}",
            "queue_length": queue_len,
            "done_count": done_count,
            "failed_count": failed_count,
            "fail_rate_pct": f"{fail_rate:.2f}",
            "updated_at": int(time.time()),
        }

        r.hset(KEY_METRICS_SUMMARY, mapping=summary)
        r.expire(KEY_METRICS_SUMMARY, 3600)  # 1시간 TTL

    except redis.RedisError as e:
        logger.debug(f"Failed to update metrics summary: {e}")
    except Exception as e:
        logger.debug(f"Metrics calculation error: {e}")


def connect_redis_with_retry(
    max_retries: int = REDIS_CONNECT_MAX_RETRIES,
) -> redis.Redis:
    """Redis 연결 with exponential backoff.

    Args:
        max_retries: 최대 재시도 횟수

    Returns:
        redis.Redis 인스턴스

    Raises:
        redis.ConnectionError: 모든 재시도 실패 시
    """
    for attempt in range(1, max_retries + 1):
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            r.ping()
            logger.info(f"Redis connected (attempt {attempt}/{max_retries})")
            return r
        except redis.ConnectionError as e:
            if attempt < max_retries:
                delay = REDIS_CONNECT_RETRY_DELAY * attempt  # Exponential backoff
                logger.warning(
                    f"Redis connection failed (attempt {attempt}/{max_retries}): {e}. " f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to connect to Redis after {max_retries} attempts")
                raise


def get_cached_parser(ocr_enabled: bool):
    """워커 프로세스 내 파서 캐시 반환 (thread-safe, LRU).

    SAFE_MODE에 따라 SafeParser 또는 UnifiedParser를 단일 인스턴스로 재사용하여
    모델 로딩/프로세스 스핀업 오버헤드를 줄입니다.

    Double-checked locking 패턴으로 thread-safety 보장.
    LRU 정책으로 MAX_PARSER_CACHE 개수 초과 시 가장 오래된 파서 제거.
    """
    key = f"safe:{ocr_enabled}" if SAFE_MODE else f"fast:{ocr_enabled}"

    # Fast path: 락 없이 먼저 확인
    if key in _PARSER_CACHE:
        # LRU 순서 업데이트 (락 필요)
        with _PARSER_CACHE_LOCK:
            if key in _PARSER_CACHE_ORDER:
                _PARSER_CACHE_ORDER.remove(key)
                _PARSER_CACHE_ORDER.append(key)
        return _PARSER_CACHE[key]

    # Slow path: 락 획득 후 생성
    with _PARSER_CACHE_LOCK:
        # Double-check: 락 획득 사이에 다른 스레드가 생성했을 수 있음
        if key in _PARSER_CACHE:
            if key in _PARSER_CACHE_ORDER:
                _PARSER_CACHE_ORDER.remove(key)
                _PARSER_CACHE_ORDER.append(key)
            return _PARSER_CACHE[key]

        # LRU 제거: 캐시가 상한에 도달하면 가장 오래된 파서 제거
        while len(_PARSER_CACHE) >= MAX_PARSER_CACHE and _PARSER_CACHE_ORDER:
            oldest_key = _PARSER_CACHE_ORDER.pop(0)
            old_parser = _PARSER_CACHE.pop(oldest_key, None)
            if old_parser:
                close = getattr(old_parser, "close", None)
                if callable(close):
                    try:
                        close()
                        logger.debug(f"LRU evicted parser: {oldest_key}")
                    except Exception as e:
                        logger.warning(f"Failed to close evicted parser {oldest_key}: {e}")

        if SAFE_MODE:
            from src.pipeline.parsers.safe_parser import SafeParser

            parser = SafeParser(
                timeout=TIMEOUT,
                ocr_enabled=ocr_enabled,
                dynamic_timeout=False,
            )
        else:
            from src.pipeline.parsers.unified_parser import UnifiedParser

            parser = UnifiedParser(ocr_enabled=ocr_enabled)

        _PARSER_CACHE[key] = parser
        _PARSER_CACHE_ORDER.append(key)
        return parser


def determine_ocr_setting(file_path: Path) -> bool:
    """Triage 기반 OCR 설정 결정.

    디지털 PDF: OCR 불필요 (텍스트 레이어 존재)
    스캔 PDF: OCR 필요 (이미지 기반)

    Args:
        file_path: PDF 파일 경로

    Returns:
        ocr_enabled 값
    """
    if not AUTO_TRIAGE:
        return OCR_ENABLED

    # PDF 파일만 triage
    if file_path.suffix.lower() != ".pdf":
        return OCR_ENABLED

    try:
        from src.pipeline.triage.pdf_classifier import PDFClassifier

        classifier = PDFClassifier()
        result = classifier.classify_pdf(file_path)

        if result.needs_ocr:
            logger.debug(f"Triage: {file_path.name} → scanned (OCR enabled)")
            return True
        else:
            logger.debug(f"Triage: {file_path.name} → digital (OCR disabled)")
            return False

    except Exception as e:
        logger.warning(f"Triage failed for {file_path.name}: {e}, using default OCR={OCR_ENABLED}")
        return OCR_ENABLED


def process_file(file_path: Path, output_dir: Path) -> bool:
    """단일 파일 처리.

    Args:
        file_path: 입력 파일 경로
        output_dir: 출력 디렉토리

    Returns:
        성공 여부
    """
    from src.pipeline.parsers.safe_parser import SafeParseResult
    from src.pipeline.utils.timeout_calculator import calculate_timeout

    try:
        # 출력 파일 경로
        try:
            relative_path = file_path.relative_to(INPUT_DIR)
            output_path = output_dir / relative_path.with_suffix(".yaml")
        except ValueError:
            output_path = output_dir / f"{file_path.stem}.yaml"

        # 이미 출력 파일 존재하면 스킵
        if output_path.exists():
            logger.info(f"Already exists, skipping: {output_path}")
            return True

        # Triage 기반 OCR 설정 결정
        ocr_enabled = determine_ocr_setting(file_path)

        # 동적 타임아웃 계산
        effective_timeout = calculate_timeout(file_path, base_timeout=TIMEOUT)

        # 진단 로깅
        try:
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            logger.info(
                f"Processing: {file_path.name} "
                f"(size={file_size_mb:.1f}MB, timeout={effective_timeout}s, ocr={ocr_enabled})"
            )
        except OSError:
            logger.info(f"Processing: {file_path.name} (timeout={effective_timeout}s)")

        # 파서 선택 (워커 단위 캐싱, timeout은 parse()에 전달)
        parser = get_cached_parser(ocr_enabled)

        # 파싱 (timeout_override로 캐시된 파서의 상태 변경 없이 타임아웃 적용)
        start_time = time.time()
        if SAFE_MODE:
            result = parser.parse(file_path, timeout_override=effective_timeout)
            if not isinstance(result, SafeParseResult):
                logger.error("Safe mode expected SafeParseResult, got %s", type(result))
                return False
            if not result.success:
                logger.error(
                    "SafeParse failed (%s): %s",
                    result.reason or "error",
                    (result.error or "unknown"),
                )
                return False
            parsed_doc = result.data
        else:
            parsed_doc = parser.parse(file_path)
        elapsed = time.time() - start_time

        # YAML 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(parsed_doc, f, allow_unicode=True, default_flow_style=False)

        # 완료 로깅 (처리 시간 포함)
        try:
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            throughput = file_size_mb / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {output_path.name} " f"({elapsed:.1f}s, {throughput:.2f} MB/s)")
        except OSError:
            logger.info(f"Completed: {output_path.name} ({elapsed:.1f}s)")

        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path} - {e}")
        dc_notify_error(WORKER_ID, file_path.name, f"FileNotFoundError: {e}")
        return False
    except PermissionError as e:
        logger.error(f"Permission denied: {file_path} - {e}")
        dc_notify_error(WORKER_ID, file_path.name, f"PermissionError: {e}")
        return False
    except (OSError, IOError) as e:
        logger.error(f"I/O error processing {file_path}: {e}")
        dc_notify_error(WORKER_ID, file_path.name, f"IOError: {e}")
        return False
    except yaml.YAMLError as e:
        logger.error(f"YAML serialization error for {file_path}: {e}")
        dc_notify_error(WORKER_ID, file_path.name, f"YAMLError: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error processing {file_path}: {e}")
        dc_notify_error(WORKER_ID, file_path.name, str(e))
        return False
    finally:
        # 명시적 가비지 컬렉션으로 메모리 누수 방지
        gc.collect()


def log_idle_status(r: redis.Redis, idle_count: int) -> None:
    """유휴 상태 로그 출력."""
    if idle_count % IDLE_LOG_INTERVAL == 0:
        queue_len = r.llen(KEY_QUEUE)
        processing_len = r.llen(KEY_PROCESSING)
        done_count = r.scard(KEY_DONE)
        logger.debug(f"Idle... queue={queue_len}, " f"processing={processing_len}, done={done_count}")


def handle_success(r: redis.Redis, file_name: str, elapsed_seconds: float = 0) -> None:
    """처리 성공 후 완료 마킹 및 메트릭 기록.

    Args:
        r: Redis 연결
        file_name: 파일명
        elapsed_seconds: 처리 시간 (초)
    """
    r.sadd(KEY_DONE, file_name)
    logger.info(f"Marked as done: {file_name}")

    # 처리 시간 메트릭 기록
    if elapsed_seconds > 0:
        record_processing_time(r, elapsed_seconds)

    # 마일스톤 알림 (1000건 단위)
    try:
        done_count = r.scard(KEY_DONE)
        queue_count = r.llen(KEY_QUEUE)
        failed_count = r.scard(KEY_FAILED)
        dc_notify_milestone(done_count, queue_count, failed_count)
    except Exception:
        pass  # 알림 실패해도 처리는 계속


def handle_failure(r: redis.Redis, file_name: str) -> None:
    """처리 실패 후 재시도 또는 DLQ 이동."""
    retry_count = increment_retry(r, file_name)
    if retry_count < MAX_RETRIES:
        logger.warning(f"Retry {retry_count}/{MAX_RETRIES}: {file_name}")
        r.lpush(KEY_QUEUE, file_name)
    else:
        logger.error(f"Max retries exceeded, moving to DLQ: {file_name}")
        r.sadd(KEY_FAILED, file_name)


def worker_loop(initial_r: redis.Redis):
    """메인 워커 루프.

    Args:
        initial_r: 초기 Redis 연결. 연결 실패 시 내부에서 재연결됨.
    """
    # 로컬 변수로 관리하여 재연결 시 명확하게 업데이트
    r = initial_r

    logger.info(f"Worker {WORKER_ID} started")
    logger.info(f"INPUT_DIR: {INPUT_DIR}, OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"SAFE_MODE: {SAFE_MODE}, TIMEOUT: {TIMEOUT}s, OCR: {OCR_ENABLED}, AUTO_TRIAGE: {AUTO_TRIAGE}")

    # 시작 시 stale processing 항목 복구 (재부팅 후 복구)
    logger.info("Recovering stale processing items on startup...")
    recovered = recover_stale_processing(r)
    if recovered:
        logger.info(f"Startup recovery: {recovered} items returned to queue")

    # 시작 시 재시작 카운트 증가 및 초기 상태 보고
    increment_worker_metric(r, "restart_count")
    report_worker_status(r, "Starting", current_file="", error_count=0)

    idle_count = 0
    redis_error_count = 0
    last_recovery = time.monotonic()
    last_metrics_update = time.monotonic()

    while not shutdown_handler.is_shutdown:
        # 주기적으로 Heartbeat (Idle 상태일 때도 갱신)
        # report_worker_status는 hset으로 덮어쓰므로, 상태 변경이 없을때는 heartbeat만 갱신 필요
        # 하지만 여기선 간단하게 루프 돌때마다 혹은 Idle일때 처리

        # 주기적으로 processing 리스트 정리 (stale 복구)
        if time.monotonic() - last_recovery >= STALE_RECOVERY_INTERVAL:
            recover_stale_processing(r)
            last_recovery = time.monotonic()

        # 주기적으로 메트릭 집계 업데이트
        if time.monotonic() - last_metrics_update >= METRICS_SUMMARY_INTERVAL:
            update_metrics_summary(r)
            last_metrics_update = time.monotonic()

        # 큐에서 파일 가져오기 (RPOPLPUSH: atomic하게 이동)
        try:
            file_name = r.rpoplpush(KEY_QUEUE, KEY_PROCESSING)
            redis_error_count = 0  # 성공 시 카운터 리셋
        except redis.ConnectionError as e:
            redis_error_count += 1
            logger.warning(f"Redis connection error ({redis_error_count}/{MAX_REDIS_ERRORS}): {e}")
            if redis_error_count >= MAX_REDIS_ERRORS:
                logger.error("Max Redis errors exceeded, attempting reconnect...")
                # Core 재부팅 시간(~3분) 고려, 5분간 재시도
                max_reconnect_attempts = 30
                reconnect_interval = 10  # 10초 간격

                for attempt in range(1, max_reconnect_attempts + 1):
                    try:
                        r = connect_redis_with_retry(max_retries=1)
                        redis_error_count = 0
                        logger.info(f"Redis reconnected successfully (attempt {attempt})")
                        break
                    except redis.ConnectionError:
                        logger.warning(f"Reconnect attempt {attempt}/{max_reconnect_attempts} failed")
                        if attempt < max_reconnect_attempts:
                            time.sleep(reconnect_interval)
                else:
                    # 모든 시도 실패 시 워커 종료 (docker restart로 복구)
                    logger.error("Max reconnect attempts exceeded, exiting worker loop...")
                    break
            else:
                time.sleep(2)
            continue
        except redis.RedisError as e:
            logger.error(f"Redis error: {e}")
            time.sleep(1)
            continue

        if not file_name:
            idle_count += 1
            if idle_count % 10 == 0:  # 너무 자주 업데이트하지 않도록 throttle
                report_worker_status(r, "Idle", current_file="")
            log_idle_status(r, idle_count)
            time.sleep(1)
            continue

        idle_count = 0
        r.srem(KEY_QUEUE_SET, file_name)
        r.sadd(KEY_PROCESSING_SET, file_name)

        # 종료 요청 시 새 작업 시작 금지
        if shutdown_handler.is_shutdown:
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            r.sadd(KEY_QUEUE_SET, file_name)
            r.lpush(KEY_QUEUE, file_name)
            break

        # 호스트 절대경로 → 상대경로 변환 (경로 호환성)
        host_prefixes = ["/data/"]
        for prefix in host_prefixes:
            if file_name.startswith(prefix):
                file_name = file_name[len(prefix) :]
                break

        logger.info(f"Pulled from queue: {file_name}")

        # 처리 시작 보고
        report_worker_status(r, "Processing", current_file=file_name, start_time=int(time.time()))

        # 사전 중복 체크 (락 획득 전 - 불필요한 락 경합 방지)
        if is_already_processed(r, file_name):
            logger.info(f"Pre-check: already processed, skipping: {file_name}")
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            continue

        # 락 획득 시도
        if not acquire_lock(r, file_name):
            logger.warning(f"Lock exists, another worker processing: {file_name}")
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            r.sadd(KEY_QUEUE_SET, file_name)
            r.lpush(KEY_QUEUE, file_name)
            continue

        # 락 획득 후 중복 체크 (double-check pattern - race condition 방지)
        if is_already_processed(r, file_name):
            logger.info(f"Already processed, skipping: {file_name}")
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            release_lock(r, file_name)
            continue

        file_path = INPUT_DIR / file_name

        # 대용량 파일 체크 (Low-VRAM GPU에서는 스킵)
        if WORKER_ID.startswith(LOW_VRAM_WORKER_PREFIXES):
            try:
                file_size_mb = file_path.stat().st_size / 1024 / 1024
                if file_size_mb >= LARGE_FILE_THRESHOLD_MB:
                    logger.info(
                        f"Large file ({file_size_mb:.1f}MB >= {LARGE_FILE_THRESHOLD_MB}MB), "
                        f"skipping on {WORKER_ID} (low VRAM): {file_name}"
                    )
                    r.lrem(KEY_PROCESSING, 0, file_name)
                    r.srem(KEY_PROCESSING_SET, file_name)
                    release_lock(r, file_name)
                    # 큐 맨 앞에 넣어서 3090 워커가 처리하도록
                    r.sadd(KEY_QUEUE_SET, file_name)
                    r.lpush(KEY_QUEUE, file_name)
                    continue
            except OSError:
                pass  # 파일 접근 실패 시 그냥 진행

        success = False
        processing_start_time = time.monotonic()
        lock_stop = threading.Event()
        lock_thread, lock_failed_event = start_lock_extender(r, file_name, lock_stop)

        try:
            if not file_path.exists():
                # 파일이 없으면 (이미 격리로 이동됨) 바로 DLQ로 보내고 재큐잉 방지
                logger.warning(f"File not found (already isolated?): {file_path}")
                r.sadd(KEY_FAILED, file_name)
                r.srem(KEY_QUEUE_SET, file_name)
                success = False  # finally에서 handle_failure_atomic 호출 방지
                continue  # 다음 파일로
            else:
                success = process_file(file_path, OUTPUT_DIR)

                # 락 연장 실패 체크 (처리 완료 후)
                if lock_failed_event.is_set():
                    logger.warning(f"Lock was lost during processing, requeuing: {file_name}")
                    success = False  # 재큐잉 트리거

        except (KeyboardInterrupt, SystemExit):
            logger.warning("Received shutdown signal during processing")
            shutdown_handler._shutdown = True

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_name}: {e}")
            handle_failure_atomic(r, file_name, error_code="FILE_NOT_FOUND", error_detail=str(e))
            increment_worker_metric(r, "error_count")
            success = False

        except PermissionError as e:
            logger.error(f"Permission denied: {file_name}: {e}")
            handle_failure_atomic(r, file_name, error_code="PERMISSION_DENIED", error_detail=str(e))
            increment_worker_metric(r, "error_count")
            success = False

        except OSError as e:
            logger.error(f"I/O error for {file_name}: {e}")
            handle_failure_atomic(r, file_name, error_code="IO_ERROR", error_detail=str(e))
            increment_worker_metric(r, "error_count")
            success = False

        except Exception as e:
            logger.exception(f"Unexpected error processing {file_name}: {e}")
            increment_worker_metric(r, "error_count")
            dc_notify_error(WORKER_ID, file_name, str(e))

        finally:
            lock_stop.set()
            lock_thread.join(timeout=2)
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            release_lock(r, file_name)

            if success:
                elapsed = time.monotonic() - processing_start_time
                handle_success(r, file_name, elapsed_seconds=elapsed)
            else:
                # 이미 handle_failure_atomic에서 처리된 경우는 skip
                # process_file에서 False를 반환한 경우만 처리
                # O(1) sismember 사용 (smembers O(N) 대신)
                if not r.sismember(KEY_FAILED, file_name):
                    # 락 연장 실패인지 파싱 실패인지 구분
                    if lock_failed_event.is_set():
                        error_code = "LOCK_LOST"
                        error_detail = "Lock extension failed during processing"
                    else:
                        error_code = "PARSE_ERROR"
                        error_detail = "process_file returned False"
                    handle_failure_atomic(r, file_name, error_code=error_code, error_detail=error_detail)
                    increment_worker_metric(r, "error_count")

        # 처리 완료/실패 후 Idle 복귀 (다음 루프 초기에 어차피 되지만 즉각 반영)
        report_worker_status(r, "Idle", current_file="")

    logger.info(f"Worker {WORKER_ID} shutting down")


def main():
    """메인 엔트리포인트."""
    # 시그널 핸들러 등록
    shutdown_handler.register()
    signal.siginterrupt(signal.SIGTERM, True)
    signal.siginterrupt(signal.SIGINT, True)

    # Redis 연결 (재시도 로직 포함)
    logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    try:
        r = connect_redis_with_retry()
    except redis.ConnectionError:
        sys.exit(1)

    # 워커 루프 실행
    try:
        worker_loop(r)
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")
        dc_notify_worker_crash(WORKER_ID, str(e))
        sys.exit(1)
    finally:
        shutdown_parsers()


if __name__ == "__main__":
    main()
