"""Safe Parser with subprocess isolation.

SIGSEGV 등 C 레벨 크래시로부터 메인 프로세스를 보호하기 위해
Persistent Single-Process Pool 패턴을 사용합니다.

설계:
- 단일 백그라운드 프로세스가 모델을 로드하고 유지
- 파싱 작업을 백그라운드 프로세스로 전송
- 프로세스 크래시 시 자동 재시작 (모델 재로드)

Usage:
    from src.pipeline.parsers.safe_parser import SafeParser

    parser = SafeParser(timeout=300, skip_dir="/tmp/skip_pdf")
    result = parser.parse(file_path)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import shutil
import time
import uuid
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

try:
    import redis
except ImportError:
    redis = None

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Redis 설정 (PROBATION 상태 저장용)
_REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
_PROBATION_PASSED_KEY = "probation:passed:v1"  # 버전 포함 (향후 확장성)
_PROBATION_TTL_DAYS = 7  # 7일 후 만료 (장기 운영 안정성)

from src.pipeline.parsers.config_schema import (
    _get_probation_timeout,
)
from src.pipeline.utils.timeout_calculator import calculate_timeout


@dataclass
class SafeParseResult:
    """Safe parse result container."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    reason: Optional[str] = None  # "timeout", "crash", "error"


def _parse_worker_loop(
    req_q: mp.queues.Queue,
    res_q: mp.queues.Queue,
    config: Dict[str, Any],
) -> None:
    """Spawned worker process loop.

    단일 프로세스가 UnifiedParser를 로드한 뒤 순차적으로 요청을 처리합니다.
    Segfault 등으로 크래시 시 상위 프로세스가 재기동합니다.
    """
    from pathlib import Path
    from src.pipeline.parsers.unified_parser import UnifiedParser, UnifiedParserConfig

    parser_config = UnifiedParserConfig(
        ocr_engine=config["ocr_engine"],
        ocr_enabled=config["ocr_enabled"],
        table_extraction=config["table_extraction"],
        overwrite_empty_tables_with_ocr=config["overwrite_empty_tables_with_ocr"],
    )
    parser = UnifiedParser(config=parser_config)

    while True:
        payload = req_q.get()
        if payload is None:
            break

        job_id = payload.get("job_id")
        file_path_str = payload.get("file_path")

        try:
            result = parser.parse(Path(file_path_str))
            if not isinstance(result, dict):
                result = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
            res_q.put({"job_id": job_id, "success": True, "data": result})
        except Exception as e:  # noqa: BLE001
            res_q.put(
                {
                    "job_id": job_id,
                    "success": False,
                    "error": str(e),
                }
            )


class SafeParser:
    """SIGSEGV-safe parser with persistent subprocess isolation.

    Persistent Single-Process Pool 패턴을 사용하여:
    - 단일 백그라운드 프로세스가 모델을 로드하고 유지
    - 파싱 작업을 백그라운드 프로세스로 전송
    - 프로세스 크래시 시 자동 재시작 (모델 재로드)

    이 방식은 모델 로딩 오버헤드를 최소화하면서도 Segfault 격리를 제공합니다.

    Attributes:
        timeout: Maximum seconds per file (default: 600)
        skip_dir: Directory to move problem files (default: /tmp/skip_pdf)
        ocr_enabled: Enable OCR for scanned documents
        table_extraction: Enable table extraction
        ocr_engine: OCR engine to use (default: "paddle")
        overwrite_empty_tables_with_ocr: Fill empty tables with OCR
    """

    def __init__(
        self,
        timeout: int = 600,
        skip_dir: str = "/tmp/skip_pdf",
        ocr_enabled: bool = True,
        table_extraction: bool = True,
        ocr_engine: str = "paddle",
        overwrite_empty_tables_with_ocr: bool = False,
        dynamic_timeout: bool = True,
        probation_enabled: bool = True,
    ):
        self.timeout = timeout
        self.skip_dir = Path(skip_dir)
        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.ocr_engine = ocr_engine
        self.overwrite_empty_tables_with_ocr = overwrite_empty_tables_with_ocr
        self.dynamic_timeout = dynamic_timeout
        self.probation_enabled = probation_enabled

        # PROBATION timeout (ENV override + clamp)
        self._probation_timeout = _get_probation_timeout()

        # 파일별 PROBATION 통과 추적 (Redis + 메모리 캐시)
        self._probation_passed_cache: Set[str] = set()  # 메모리 캐시 (Redis 폴백용)
        self._redis: Optional["redis.Redis"] = None
        self._init_redis()

        # Ensure skip directory exists
        self.skip_dir.mkdir(parents=True, exist_ok=True)

        # Skip log file
        self.skip_log = self.skip_dir / "skip_log.jsonl"

        # Spawn 기반 단일 워커 프로세스 + IPC 큐
        self._ctx = mp.get_context("spawn")
        self._req_q: Optional[mp.queues.Queue] = None
        self._res_q: Optional[mp.queues.Queue] = None
        self._worker: Optional[mp.Process] = None
        self._consecutive_failures = 0

        logger.info(
            f"SafeParser initialized (timeout={timeout}s, "
            f"dynamic_timeout={dynamic_timeout}, skip_dir={skip_dir}, "
            f"probation={self._probation_timeout}s, persistent_pool=True, "
            f"redis={'enabled' if self._redis else 'disabled'})"
        )

    def _init_redis(self) -> None:
        """Redis 연결 초기화 (PROBATION 상태 저장용)."""
        if redis is None:
            logger.debug("Redis not available, using in-memory PROBATION tracking")
            return

        try:
            self._redis = redis.Redis(
                host=_REDIS_HOST,
                port=_REDIS_PORT,
                password=_REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            self._redis.ping()
            logger.debug("Redis connected for PROBATION persistence")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
            self._redis = None

    def _is_probation_passed(self, file_key: str) -> bool:
        """파일이 PROBATION을 통과했는지 확인.

        Args:
            file_key: 파일 경로의 절대 경로 문자열

        Returns:
            True if file has passed PROBATION
        """
        # 메모리 캐시 먼저 확인 (빠른 경로)
        if file_key in self._probation_passed_cache:
            return True

        # Redis 확인
        if self._redis is not None:
            try:
                if self._redis.sismember(_PROBATION_PASSED_KEY, file_key):
                    # 캐시에도 추가
                    self._probation_passed_cache.add(file_key)
                    return True
            except Exception as e:
                logger.warning(f"Redis sismember failed, using cache only: {e}")

        return False

    def _mark_probation_passed(self, file_key: str) -> None:
        """파일을 PROBATION 통과로 마킹.

        Args:
            file_key: 파일 경로의 절대 경로 문자열
        """
        # 메모리 캐시에 추가
        self._probation_passed_cache.add(file_key)

        # Redis에 저장 (TTL 설정: 7일 후 만료 → 장기 운영 안정성)
        if self._redis is not None:
            try:
                self._redis.sadd(_PROBATION_PASSED_KEY, file_key)
                # Set 전체에 TTL 설정 (Redis 6.2+ EXPIRE는 개별 멤버 TTL 미지원)
                # 대신 주기적으로 만료된 키 정리하거나, 여기서는 Set 자체에 TTL 설정
                self._redis.expire(_PROBATION_PASSED_KEY, _PROBATION_TTL_DAYS * 86400)
            except Exception as e:
                logger.warning(f"Redis sadd failed: {e}")

    def _ensure_worker(self) -> None:
        """워커 프로세스를 보장합니다 (크래시/미존재 시 재시작)."""
        if self._worker is not None and self._worker.is_alive():
            return

        # Cleanup previous queues/process if any
        self._kill_worker()

        self._req_q = self._ctx.Queue(maxsize=1)
        self._res_q = self._ctx.Queue(maxsize=1)

        worker_config = {
            "ocr_engine": self.ocr_engine,
            "ocr_enabled": self.ocr_enabled,
            "table_extraction": self.table_extraction,
            "overwrite_empty_tables_with_ocr": self.overwrite_empty_tables_with_ocr,
        }

        self._worker = self._ctx.Process(
            target=_parse_worker_loop,
            args=(self._req_q, self._res_q, worker_config),
        )
        self._worker.start()
        self._consecutive_failures = 0
        logger.debug("Started SafeParser worker process (spawn)")

    def _kill_worker(self) -> None:
        """워커를 강제 종료하고 자원 해제 (좀비 방지)."""
        if self._worker is not None:
            try:
                if self._worker.is_alive():
                    # Best-effort graceful shutdown before force terminate.
                    if self._req_q is not None:
                        try:
                            self._req_q.put_nowait(None)
                        except Exception:
                            pass
                    self._worker.join(timeout=2)
                    if self._worker.is_alive():
                        self._worker.terminate()
                        self._worker.join(timeout=5)
                        # terminate 후에도 살아있으면 kill
                        if self._worker.is_alive():
                            self._worker.kill()
                            self._worker.join(timeout=2)
                else:
                    # 이미 죽은 프로세스도 join 호출하여 좀비 방지
                    self._worker.join(timeout=1)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error terminating worker: {e}")
            finally:
                # close()로 리소스 해제 (Python 3.7+)
                try:
                    self._worker.close()
                except Exception:
                    pass
        self._worker = None
        for q in (self._req_q, self._res_q):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
                try:
                    q.join_thread()
                except Exception:
                    pass
        self._req_q = None
        self._res_q = None
        # 좀비 프로세스 수거 (PID 1 문제 대응)
        self._reap_zombies()

    def _reap_zombies(self) -> None:
        """좀비 프로세스 수거 (SafeParser worker 전용).

        Docker 컨테이너에서 Python이 PID 1로 실행될 때,
        자식 프로세스가 종료되어도 좀비로 남을 수 있습니다.
        SafeParser worker PID로만 제한하여 다른 subprocess 수거 방지.
        """
        if self._worker is None:
            return

        worker_pid = self._worker.pid
        if worker_pid is None:
            return

        try:
            # 특정 워커 PID만 대상 (다른 subprocess 수거 방지)
            pid, _ = os.waitpid(worker_pid, os.WNOHANG)
            if pid > 0:
                logger.debug(f"Reaped SafeParser worker zombie: {pid}")
        except ChildProcessError:
            # 이미 수거되었거나 존재하지 않음 (정상)
            pass
        except Exception as e:
            logger.debug(f"Worker zombie reap error (ignorable): {e}")

    def parse(
        self,
        file_path: Path,
        timeout_override: Optional[int] = None,
    ) -> SafeParseResult:
        """Parse file in isolated subprocess with Segfault protection.

        PROBATION 시스템:
        - 새 파일 첫 시도: PROBATION timeout (기본 30s) 적용
        - PROBATION 성공: normal timeout으로 전환, 파일을 통과 목록에 추가
        - PROBATION 실패: skip_dir로 격리

        Args:
            file_path: Path to the file to parse
            timeout_override: Optional timeout override (takes precedence over
                instance timeout when PROBATION is not active)

        Returns:
            SafeParseResult with success/failure info
        """
        file_path = Path(file_path)
        file_key = str(file_path.resolve())

        # PROBATION 여부 결정 (Redis + 메모리 캐시 확인)
        is_probation = self.probation_enabled and not self._is_probation_passed(file_key)

        # 타임아웃 결정 (우선순위: PROBATION > override > dynamic > instance)
        if is_probation:
            effective_timeout = self._probation_timeout
        elif timeout_override is not None:
            effective_timeout = timeout_override
        elif self.dynamic_timeout:
            effective_timeout = calculate_timeout(file_path, base_timeout=self.timeout)
        else:
            effective_timeout = self.timeout

        # 진단 로깅
        try:
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            mode = "PROBATION" if is_probation else "normal"
            logger.info(
                f"Parsing: {file_path.name} " f"(size={file_size_mb:.1f}MB, timeout={effective_timeout}s, mode={mode})"
            )
        except OSError:
            logger.info(f"Parsing: {file_path.name}")

        max_retries = 2  # 프로세스 크래시 시 최대 2회 재시도
        retry_count = 0

        while retry_count < max_retries:  # off-by-one 수정: <= → <
            try:
                self._ensure_worker()

                job_id = str(uuid.uuid4())
                assert self._req_q is not None
                assert self._res_q is not None
                self._req_q.put(
                    {
                        "job_id": job_id,
                        "file_path": str(file_path),
                    }
                )

                started = time.monotonic()
                while True:
                    remaining = effective_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("parse timeout")
                    try:
                        res = self._res_q.get(timeout=min(remaining, 1.0))
                    except queue.Empty:
                        if self._worker is None or not self._worker.is_alive():
                            raise BrokenProcessPool("worker died during parse")
                        continue

                    if res.get("job_id") != job_id:
                        # Unexpected payload; skip but warn
                        logger.warning("Received mismatched job_id from worker")
                        continue

                    if res.get("success"):
                        self._consecutive_failures = 0
                        # PROBATION 통과 기록 (Redis + 메모리 캐시)
                        if is_probation:
                            self._mark_probation_passed(file_key)
                            logger.info(f"PROBATION success: {file_path.name} → NORMAL")
                        return SafeParseResult(success=True, data=res.get("data"))

                    error_msg = res.get("error", "Unknown error")
                    self._consecutive_failures += 1
                    # PROBATION 실패 시 즉시 격리
                    if is_probation:
                        logger.error(f"PROBATION error: {file_path.name} → isolated")
                        self._isolate_file(file_path, reason="probation_error", error=error_msg)
                    return SafeParseResult(success=False, error=error_msg, reason="error")

            except BrokenProcessPool as e:
                self._consecutive_failures += 1
                self._kill_worker()

                # PROBATION 실패: 즉시 격리, 재시도 없음 (원본 이동으로 재큐잉 방지)
                if is_probation:
                    logger.error(f"PROBATION crash: {file_path.name} → isolated")
                    self._isolate_file(
                        file_path,
                        reason="probation_crash",
                        error=str(e),
                        move_original=True,
                    )
                    return SafeParseResult(
                        success=False,
                        error=f"Process crashed during PROBATION: {e}",
                        reason="crash",
                    )

                # Normal 모드: 재시도 허용
                retry_count += 1
                logger.error(
                    f"Worker crashed while parsing {file_path.name} " f"(retry {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:  # off-by-one 수정: > → >=
                    self._isolate_file(file_path, reason="crash", error=str(e), move_original=True)
                    return SafeParseResult(
                        success=False,
                        error=f"Process crashed after {max_retries} retries: {e}",
                        reason="crash",
                    )
                continue

            except TimeoutError:
                self._consecutive_failures += 1
                self._kill_worker()

                # PROBATION 타임아웃: 즉시 격리 (원본 이동으로 재큐잉 방지)
                if is_probation:
                    logger.error(f"PROBATION timeout: {file_path.name} " f"({effective_timeout}s) → isolated")
                    self._isolate_file(
                        file_path,
                        reason="probation_timeout",
                        error=f"Timeout after {effective_timeout}s (PROBATION)",
                        move_original=True,
                    )
                    return SafeParseResult(
                        success=False,
                        error=f"Timeout after {effective_timeout}s (PROBATION)",
                        reason="timeout",
                    )

                # Normal 모드: 기존 처리 (원본 이동으로 재큐잉 방지)
                logger.error(f"Timeout parsing {file_path.name} " f"(timeout={effective_timeout}s)")
                self._isolate_file(
                    file_path,
                    reason="timeout",
                    error=f"Timeout after {effective_timeout}s",
                    move_original=True,
                )
                return SafeParseResult(
                    success=False,
                    error=f"Timeout after {effective_timeout}s",
                    reason="timeout",
                )

            except Exception as e:  # noqa: BLE001
                error_msg = str(e)
                self._consecutive_failures += 1
                self._kill_worker()

                # PROBATION 실패: 즉시 격리
                if is_probation:
                    logger.error(f"PROBATION exception: {file_path.name} → isolated ({error_msg})")
                    self._isolate_file(file_path, reason="probation_exception", error=error_msg)
                else:
                    logger.error(f"Error parsing {file_path.name}: {error_msg}")
                    self._isolate_file(file_path, reason="error", error=error_msg)

                return SafeParseResult(
                    success=False,
                    error=error_msg,
                    reason="error",
                )

        return SafeParseResult(success=False, error="Unexpected error", reason="error")

    def close(self) -> None:
        """명시적 리소스 정리.

        워커 프로세스를 종료하고 Redis 연결을 닫습니다.
        context manager 사용 시 자동으로 호출됩니다.
        """
        self._kill_worker()
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None
        logger.debug("SafeParser closed")

    def __enter__(self) -> "SafeParser":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - 리소스 정리."""
        self.close()

    def __del__(self):
        """Fallback 리소스 정리 (GC 의존, 보장 안 됨)."""
        try:
            self._kill_worker()
        except Exception:
            pass

    def _isolate_file(
        self,
        file_path: Path,
        reason: str,
        error: Optional[str] = None,
        move_original: bool = False,
    ) -> None:
        """Move problem file to skip directory and log.

        Args:
            file_path: Path to the problem file
            reason: Why the file was isolated ("timeout", "crash", "error")
            error: Optional error message
            move_original: True면 원본 이동 (crash 시 재큐잉 방지), False면 복사
        """
        try:
            # Create subdirectory by reason
            target_dir = self.skip_dir / reason
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / file_path.name

            # Handle duplicate names (with max iteration to prevent infinite loop)
            if target_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                max_attempts = 10000
                for counter in range(1, max_attempts + 1):
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    if not target_path.exists():
                        break
                else:
                    logger.error(f"Too many duplicates for {file_path.name}")
                    return

            # 원본 이동 vs 복사 결정
            if move_original:
                shutil.move(str(file_path), str(target_path))
                logger.warning(f"Isolated (moved): {file_path} -> {target_path}")
            else:
                shutil.copy2(file_path, target_path)
                logger.warning(f"Isolated (copied): {file_path} -> {target_path}")

            # Log to skip log
            self._log_skip(file_path, target_path, reason, error)
            if reason == "probation_crash" and self._redis is not None:
                try:
                    self._redis.sadd("probation:failed", str(file_path.resolve()))
                except Exception as e:
                    logger.warning(f"Failed to record probation crash: {e}")

        except Exception as e:
            logger.error(f"Failed to isolate {file_path}: {e}")

    def _log_skip(
        self,
        original_path: Path,
        target_path: Path,
        reason: str,
        error: Optional[str] = None,
    ) -> None:
        """Append skip record to JSONL log."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "original_path": str(original_path),
            "isolated_path": str(target_path),
            "reason": reason,
            "error": error,
            "file_size": original_path.stat().st_size if original_path.exists() else None,
        }

        try:
            with self.skip_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write skip log: {e}")


__all__ = ["SafeParser", "SafeParseResult"]
