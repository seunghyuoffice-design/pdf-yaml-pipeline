"""SafeParser Configuration Schema with Strict Validation.

설정 필드에 대한 타입 검증 및 제약조건을 적용합니다.

Usage:
    from pdf_yaml_pipeline.parsers.config_schema import SafeParserConfig

    config = SafeParserConfig(timeout_s=300, gpu_idx=0)
    # 잘못된 값은 ValidationError 발생
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# PROBATION 상수
PROBATION_DEFAULT_S = 1200  # 20분
PROBATION_MIN_S = 10  # 10초
PROBATION_MAX_S = 1200  # 20분 (대용량 PDF 지원)


def _get_gpu_idx() -> int:
    """GPU 인덱스를 환경변수에서 읽음.

    우선순위:
    1. GPU_IDX 환경변수
    2. CUDA_VISIBLE_DEVICES의 첫 번째 GPU
    3. 기본값: 0
    """
    # 직접 설정된 GPU_IDX
    if env_gpu := os.getenv("GPU_IDX"):
        try:
            return int(env_gpu)
        except ValueError:
            pass

    # CUDA_VISIBLE_DEVICES에서 추출
    if cuda_visible := os.getenv("CUDA_VISIBLE_DEVICES"):
        try:
            first_gpu = cuda_visible.split(",")[0].strip()
            return int(first_gpu)
        except (ValueError, IndexError):
            pass

    return 0


class ValidationError(ValueError):
    """Configuration validation error."""

    pass


def _get_probation_timeout() -> int:
    """PROBATION 타임아웃 계산 (ENV override + clamp).

    환경변수 PROBATION_TIMEOUT_S로 오버라이드 가능.
    값은 10-1200초 범위로 제한됨.

    Returns:
        PROBATION 타임아웃 (초)
    """
    env_val = os.getenv("PROBATION_TIMEOUT_S")
    if env_val:
        try:
            val = int(env_val)
            return max(PROBATION_MIN_S, min(val, PROBATION_MAX_S))
        except ValueError:
            pass
    return PROBATION_DEFAULT_S


def _is_monotonic(lst: List[int]) -> bool:
    """리스트가 단조 증가인지 확인."""
    if not lst:
        return True
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


@dataclass
class SafeParserConfig:
    """SafeParser 설정 스키마.

    모든 필드는 __post_init__에서 strict validation을 거칩니다.

    Attributes:
        timeout_s: 파일당 최대 타임아웃 (초). 기본 600.
        probation_timeout_s: 초기 테스트 타임아웃 (초). 기본 30, ENV override 가능.
        per_stage_timeout: 단계별 타임아웃 (초). 예: {"ocr": 120, "table": 60}
        retries: 재시도 횟수 리스트. 단조 증가 필수. 예: [1, 2, 3]
        backoff: 백오프 간격 리스트 (초). 단조 증가 필수. 예: [1, 2, 4]
        gpu_idx: GPU 인덱스. >= 0.
        max_concurrency_per_gpu: GPU당 최대 동시 작업 수. > 0.
        ocr_enabled: OCR 활성화 여부.
        table_extraction: 테이블 추출 활성화 여부.
        job_timeout_s: 전체 작업 타임아웃 (초). > 0.
        kill_grace_s: 강제 종료 전 유예 시간 (초). > 0.
        skip_dir: 문제 파일 격리 디렉토리 경로.
        worker_id: 워커 식별자. 비어있으면 안 됨.
    """

    timeout_s: int = 600
    probation_timeout_s: int = field(default_factory=_get_probation_timeout)
    per_stage_timeout: Dict[str, int] = field(default_factory=dict)
    retries: List[int] = field(default_factory=lambda: [1, 2, 3])
    backoff: List[int] = field(default_factory=lambda: [1, 2, 4])
    gpu_idx: int = field(default_factory=_get_gpu_idx)
    max_concurrency_per_gpu: int = 1
    ocr_enabled: bool = True
    table_extraction: bool = True
    job_timeout_s: int = 600
    kill_grace_s: int = 10
    skip_dir: str = "/tmp/skip_pdf"
    worker_id: Optional[str] = None

    def __post_init__(self) -> None:
        """생성 후 strict validation 수행."""
        self._validate()

    def _validate(self) -> None:
        """Strict field validation.

        Raises:
            ValidationError: 검증 실패 시
        """
        # 타임아웃 검증
        if not isinstance(self.timeout_s, int) or self.timeout_s <= 0:
            raise ValidationError(f"timeout_s must be positive int, got {self.timeout_s}")

        if not isinstance(self.probation_timeout_s, int) or self.probation_timeout_s <= 0:
            raise ValidationError(f"probation_timeout_s must be positive int, got {self.probation_timeout_s}")

        # per_stage_timeout 검증
        if not isinstance(self.per_stage_timeout, dict):
            raise ValidationError("per_stage_timeout must be a dict")
        for stage, timeout in self.per_stage_timeout.items():
            if not isinstance(stage, str) or not stage:
                raise ValidationError(f"per_stage_timeout key must be non-empty str, got {stage!r}")
            if not isinstance(timeout, int) or timeout <= 0:
                raise ValidationError(f"per_stage_timeout[{stage!r}] must be positive int, got {timeout}")

        # retries/backoff 단조 증가 검증
        if not isinstance(self.retries, list):
            raise ValidationError("retries must be a list")
        if not all(isinstance(x, int) for x in self.retries):
            raise ValidationError("retries must contain only integers")
        if not _is_monotonic(self.retries):
            raise ValidationError(f"retries must be monotonic increasing, got {self.retries}")

        if not isinstance(self.backoff, list):
            raise ValidationError("backoff must be a list")
        if not all(isinstance(x, int) for x in self.backoff):
            raise ValidationError("backoff must contain only integers")
        if not _is_monotonic(self.backoff):
            raise ValidationError(f"backoff must be monotonic increasing, got {self.backoff}")

        # GPU 인덱스 검증
        if not isinstance(self.gpu_idx, int) or self.gpu_idx < 0:
            raise ValidationError(f"gpu_idx must be non-negative int, got {self.gpu_idx}")

        # 동시성 검증
        if not isinstance(self.max_concurrency_per_gpu, int) or self.max_concurrency_per_gpu <= 0:
            raise ValidationError(f"max_concurrency_per_gpu must be positive int, got {self.max_concurrency_per_gpu}")

        # Boolean 검증
        if not isinstance(self.ocr_enabled, bool):
            raise ValidationError(f"ocr_enabled must be bool, got {type(self.ocr_enabled).__name__}")
        if not isinstance(self.table_extraction, bool):
            raise ValidationError(f"table_extraction must be bool, got {type(self.table_extraction).__name__}")

        # job_timeout_s, kill_grace_s 검증
        if not isinstance(self.job_timeout_s, int) or self.job_timeout_s <= 0:
            raise ValidationError(f"job_timeout_s must be positive int, got {self.job_timeout_s}")
        if not isinstance(self.kill_grace_s, int) or self.kill_grace_s <= 0:
            raise ValidationError(f"kill_grace_s must be positive int, got {self.kill_grace_s}")

        # skip_dir 검증
        if not isinstance(self.skip_dir, str) or not self.skip_dir:
            raise ValidationError(f"skip_dir must be non-empty str, got {self.skip_dir!r}")

        # worker_id 검증 (None 허용, 비어있는 문자열은 불허)
        if self.worker_id is not None:
            if not isinstance(self.worker_id, str) or not self.worker_id:
                raise ValidationError(f"worker_id must be non-empty str or None, got {self.worker_id!r}")

    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환."""
        return {
            "timeout_s": self.timeout_s,
            "probation_timeout_s": self.probation_timeout_s,
            "per_stage_timeout": self.per_stage_timeout,
            "retries": self.retries,
            "backoff": self.backoff,
            "gpu_idx": self.gpu_idx,
            "max_concurrency_per_gpu": self.max_concurrency_per_gpu,
            "ocr_enabled": self.ocr_enabled,
            "table_extraction": self.table_extraction,
            "job_timeout_s": self.job_timeout_s,
            "kill_grace_s": self.kill_grace_s,
            "skip_dir": self.skip_dir,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SafeParserConfig":
        """딕셔너리에서 설정 생성.

        Args:
            data: 설정 딕셔너리

        Returns:
            SafeParserConfig 인스턴스

        Raises:
            ValidationError: 검증 실패 시
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


__all__ = [
    "SafeParserConfig",
    "ValidationError",
    "PROBATION_DEFAULT_S",
    "PROBATION_MIN_S",
    "PROBATION_MAX_S",
    "_get_probation_timeout",
]
