"""Dynamic timeout calculator based on file characteristics.

파일 크기 및 페이지 수를 기반으로 동적 타임아웃을 계산합니다.
대용량 PDF의 타임아웃 초과 문제를 해결합니다.

Usage:
    from pdf_yaml_pipeline.utils.timeout_calculator import calculate_timeout

    timeout = calculate_timeout(Path("/data/large_file.pdf"))
    # 파일 크기/페이지에 따라 300-3600초 반환
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# pypdfium2 optional import
_DISABLE_PYPDFIUM2 = os.getenv("DISABLE_PYPDFIUM2", "false").lower() == "true"
try:
    if _DISABLE_PYPDFIUM2:
        raise ImportError("pypdfium2 disabled via env")
    import pypdfium2 as pdfium

    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False
    pdfium = None

if _DISABLE_PYPDFIUM2:
    logger.debug("pypdfium2 disabled via DISABLE_PYPDFIUM2")


# 타임아웃 상수
BASE_TIMEOUT = 300  # 기본 타임아웃 (초)
MIN_TIMEOUT = 120  # 최소 타임아웃
MAX_TIMEOUT = 3600  # 최대 타임아웃 (1시간)

# 파일 크기 기준 (bytes) → 타임아웃 (초)
SIZE_THRESHOLDS = {
    10 * 1024 * 1024: 300,  # ~10MB: 300초 (5분)
    50 * 1024 * 1024: 600,  # ~50MB: 600초 (10분)
    100 * 1024 * 1024: 1200,  # ~100MB: 1200초 (20분)
    200 * 1024 * 1024: 1800,  # ~200MB: 1800초 (30분)
}

# 페이지당 추가 시간 (초)
SECONDS_PER_PAGE = 0.5


def get_page_count_fast(file_path: Path) -> Optional[int]:
    """PDF 페이지 수 추출 (pypdfium2).

    Args:
        file_path: PDF 파일 경로

    Returns:
        페이지 수 또는 None (실패 시)
    """
    if not PYPDFIUM2_AVAILABLE:
        return None

    if file_path.suffix.lower() != ".pdf":
        return None

    try:
        pdf = pdfium.PdfDocument(str(file_path))
        page_count = len(pdf)
        pdf.close()
        return page_count
    except Exception as e:
        logger.debug(f"Failed to get page count for {file_path.name}: {e}")
        return None


def calculate_timeout(
    file_path: Path,
    base_timeout: int = BASE_TIMEOUT,
    min_timeout: int = MIN_TIMEOUT,
    max_timeout: int = MAX_TIMEOUT,
) -> int:
    """파일 크기 및 페이지 수 기반 동적 타임아웃 계산.

    Args:
        file_path: 대상 파일 경로
        base_timeout: 기본 타임아웃 (초)
        min_timeout: 최소 타임아웃 (초)
        max_timeout: 최대 타임아웃 (초)

    Returns:
        계산된 타임아웃 (초)

    Examples:
        >>> calculate_timeout(Path("small.pdf"))  # 5MB
        300
        >>> calculate_timeout(Path("large.pdf"))  # 150MB, 2000 pages
        1800  # 크기 기반 1200초 + 페이지 기반 1000초 → max 제한
    """
    file_path = Path(file_path)

    # 파일 크기 확인
    try:
        file_size = file_path.stat().st_size
    except OSError:
        logger.warning(f"Cannot stat file: {file_path}")
        return base_timeout

    # 크기 기반 타임아웃 계산
    size_timeout = base_timeout
    for threshold, timeout in sorted(SIZE_THRESHOLDS.items()):
        if file_size <= threshold:
            size_timeout = timeout
            break
    else:
        # 가장 큰 임계값 초과 시 최대값 사용
        size_timeout = max(SIZE_THRESHOLDS.values())

    # 페이지 기반 추가 시간 (PDF만)
    page_timeout = 0
    page_count = get_page_count_fast(file_path)
    if page_count:
        page_timeout = int(page_count * SECONDS_PER_PAGE)

    # 최종 타임아웃 계산
    calculated = size_timeout + page_timeout
    final_timeout = max(min_timeout, min(calculated, max_timeout))

    # 디버그 로그 (큰 타임아웃만)
    if final_timeout > base_timeout:
        file_size_mb = file_size / 1024 / 1024
        logger.debug(
            f"Dynamic timeout: {file_path.name} → {final_timeout}s "
            f"(size={file_size_mb:.1f}MB, pages={page_count or 'N/A'})"
        )

    return final_timeout


__all__ = ["calculate_timeout", "get_page_count_fast"]
