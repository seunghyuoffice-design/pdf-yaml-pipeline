"""Parser factory for document conversion.

DEPRECATED: 이 모듈은 더 이상 사용되지 않습니다.
대신 UnifiedParser를 직접 사용하세요:

    from pdf_yaml_pipeline.parsers.unified_parser import UnifiedParser

    parser = UnifiedParser()
    result = parser.parse(file_path)  # Returns YAML dict
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pdf_yaml_pipeline.parsers.unified_parser import UnifiedParser


class ParserFactory:
    """DEPRECATED: Use UnifiedParser directly.

    이 팩토리는 레거시 코드 호환성을 위해 유지됩니다.
    새 코드에서는 UnifiedParser를 직접 사용하세요.
    """

    _instance: "UnifiedParser | None" = None

    @classmethod
    def _get_unified_parser(cls) -> "UnifiedParser":
        """Lazy load UnifiedParser singleton."""
        if cls._instance is None:
            from pdf_yaml_pipeline.parsers.unified_parser import UnifiedParser

            cls._instance = UnifiedParser()
        return cls._instance

    @classmethod
    def get_parser(cls, file_path: Path) -> "UnifiedParser":
        """DEPRECATED: Use UnifiedParser directly.

        Args:
            file_path: 파싱할 파일 경로 (확장자 체크용)

        Returns:
            UnifiedParser 인스턴스
        """
        warnings.warn(
            "ParserFactory.get_parser() is deprecated. " "Use UnifiedParser directly: UnifiedParser().parse(file_path)",
            DeprecationWarning,
            stacklevel=2,
        )

        # 확장자 체크
        ext = file_path.suffix.lower().lstrip(".")
        supported = {"pdf", "hwp", "hwpx"}
        if ext not in supported:
            raise ValueError(f"Unsupported extension: .{ext}. Supported: {', '.join(sorted(supported))}")

        return cls._get_unified_parser()

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """지원하는 파일 확장자 목록."""
        return ["pdf", "hwp", "hwpx"]

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """파일이 지원되는 형식인지 확인."""
        ext = file_path.suffix.lower().lstrip(".")
        return ext in {"pdf", "hwp", "hwpx"}

    # Legacy methods for compatibility
    @classmethod
    def register_parser(cls, ext: str, class_path: str) -> None:
        """DEPRECATED: No-op for compatibility."""
        warnings.warn(
            "ParserFactory.register_parser() is deprecated and has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None


__all__ = ["ParserFactory"]
