"""Unified parser for PDF/HWP/HWPX to YAML.

모든 문서 포맷을 단일 인터페이스로 YAML 정본으로 변환.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from pdf_yaml_pipeline.parsers.cpu_pdf_parser import CPUPDFParser
from pdf_yaml_pipeline.parsers.hwp_yaml_adapter import HWPYAMLAdapter
from pdf_yaml_pipeline.parsers.hwpx_yaml_adapter import HWPXYAMLAdapter


@dataclass(frozen=True)
class UnifiedParserConfig:
    """Configuration for UnifiedParser."""

    ocr_engine: str = "paddle"  # only paddle supported
    ocr_enabled: bool = True
    table_extraction: bool = True
    overwrite_empty_tables_with_ocr: bool = True


class UnifiedParser:
    """Unified parser for all document formats.

    Returns YAML-serializable dict only (no markdown as intermediate).

    Canonical output schema:
        {
            "document": {
                "source_path": str,
                "format": str,  # pdf, hwp, hwpx
                "encrypted": bool,
                "parser": str,
            },
            "content": {
                "paragraphs": List[str],
            },
            "tables": List[{
                "table_id": str,
                "page": int,
                "shape": {"rows": int, "cols": int},
                "cells": List[{
                    "row": int,
                    "col": int,
                    "text": str,
                    "bbox": Optional[List[float]],
                    "ocr_text": Optional[str],
                    "ocr_confidence": Optional[float],
                    "reliability": Optional[float],
                    ...
                }],
                "quality": {...}
            }],
            "assets": {
                "images": List[{...}],
            }
        }
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".hwp", ".hwpx"}

    def __init__(self, config: Optional[UnifiedParserConfig] = None, **kwargs: Any) -> None:
        """Initialize UnifiedParser.

        Args:
            config: Parser configuration
            **kwargs: Passed to UnifiedParserConfig if config is None
        """
        if config is None:
            config = UnifiedParserConfig(**kwargs)
        self.config = config

        # Common adapter kwargs for HWP/HWPX
        adapter_kwargs = {
            "ocr_enabled": config.ocr_enabled,
            "table_extraction": config.table_extraction,
            "ocr_engine": config.ocr_engine,
            "overwrite_empty_tables_with_ocr": config.overwrite_empty_tables_with_ocr,
        }

        # Extension to adapter mapping
        self._adapters = {
            ".pdf": CPUPDFParser(enable_table_extraction=config.table_extraction),
            ".hwp": HWPYAMLAdapter(**adapter_kwargs),
            ".hwpx": HWPXYAMLAdapter(**adapter_kwargs),
        }

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse document to YAML-serializable dict.

        Args:
            file_path: Path to document file

        Returns:
            YAML-serializable dict

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file does not exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower().strip()
        adapter = self._adapters.get(ext)

        if adapter is None:
            raise ValueError(
                f"Unsupported extension: {ext}. " f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        return adapter.parse(file_path)

    @staticmethod
    def validate_yaml_contract(doc: Dict[str, Any]) -> None:
        """Validate that document follows YAML contract.

        Lightweight contract check to catch adapter regressions early.

        Args:
            doc: Document dict to validate

        Raises:
            ValueError: If contract is violated
        """
        # Top-level keys
        for key in ("document", "content", "tables", "assets"):
            if key not in doc:
                raise ValueError(f"Missing top-level key: {key}")

        # document section
        d = doc["document"]
        for key in ("source_path", "format", "parser", "encrypted"):
            if key not in d:
                raise ValueError(f"Missing document.{key}")

        # content section
        c = doc["content"]
        if "paragraphs" not in c or not isinstance(c["paragraphs"], list):
            raise ValueError("content.paragraphs must be a list")

        # tables section
        if not isinstance(doc["tables"], list):
            raise ValueError("tables must be a list")

        # Validate each table
        for i, table in enumerate(doc["tables"]):
            if "table_id" not in table:
                raise ValueError(f"tables[{i}] missing table_id")
            if "cells" not in table or not isinstance(table["cells"], list):
                raise ValueError(f"tables[{i}].cells must be a list")
            if "shape" not in table:
                raise ValueError(f"tables[{i}] missing shape")

        # assets section
        a = doc["assets"]
        if "images" not in a or not isinstance(a["images"], list):
            raise ValueError("assets.images must be a list")

        logger.debug("YAML contract OK")

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file extension is supported.

        Args:
            file_path: Path to check

        Returns:
            True if supported
        """
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS


__all__ = ["UnifiedParser", "UnifiedParserConfig"]
