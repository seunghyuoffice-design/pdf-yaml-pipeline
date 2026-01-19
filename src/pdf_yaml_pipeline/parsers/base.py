"""Base classes for document parsing.

DEPRECATED: 이 모듈의 클래스들은 더 이상 사용되지 않습니다.
새 코드에서는 UnifiedParser가 반환하는 YAML dict를 직접 사용하세요.

권장:
    from src.pipeline.parsers import UnifiedParser

    parser = UnifiedParser()
    result = parser.parse(file_path)  # Returns YAML dict, not ParsedDocument

레거시 호환성을 위해 유지됩니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TableData:
    """표 데이터 구조.

    Args:
        headers: 헤더 행
        rows: 데이터 행 (헤더, 데이터) 튜플 리스트
        page_number: 표가 위치한 페이지 번호
    """

    headers: List[str]
    rows: List[Tuple[List[str], List[str]]]
    page_number: Optional[int] = None


@dataclass
class DocumentStructure:
    """문서 구조 정보.

    Args:
        headings: 헤딩 정보 리스트 [{text, level, page}, ...]
        sections: 섹션 정보 리스트 [{text, page}, ...]
        toc: 목차 정보 리스트
    """

    headings: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    toc: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """파싱된 문서 데이터.

    Args:
        source_path: 원본 파일 경로
        markdown: 변환된 마크다운 텍스트
        tables: 추출된 표 데이터 목록
        structure: 문서 구조 정보
        metadata: 메타데이터
        page_count: 페이지 수
        extraction_method: 추출 방법
    """

    source_path: str
    markdown: str
    tables: List[TableData]
    structure: DocumentStructure
    metadata: Dict[str, Any]
    page_count: int
    extraction_method: str


class BaseParser(ABC):
    """문서 파서 추상 기본 클래스.

    모든 문서 파서는 이 클래스를 상속받아 구현해야 합니다.

    Attributes:
        SUPPORTED_EXTENSIONS: 지원하는 파일 확장자 목록

    Example:
        >>> class PdfParser(BaseParser):
        ...     SUPPORTED_EXTENSIONS = ["pdf"]
        ...     def parse(self, file_path: Path) -> ParsedDocument:
        ...         # PDF 파싱 로직
        ...         pass
    """

    SUPPORTED_EXTENSIONS: List[str] = []

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """파일을 파싱하여 ParsedDocument 반환.

        Args:
            file_path: 파싱할 파일 경로

        Returns:
            ParsedDocument: 파싱된 문서 데이터

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 지원하지 않는 파일 형식일 때
        """
        pass

    def validate_file(self, file_path: Path) -> None:
        """파일 유효성 검증.

        Args:
            file_path: 검증할 파일 경로

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 지원하지 않는 파일 형식일 때
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower().lstrip(".")
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS)
            raise ValueError(f"Unsupported file type: {file_path.suffix}. " f"Supported extensions: {supported}")


__all__ = [
    "TableData",
    "DocumentStructure",
    "ParsedDocument",
    "BaseParser",
]
