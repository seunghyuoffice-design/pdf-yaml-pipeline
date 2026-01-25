"""Document parsers for the pipeline.

PDF, HWP, HWPX 등 다양한 문서 형식을 YAML로 직접 변환.

권장 사용법:
    from pdf_yaml_pipeline.parsers import UnifiedParser

    parser = UnifiedParser()
    result = parser.parse(file_path)  # Returns YAML dict
"""

from pdf_yaml_pipeline.parsers.base import (
    BaseParser,
    DocumentStructure,
    ParsedDocument,
    TableData,
)
from pdf_yaml_pipeline.parsers.factory import ParserFactory  # Deprecated

__all__ = [
    # Base classes
    "BaseParser",
    "ParsedDocument",
    "DocumentStructure",
    "TableData",
    # Factory (deprecated)
    "ParserFactory",
    # YAML adapters (recommended)
    "UnifiedParser",
    "HWPYAMLAdapter",
    "HWPXYAMLAdapter",
    # Chunk parser
    "ChunkParser",
    "Chunk",
    "ChunkType",
    "ParseResult",
    # Special clause parser
    "SpecialClauseParser",
    "ClauseChunk",
    # PDF normalizer (qpdf-based)
    "PDFNormalizer",
    "NormalizationResult",
    "normalize_pdf",
    "normalize_pdf_chunks",
    "split_pdf_by_pages",
    "count_pages_qpdf",
    # CPU-only parser
    "CPUPDFParser",
    "SafeCPUParser",
]


def __getattr__(name: str):
    """Lazy loading of parser classes to avoid circular imports."""
    # YAML adapters (recommended)
    if name == "UnifiedParser":
        from pdf_yaml_pipeline.parsers.unified_parser import UnifiedParser

        return UnifiedParser
    elif name == "HWPYAMLAdapter":
        from pdf_yaml_pipeline.parsers.hwp_yaml_adapter import HWPYAMLAdapter

        return HWPYAMLAdapter
    elif name == "HWPXYAMLAdapter":
        from pdf_yaml_pipeline.parsers.hwpx_yaml_adapter import HWPXYAMLAdapter

        return HWPXYAMLAdapter
    # Chunk parser
    elif name == "ChunkParser":
        from pdf_yaml_pipeline.parsers.chunk_parser import ChunkParser

        return ChunkParser
    elif name == "Chunk":
        from pdf_yaml_pipeline.parsers.chunk_parser import Chunk

        return Chunk
    elif name == "ChunkType":
        from pdf_yaml_pipeline.parsers.chunk_parser import ChunkType

        return ChunkType
    elif name == "ParseResult":
        from pdf_yaml_pipeline.parsers.chunk_parser import ParseResult

        return ParseResult
    # Special clause parser
    elif name == "SpecialClauseParser":
        from pdf_yaml_pipeline.parsers.special_clause_parser import SpecialClauseParser

        return SpecialClauseParser
    elif name == "ClauseChunk":
        from pdf_yaml_pipeline.parsers.special_clause_parser import ClauseChunk

        return ClauseChunk
    # PDF normalizer (qpdf-based)
    elif name == "PDFNormalizer":
        from pdf_yaml_pipeline.parsers.pdf_normalizer import PDFNormalizer

        return PDFNormalizer
    elif name == "NormalizationResult":
        from pdf_yaml_pipeline.parsers.pdf_normalizer import NormalizationResult

        return NormalizationResult
    elif name == "normalize_pdf":
        from pdf_yaml_pipeline.parsers.pdf_normalizer import normalize_pdf

        return normalize_pdf
    elif name == "normalize_pdf_chunks":
        from pdf_yaml_pipeline.parsers.pdf_normalizer import normalize_pdf_chunks

        return normalize_pdf_chunks
    elif name == "split_pdf_by_pages":
        from pdf_yaml_pipeline.parsers.pdf_normalizer import split_pdf_by_pages

        return split_pdf_by_pages
    elif name == "count_pages_qpdf":
        from pdf_yaml_pipeline.parsers.pdf_normalizer import count_pages_qpdf

        return count_pages_qpdf
    # CPU-only parser (GPU 없는 환경용)
    elif name == "CPUPDFParser":
        from pdf_yaml_pipeline.parsers.cpu_pdf_parser import CPUPDFParser

        return CPUPDFParser
    elif name == "SafeCPUParser":
        from pdf_yaml_pipeline.parsers.cpu_pdf_parser import SafeCPUParser

        return SafeCPUParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
