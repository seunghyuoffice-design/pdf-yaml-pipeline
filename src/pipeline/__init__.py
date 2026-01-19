"""Pipeline package for document conversion.

PDF/HWP/HWPX → YAML conversion for LLM training data.

주요 컴포넌트:
- UnifiedParser: PDF/HWP/HWPX → YAML 직접 변환
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "UnifiedParser",
    "UnifiedParserConfig",
]


def __getattr__(name: str):
    """Lazy loading to avoid import errors."""
    if name == "UnifiedParser":
        from pipeline.parsers.unified_parser import UnifiedParser
        return UnifiedParser
    elif name == "UnifiedParserConfig":
        from pipeline.parsers.unified_parser import UnifiedParserConfig
        return UnifiedParserConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
