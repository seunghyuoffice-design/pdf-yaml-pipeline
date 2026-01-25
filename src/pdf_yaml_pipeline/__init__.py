"""Pipeline package for document conversion.

주요 컴포넌트:
- UnifiedParser: PDF/HWP/HWPX → YAML 직접 변환
- Pipeline: 문서 변환 오케스트레이션 (선택적)
"""

__all__ = [
    # Parsers (recommended)
    "UnifiedParser",
    # Orchestration (optional, requires src.schemas)
    "Pipeline",
    "PipelineConfig",
    "PipelineStats",
    "create_pipeline",
]


def __getattr__(name: str):
    """Lazy loading to avoid import errors."""
    if name == "UnifiedParser":
        from pdf_yaml_pipeline.parsers.unified_parser import UnifiedParser

        return UnifiedParser
    elif name in ("Pipeline", "PipelineConfig", "PipelineStats", "create_pipeline"):
        from pdf_yaml_pipeline.orchestrator import (
            Pipeline,
            PipelineConfig,
            PipelineStats,
            create_pipeline,
        )

        return {
            "Pipeline": Pipeline,
            "PipelineConfig": PipelineConfig,
            "PipelineStats": PipelineStats,
            "create_pipeline": create_pipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
