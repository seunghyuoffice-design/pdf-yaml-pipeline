# Project Structure (pdf-yaml-pipeline)

This document defines stable entrypoints and module boundaries. The goal is to
keep imports predictable and prevent internal coupling from creeping into
external users or scripts.

## Public API (Stable)

Prefer importing only from the package root:

- `pdf_yaml_pipeline.UnifiedParser` (document parsing)
- `pdf_yaml_pipeline.Pipeline` / `PipelineConfig` / `PipelineStats`
- `pdf_yaml_pipeline.create_pipeline`

These are lazily loaded from `pdf_yaml_pipeline/__init__.py` to avoid optional
dependency failures in light-weight environments.

## Entrypoints (Operational)

Use these only for runtime execution or integration scripts:

- `pdf_yaml_pipeline/orchestrator.py` (pipeline orchestration)
- `pdf_yaml_pipeline/yaml_runner.py` (CLI-style runner)
- `pdf_yaml_pipeline/four_stage_pipeline.py` (legacy, documented for backward compatibility)

If a new entrypoint is needed, add it here and update `__init__.py` or README as
appropriate.

### four_stage_pipeline (Legacy)

Status: maintained for compatibility with existing docs and internal usage.

- Not exported from `pdf_yaml_pipeline/__init__.py` (intentional)
- Use via direct import: `from pdf_yaml_pipeline.four_stage_pipeline import FourStagePipeline`
- Keep APIs stable unless a migration guide is provided
- If removing, first update `src/pdf_yaml_pipeline/README.md` and add deprecation notes

Usage evidence (as of 2026-01-25):

- No direct Python imports found in `src/` (only documentation references)
- Documented in `src/pdf_yaml_pipeline/README.md`

## Directory Map (Ownership)

```
pdf_yaml_pipeline/
├── converters/   # format conversion (YAML/JSONL)
├── parsers/      # document parsing (PDF/HWP/HWPX, OCR, validation)
├── triage/       # scanned vs digital detection
├── quality/      # quality checks, gating, metadata
├── qa/           # QA-specific pipelines and validators
├── rag/          # RAG support utilities
├── ocr/          # OCR post-processing
├── deduplication/# deduplication logic
├── security/     # security and encryption checks
└── utils/        # small shared helpers (no heavy dependencies)
```

## Dependency Direction

- `orchestrator.py` is the top-level coordinator and may depend on any module.
- `parsers/` may depend on `utils/`, `triage/`, `ocr/`, and `security/`.
- `converters/` should only depend on `utils/` and data structures from parsing.
- `quality/` and `qa/` may depend on `utils/` and parsed data structures, but
  should not reach into `parsers/` internals.
- `utils/` must remain dependency-light and never import from higher layers.

## Structure Hardening Checklist

- New public entrypoints must be added to `__init__.py` and documented here.
- Avoid cross-imports between `parsers/`, `quality/`, and `qa/` that create cycles.
- Keep `utils/` free of heavy dependencies to avoid import-time failures.
- Run `scripts/structure_check.sh` before release to detect boundary violations.

## Allowed Exceptions (Documented)

These are intentional, limited cross-layer imports that are permitted:

- `converters/yaml_converter.py` imports `parsers.base.ParsedDocument` for typed data handling.
- `qa/qa_pipeline.py` imports `parsers.special_clause_parser` for clause-specific extraction.
