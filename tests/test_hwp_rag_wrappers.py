"""HWP RAG wrapper regression tests.

hwp_parser imports Pillow at module import time, but Pillow may not be available
in minimal CI environments. These tests only exercise the lightweight wrapper
helpers, so we stub PIL when needed to keep the contract covered.
"""

from __future__ import annotations

import sys
import types


def _import_hwp_parser():
    try:
        from pdf_yaml_pipeline.parsers import hwp_parser

        return hwp_parser
    except ImportError as exc:
        # Pillow binary wheels may be unavailable in some environments; the wrapper
        # helpers we test here don't require Pillow at runtime.
        msg = str(exc)
        if "PIL" not in msg and "_imaging" not in msg:
            raise

        sys.modules.pop("pdf_yaml_pipeline.parsers.hwp_parser", None)

        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")

        class _DummyImage:  # pragma: no cover - only for import-time type refs
            pass

        def _unavailable(*_args, **_kwargs):  # pragma: no cover
            raise RuntimeError("Pillow is not available in this test environment")

        pil_image.Image = _DummyImage
        pil_image.open = _unavailable
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image

        from pdf_yaml_pipeline.parsers import hwp_parser

        return hwp_parser


def _sample_table(hwp_parser):
    # 2x2 table: only one value cell passes min_reliability=0.60
    return hwp_parser.TableYAML(
        table_id="t1",
        page=1,
        shape=hwp_parser.TableShape(rows=2, cols=2),
        quality=hwp_parser.TableQuality(
            passed=True,
            action="index",
            fill_rate=1.0,
            avg_reliability=0.75,
            low_conf_ratio=0.0,
            header_ok=True,
        ),
        cells=[
            hwp_parser.TableCell(row=0, col=0, text="H1", reliability=1.0),
            hwp_parser.TableCell(row=0, col=1, text="H2", reliability=1.0),
            hwp_parser.TableCell(row=1, col=0, text="V1", reliability=0.70),
            hwp_parser.TableCell(row=1, col=1, text="V2", reliability=0.50),
        ],
    )


def test_hwp_table_rows_to_sentences_wrapper_filters_by_reliability():
    hwp_parser = _import_hwp_parser()
    table = _sample_table(hwp_parser)
    sentences = hwp_parser.table_rows_to_sentences(table, min_reliability=0.60)

    assert sentences == ["H1: V1"]


def test_hwp_build_rag_chunks_wrapper_emits_table_row_chunks():
    hwp_parser = _import_hwp_parser()
    parsed = hwp_parser.ParsedDocumentYAML(
        document=hwp_parser.DocumentMeta(
            source_path="dummy.hwp",
            format="hwp",
            encrypted=False,
            parser="hwp_parser",
            page_count=1,
            file_size=123,
        ),
        content=hwp_parser.DocumentContent(paragraphs=["Some paragraph."]),
        tables=[_sample_table(hwp_parser)],
    )

    chunks = hwp_parser.build_rag_chunks(parsed, min_table_reliability=0.60)

    table_row_chunks = [c for c in chunks if (c.get("meta") or {}).get("type") == "table_row"]
    assert any(c.get("text") == "H1: V1" for c in table_row_chunks)
