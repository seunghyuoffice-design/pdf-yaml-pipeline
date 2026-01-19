"""Table quality assessment module.

셀 단위 신뢰도 계산 및 테이블 품질 게이트 구현.
"""

from __future__ import annotations

from typing import Any, Dict, List


def calculate_cell_reliability(cell: Dict[str, Any]) -> float:
    """Calculate reliability score for a single cell.

    This is the canonical implementation. Use this function across all parsers.

    Args:
        cell: Cell dict with text, ocr_text, ocr_confidence

    Returns:
        Reliability score between 0.0 and 1.0
    """
    text = (cell.get("text") or "").strip()
    ocr_text = (cell.get("ocr_text") or "").strip()
    conf = cell.get("ocr_confidence", None)

    if text:
        score = 0.90
        if len(text) <= 1:
            score -= 0.25
        return max(0.0, min(1.0, score))

    if ocr_text:
        if conf is None:
            conf = 0.50
        score = float(conf)
        if len(ocr_text) <= 1:
            score -= 0.25
        return max(0.0, min(1.0, score))

    return 0.0


def attach_cell_reliability(table: Dict[str, Any]) -> None:
    """Attach reliability scores to all cells in a table.

    Args:
        table: Table dict with cells list
    """
    for c in table.get("cells", []) or []:
        if c.get("reliability") is None:
            c["reliability"] = round(calculate_cell_reliability(c), 4)


def attach_table_quality(table: Dict[str, Any]) -> None:
    """Calculate and attach quality metrics to a table.

    Quality gate criteria:
    - min_fill_rate: 0.60 (60% of cells must have content)
    - min_avg_rel: 0.70 (average reliability >= 0.70)
    - max_low_ratio: 0.25 (max 25% of cells with low confidence)
    - header_ok: First row must be valid header

    Args:
        table: Table dict with cells list
    """
    cells = table.get("cells", []) or []
    if not cells:
        table["quality"] = {
            "passed": False,
            "action": "reprocess",
            "metrics": {
                "fill_rate": 0.0,
                "avg_reliability": 0.0,
                "low_conf_ratio": 1.0,
                "header_ok": False,
            },
        }
        return

    filled = 0
    low_conf = 0
    rels: List[float] = []

    for c in cells:
        r = float(c.get("reliability") or 0.0)
        rels.append(r)
        if r < 0.60:
            low_conf += 1

        has_any = bool((c.get("text") or "").strip() or (c.get("ocr_text") or "").strip())
        if has_any:
            filled += 1

    fill_rate = filled / max(1, len(cells))
    avg_rel = sum(rels) / max(1, len(rels))
    low_ratio = low_conf / max(1, len(cells))

    # Header validation (row 0)
    header_cells = [c for c in cells if int(c.get("row", -1)) == 0]
    header_filled = sum(1 for c in header_cells if (c.get("text") or "").strip() or (c.get("ocr_text") or "").strip())
    # 0 나누기 방어: header_cells가 비어있으면 0.0 반환
    header_avg = (
        sum(float(c.get("reliability") or 0.0) for c in header_cells) / len(header_cells)
        if header_cells and len(header_cells) > 0
        else 0.0
    )
    header_ok = bool(header_cells) and (header_filled / max(1, len(header_cells)) >= 0.5) and (header_avg >= 0.75)

    # Quality gate thresholds
    min_fill_rate = 0.60
    min_avg_rel = 0.70
    max_low_ratio = 0.25

    passed = header_ok and fill_rate >= min_fill_rate and avg_rel >= min_avg_rel and low_ratio <= max_low_ratio

    table["quality"] = {
        "passed": bool(passed),
        "action": "index" if passed else "reprocess",
        "metrics": {
            "fill_rate": round(fill_rate, 4),
            "avg_reliability": round(avg_rel, 4),
            "low_conf_ratio": round(low_ratio, 4),
            "header_ok": bool(header_ok),
        },
    }


__all__ = [
    "calculate_cell_reliability",
    "attach_cell_reliability",
    "attach_table_quality",
]
