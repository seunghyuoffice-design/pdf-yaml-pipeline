"""OCR IOU-based cell remapping module.

테이블 셀과 OCR 결과를 IOU 기반으로 매칭하여
빈 셀에 OCR 텍스트와 신뢰도를 채움.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union between two bboxes.

    Args:
        a: First bbox (l, t, r, b)
        b: Second bbox (l, t, r, b)

    Returns:
        IOU score between 0.0 and 1.0
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def remap_ocr_lines_to_table_cells_iou(
    table: Dict[str, Any],
    ocr_lines: List[Dict[str, Any]],
    iou_threshold: float = 0.30,
    fill_only_when_text_empty: bool = True,
) -> None:
    """Remap OCR lines to table cells using IOU matching.

    Modifies table in-place, filling ocr_text and ocr_confidence
    for cells that match OCR lines with IOU >= threshold.

    Args:
        table: Table dict with cells[].bbox
        ocr_lines: List of {text, confidence, bbox:[l,t,r,b]}
        iou_threshold: Minimum IOU to consider a match
        fill_only_when_text_empty: Only fill if cell.text is empty

    Note:
        table.cells[*].bbox and ocr_lines[*].bbox must be in
        the same coordinate space (same coord_origin).
    """
    cells = table.get("cells", []) or []
    if not cells or not ocr_lines:
        return

    for cell in cells:
        # Skip cells that already have text (if configured)
        if fill_only_when_text_empty and (cell.get("text") or "").strip():
            continue

        cb = cell.get("bbox")
        if not cb or len(cb) != 4:
            continue
        cbox = (float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3]))

        best = None
        best_score = 0.0

        for ocr in ocr_lines:
            ob = ocr.get("bbox")
            if not ob or len(ob) != 4:
                continue
            obox = (float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3]))

            score = _iou(cbox, obox)
            if score >= iou_threshold and score > best_score:
                best = ocr
                best_score = score

        if best:
            txt = (best.get("text") or "").strip()
            if txt:
                cell["ocr_text"] = txt
                cell["ocr_confidence"] = float(best.get("confidence", 0.0) or 0.0)


__all__ = ["remap_ocr_lines_to_table_cells_iou"]
