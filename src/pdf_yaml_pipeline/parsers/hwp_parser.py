"""
HWP Parser for Dyarchy Pipeline - YAML ONLY Output

MIT License based HWP parser with:
- Encryption detection
- Table structure extraction with OCR remapping
- OCR confidence-based reliability scoring
- Table quality gates with pass/reprocess actions
- RAG indexing optimization

Uses sjunepark/hwp (MIT License) - safe for commercial use.

IMPORTANT: This module produces YAML output ONLY. No JSON intermediate representations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Canonical implementation from table_quality (avoid code duplication)
from src.pipeline.quality.table_quality import calculate_cell_reliability

# Optional dependencies - handled gracefully
_OCR_ENGINE = None
_OLEFILE_AVAILABLE = False

try:
    import olefile

    _OLEFILE_AVAILABLE = True
except ImportError:
    logger.warning("olefile not available - some features may be limited")

try:
    from paddleocr import PaddleOCR

    _OCR_ENGINE = PaddleOCR(lang="korean", use_angle_cls=True, show_log=False)
except ImportError:
    logger.warning("PaddleOCR not available - OCR features disabled")

try:
    from hwp import HWPDocument
except ImportError:
    logger.error("hwp library (sjunepark/hwp) not installed")
    HWPDocument = None


# =============================================================================
# YAML Data Schemas - Unified Output Format
# =============================================================================


@dataclass
class DocumentMeta:
    """Document metadata for YAML output."""

    source_path: str
    format: str
    encrypted: bool
    parser: str
    page_count: int = 0
    file_size: int = 0


@dataclass
class DocumentContent:
    """Main content section with paragraphs."""

    paragraphs: List[str] = field(default_factory=list)


@dataclass
class TableShape:
    """Table shape information."""

    rows: int
    cols: int


@dataclass
class TableCell:
    """Individual table cell with text, OCR, and reliability."""

    row: int
    col: int
    text: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    reliability: Optional[float] = None  # Calculated reliability score


@dataclass
class TableQuality:
    """Table quality metrics and gate result."""

    passed: bool
    action: str  # "index" or "reprocess"
    fill_rate: float
    avg_reliability: float
    low_conf_ratio: float
    header_ok: bool


@dataclass
class TableYAML:
    """Table structure for YAML output."""

    table_id: str
    page: int
    shape: TableShape
    cells: List[TableCell] = field(default_factory=list)
    quality: Optional[TableQuality] = None


@dataclass
class ImageAsset:
    """Image asset with OCR text."""

    image_id: str
    page: int
    ocr_text: Optional[str] = None


@dataclass
class Assets:
    """Assets section for images."""

    images: List[ImageAsset] = field(default_factory=list)


@dataclass
class ParsedDocumentYAML:
    """Complete YAML output structure for HWP documents.

    This is the ONLY output format. No JSON intermediate representations.
    """

    document: DocumentMeta
    content: DocumentContent
    tables: List[TableYAML] = field(default_factory=list)
    assets: Assets = field(default_factory=lambda: Assets(images=[]))


# =============================================================================
# OCR Result Schema (Internal)
# =============================================================================


@dataclass
class OCRResult:
    """OCR result with bounding box for cell remapping."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class CellBox:
    """Cell with bounding box for OCR matching."""

    row: int
    col: int
    bbox: Tuple[int, int, int, int]
    text: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================


def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box_a: First bounding box (x1, y1, x2, y2)
        box_b: Second bounding box (x1, y1, x2, y2)

    Returns:
        IOU score (0.0 to 1.0)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Calculate intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    if area_a + area_b - inter_area <= 0:
        return 0.0

    return inter_area / (area_a + area_b - inter_area)


def generate_id(prefix: str, file_path: Path, index: int = 0) -> str:
    """Generate unique ID for tables/images."""
    name_hash = hashlib.md5(file_path.name.encode()).hexdigest()[:6]
    return f"{prefix}_{name_hash}_{index:03d}"


# =============================================================================
# Reliability Scoring
# =============================================================================
# calculate_cell_reliability is imported from src.pipeline.quality.table_quality
# to avoid code duplication. See that module for the canonical implementation.


# =============================================================================
# Table Quality Gates
# =============================================================================


def evaluate_table_quality(
    table: TableYAML,
    min_fill_rate: float = 0.60,
    min_avg_rel: float = 0.70,
    max_low_conf_ratio: float = 0.25,
) -> TableQuality:
    """
    Evaluate table quality and determine gate pass/fail.

    Args:
        table: TableYAML instance
        min_fill_rate: Minimum fill rate threshold
        min_avg_rel: Minimum average reliability threshold
        max_low_conf_ratio: Maximum low confidence ratio threshold

    Returns:
        TableQuality with metrics and gate result
    """
    cells = table.cells
    if not cells:
        return TableQuality(
            passed=False,
            action="reprocess",
            fill_rate=0.0,
            avg_reliability=0.0,
            low_conf_ratio=1.0,
            header_ok=False,
        )

    # Calculate cell reliabilities if not already set
    total_filled = 0
    total_reliability = 0.0
    low_conf_count = 0

    for cell in cells:
        # Calculate and store reliability
        cell_dict = {
            "text": cell.text,
            "ocr_text": cell.ocr_text,
            "ocr_confidence": cell.ocr_confidence,
        }
        reliability = calculate_cell_reliability(cell_dict)
        cell.reliability = reliability

        # Count filled cells (any text or ocr_text)
        has_content = bool((cell.text or "").strip() or (cell.ocr_text or "").strip())
        if has_content:
            total_filled += 1

        total_reliability += reliability
        if reliability < 0.60:
            low_conf_count += 1

    fill_rate = total_filled / max(1, len(cells))
    avg_reliability = total_reliability / max(1, len(cells))
    low_conf_ratio = low_conf_count / max(1, len(cells))

    # Check header quality (row == 0)
    header_cells = [c for c in cells if c.row == 0]
    header_filled = sum(1 for c in header_cells if (c.text or "").strip() or (c.ocr_text or "").strip())
    header_avg_rel = sum(c.reliability or 0.0 for c in header_cells) / max(1, len(header_cells))

    header_ok = len(header_cells) > 0 and header_filled / len(header_cells) >= 0.5 and header_avg_rel >= 0.75

    # Gate decision
    passed = (
        header_ok
        and fill_rate >= min_fill_rate
        and avg_reliability >= min_avg_rel
        and low_conf_ratio <= max_low_conf_ratio
    )

    return TableQuality(
        passed=passed,
        action="index" if passed else "reprocess",
        fill_rate=round(fill_rate, 4),
        avg_reliability=round(avg_reliability, 4),
        low_conf_ratio=round(low_conf_ratio, 4),
        header_ok=header_ok,
    )


# =============================================================================
# RAG Indexing
# =============================================================================


def table_rows_to_sentences(table: TableYAML, min_reliability: float = 0.60) -> List[str]:
    """
    Convert table rows to RAG-indexable sentences.

    Format: "header1: value1 | header2: value2 | ..."

    Args:
        table: TableYAML instance
        min_reliability: Minimum reliability to include cell

    Returns:
        List of sentence strings
    """
    if not table.cells:
        return []

    rows = table.shape.rows
    cols = table.shape.cols
    if rows <= 0 or cols <= 0:
        return []

    # Build matrix
    matrix: List[List[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    reliability_matrix: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for cell in table.cells:
        r, k = cell.row, cell.col
        text = (cell.text or "").strip()
        if not text:
            text = (cell.ocr_text or "").strip()
        matrix[r][k] = text
        reliability_matrix[r][k] = cell.reliability or 0.0

    headers = matrix[0] if rows >= 1 else []
    sentences: List[str] = []

    for r in range(1, rows):
        parts: List[str] = []
        for c in range(cols):
            val = matrix[r][c].strip()
            if not val:
                continue
            # Skip low reliability cells
            if reliability_matrix[r][c] < min_reliability:
                continue
            key = headers[c].strip() if c < len(headers) else f"col_{c}"
            parts.append(f"{key}: {val}")
        if parts:
            sentences.append(" | ".join(parts))

    return sentences


def build_rag_chunks(
    parsed: ParsedDocumentYAML,
    min_table_reliability: float = 0.60,
) -> List[Dict[str, Any]]:
    """
    Build RAG indexable chunks from parsed YAML document.

    Args:
        parsed: ParsedDocumentYAML instance
        min_table_reliability: Minimum table reliability to index

    Returns:
        List of chunks with text and metadata
    """
    chunks: List[Dict[str, Any]] = []
    doc_meta = parsed.document

    # Paragraph chunks
    for idx, para in enumerate(parsed.content.paragraphs):
        para = (para or "").strip()
        if para:
            chunks.append(
                {
                    "text": para,
                    "meta": {
                        "source_path": doc_meta.source_path,
                        "type": "paragraph",
                        "idx": idx,
                    },
                }
            )

    # Table row chunks (only quality-gated passed tables)
    for table in parsed.tables:
        quality = table.quality
        if quality is None:
            quality = evaluate_table_quality(table)

        if not quality.passed:
            continue  # Skip failed tables

        table_sentences = table_rows_to_sentences(table, min_table_reliability)
        for sent in table_sentences:
            chunks.append(
                {
                    "text": sent,
                    "meta": {
                        "source_path": doc_meta.source_path,
                        "type": "table_row",
                        "table_id": table.table_id,
                        "page": table.page,
                    },
                }
            )

    return chunks


# =============================================================================
# Encryption Detection
# =============================================================================


def probe_hwp_encryption(file_path: Path) -> Tuple[bool, str]:
    """
    Detect if HWP file is encrypted using OLE inspection.

    Args:
        file_path: Path to HWP file

    Returns:
        Tuple of (is_encrypted, reason)
    """
    if not _OLEFILE_AVAILABLE:
        # Fallback: try to open and see if it fails
        try:
            if HWPDocument:
                HWPDocument(file_path)
                return False, "No OLE inspection (olefile not installed)"
            return False, "hwp library not installed"
        except Exception as e:
            return True, f"Open failed (possibly encrypted): {e}"

    try:
        if not olefile.isOleFile(str(file_path)):
            # Check if it's a ZIP-based HWP (newer HWPX format)
            import zipfile

            try:
                if zipfile.is_zipfile(str(file_path)):
                    with zipfile.ZipFile(str(file_path)) as zf:
                        for info in zf.infolist():
                            if info.flag_bits & 0x1:  # Encrypted flag
                                return True, "ZIP-based HWP encrypted"
                    return False, "ZIP-based HWP (no encryption detected)"
            except Exception:
                pass
            return False, "Not an OLE file (possibly not HWP or corrupted)"
    except Exception:
        return True, "OLE inspection failed (possible encrypted/corrupted)"

    try:
        with olefile.OleFileIO(str(file_path)) as ole:
            streams = ["/".join(s) for s in ole.listdir()]
            suspicious = [s for s in streams if "Encrypted" in s or "Password" in s]
            if suspicious:
                return True, f"Encrypted streams found: {suspicious[:3]}"
    except Exception as e:
        return True, f"OLE read failed: {e}"

    return False, "No encryption detected"


# =============================================================================
# OCR Processing
# =============================================================================


def extract_images_from_ole(file_path: Path, max_images: int = 50) -> List[Image.Image]:
    """Extract images from OLE container."""
    images: List[Image.Image] = []

    if not _OLEFILE_AVAILABLE:
        return images

    try:
        with olefile.OleFileIO(str(file_path)) as ole:
            for entry in ole.listdir(streams=True):
                if len(images) >= max_images:
                    break

                try:
                    data = ole.openstream(entry).read()
                    # PNG/JPEG signatures
                    if data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff"):
                        img = Image.open(BytesIO(data)).convert("RGB")
                        images.append(img)
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Image extraction failed: {e}")

    return images


def run_ocr_on_image(image: Image.Image, min_confidence: float = 0.6) -> List[OCRResult]:
    """Run PaddleOCR on image and return results with bounding boxes."""
    if _OCR_ENGINE is None:
        return []

    try:
        result = _OCR_ENGINE.ocr(image, cls=True)
        ocr_results: List[OCRResult] = []

        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    bbox_points = line[0]
                    text, confidence = line[1]

                    if confidence >= min_confidence:
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        bbox = (
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords)),
                        )
                        ocr_results.append(
                            OCRResult(
                                text=str(text).strip(),
                                confidence=float(confidence),
                                bbox=bbox,
                            )
                        )

        return ocr_results
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return []


def ocr_images(file_path: Path) -> Tuple[List[OCRResult], List[ImageAsset]]:
    """Run OCR on all images in HWP file."""
    ocr_results: List[OCRResult] = []
    image_assets: List[ImageAsset] = []

    images = extract_images_from_ole(file_path)

    for idx, img in enumerate(images):
        boxes = run_ocr_on_image(img)
        ocr_results.extend(boxes)

        # Combine OCR text for this image
        ocr_text = " ".join(b.text for b in boxes if b.text)
        if ocr_text:
            image_assets.append(
                ImageAsset(
                    image_id=generate_id("img", file_path, idx),
                    page=idx + 1,
                    ocr_text=ocr_text,
                )
            )

    return ocr_results, image_assets


# =============================================================================
# Table Extraction
# =============================================================================


def extract_tables_from_hwp(doc) -> List[TableYAML]:
    """Extract all tables from HWP document."""
    tables: List[TableYAML] = []

    if not hasattr(doc.bodytext, "sections"):
        return tables

    for section_idx, section in enumerate(doc.bodytext.sections):
        if not hasattr(section, "tables"):
            continue

        for table_idx, table in enumerate(section.tables):
            table_id = f"table_{section_idx}_{table_idx}"

            # Extract cells
            cells: List[TableCell] = []
            max_row, max_col = 0, 0

            try:
                for row_idx, row in enumerate(table.rows):
                    for col_idx, cell in enumerate(row.cells):
                        cell_text = getattr(cell, "text", None) or ""
                        cell_text = cell_text.strip()

                        # Get bounding box if available
                        bbox = getattr(cell, "bbox", (0, 0, 0, 0))

                        cell_obj = TableCell(
                            row=row_idx,
                            col=col_idx,
                            text=cell_text if cell_text else None,
                            ocr_text=None,
                            ocr_confidence=None,
                            reliability=None,
                        )
                        cells.append(cell_obj)
                        max_row = max(max_row, row_idx)
                        max_col = max(max_col, col_idx)
            except Exception as e:
                logger.warning(f"Failed to extract cells from table {table_id}: {e}")
                continue

            if cells:
                table_yaml = TableYAML(
                    table_id=table_id,
                    page=section_idx + 1,
                    shape=TableShape(rows=max_row + 1, cols=max_col + 1),
                    cells=cells,
                    quality=None,  # Will be calculated later
                )
                tables.append(table_yaml)

    logger.info(f"Extracted {len(tables)} tables from HWP document")
    return tables


def remap_ocr_to_table_cells(
    cells: List[TableCell],
    ocr_results: List[OCRResult],
    iou_threshold: float = 0.3,
    cell_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> None:
    """
    Remap OCR results to empty table cells.

    If cell_bboxes are provided, uses IOU matching for accurate mapping.
    Otherwise, uses sequential assignment based on OCR result order.

    Only fills cells that have no existing text.
    Stores ocr_text and ocr_confidence in matched cells.

    Args:
        cells: List of TableCell (modified in place)
        ocr_results: List of OCRResult with bounding boxes
        iou_threshold: Minimum IOU score for matching (only used with cell_bboxes)
        cell_bboxes: Optional list of cell bounding boxes (same order as cells)

    Note:
        TableCell doesn't currently have bbox attribute.
        For IOU matching, pass cell_bboxes separately or add bbox to TableCell.
    """
    # Get empty cells that need OCR
    empty_cells = [(i, cell) for i, cell in enumerate(cells) if not cell.text]
    if not empty_cells or not ocr_results:
        return

    # Filter OCR results with text
    valid_ocr = [ocr for ocr in ocr_results if ocr.text]
    if not valid_ocr:
        return

    # If cell bboxes provided, use IOU matching
    if cell_bboxes and len(cell_bboxes) == len(cells):
        used_ocr_indices = set()

        for cell_idx, cell in empty_cells:
            cell_bbox = cell_bboxes[cell_idx]
            best_ocr = None
            best_iou = 0.0
            best_ocr_idx = -1

            for ocr_idx, ocr in enumerate(valid_ocr):
                if ocr_idx in used_ocr_indices:
                    continue

                iou = calculate_iou(cell_bbox, ocr.bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_ocr = ocr
                    best_ocr_idx = ocr_idx

            if best_ocr:
                cell.ocr_text = best_ocr.text
                cell.ocr_confidence = best_ocr.confidence
                used_ocr_indices.add(best_ocr_idx)
    else:
        # Sequential assignment fallback (no bbox available)
        # Sort empty cells by position (row, col) for predictable order
        empty_cells_sorted = sorted(empty_cells, key=lambda x: (x[1].row, x[1].col))

        for (cell_idx, cell), ocr in zip(empty_cells_sorted, valid_ocr):
            if not cell.ocr_text:
                cell.ocr_text = ocr.text
                cell.ocr_confidence = ocr.confidence


# =============================================================================
# Main Parser
# =============================================================================


class HWPParser:
    """
    HWP Parser for Dyarchy Pipeline - YAML ONLY Output.

    Features:
    - Encryption detection
    - Text extraction
    - Structured table extraction
    - OCR remapping with confidence scoring
    - Table quality gates (index/reprocess)
    - RAG indexing optimization

    Output: YAML ONLY (no JSON intermediate representations)
    """

    SUPPORTED_EXTENSIONS = {".hwp"}

    def __init__(
        self,
        ocr_enabled: bool = True,
        min_fill_rate: float = 0.60,
        min_avg_rel: float = 0.70,
        max_low_conf_ratio: float = 0.25,
    ):
        """
        Initialize HWP parser.

        Args:
            ocr_enabled: Enable OCR fallback for empty cells
            min_fill_rate: Minimum table fill rate for quality gate
            min_avg_rel: Minimum average reliability for quality gate
            max_low_conf_ratio: Maximum low confidence ratio threshold
        """
        self._ocr_enabled = ocr_enabled
        self._min_fill_rate = min_fill_rate
        self._min_avg_rel = min_avg_rel
        self._max_low_conf_ratio = max_low_conf_ratio

        logger.info(f"Initialized HWPParser (OCR={ocr_enabled}, " f"gate: fill>={min_fill_rate}, rel>={min_avg_rel})")

    def parse(self, file_path: Path) -> ParsedDocumentYAML:
        """
        Parse HWP file and return YAML structure.

        Args:
            file_path: Path to HWP file

        Returns:
            ParsedDocumentYAML instance (YAML ONLY output)

        Raises:
            FileNotFoundError: If file doesn't exist
            NotImplementedError: If file is encrypted
            RuntimeError: If parsing fails
        """
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"HWP file not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Check encryption
        is_encrypted, encrypt_reason = probe_hwp_encryption(file_path)
        if is_encrypted:
            raise NotImplementedError(f"Encrypted HWP file detected: {encrypt_reason}")

        # Open document
        if HWPDocument is None:
            raise RuntimeError("hwp library (sjunepark/hwp) not installed")

        try:
            doc = HWPDocument(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open HWP file: {e}")

        # Extract paragraphs
        paragraphs: List[str] = []
        if hasattr(doc.bodytext, "sections"):
            for section in doc.bodytext.sections:
                if hasattr(section, "paragraphs"):
                    for para in section.paragraphs:
                        text = getattr(para, "text", None) or ""
                        text = text.strip()
                        if text:
                            paragraphs.append(text)

        # Extract tables
        tables = extract_tables_from_hwp(doc)

        # OCR remapping for empty cells
        if self._ocr_enabled and _OCR_ENGINE is not None:
            try:
                ocr_results, image_assets = ocr_images(file_path)
                for table in tables:
                    remap_ocr_to_table_cells(table.cells, ocr_results)
            except Exception as e:
                logger.warning(f"OCR processing failed: {e}")
                image_assets = []
        else:
            image_assets = []

        # Evaluate table quality
        for table in tables:
            table.quality = evaluate_table_quality(
                table, self._min_fill_rate, self._min_avg_rel, self._max_low_conf_ratio
            )

        # Estimate page count
        page_count = max(1, len(paragraphs) // 30 + 1)

        # Build YAML structure
        yaml_doc = ParsedDocumentYAML(
            document=DocumentMeta(
                source_path=str(file_path),
                format="hwp",
                encrypted=False,
                parser="hwp",
                page_count=page_count,
                file_size=file_path.stat().st_size,
            ),
            content=DocumentContent(paragraphs=paragraphs),
            tables=tables,
            assets=Assets(images=image_assets),
        )

        logger.info(f"Parsed HWP: {file_path.name} - " f"{len(paragraphs)} paragraphs, " f"{len(tables)} tables")

        return yaml_doc

    def parse_to_dict(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse HWP file and return as dictionary (YAML-compatible).

        Args:
            file_path: Path to HWP file

        Returns:
            Dictionary representation (directly serializable to YAML)
        """
        parsed = self.parse(file_path)
        return self._to_serializable_dict(parsed)

    def _to_serializable_dict(self, doc: ParsedDocumentYAML) -> Dict[str, Any]:
        """Convert ParsedDocumentYAML to serializable dictionary."""
        tables_list = []
        for t in doc.tables:
            tables_list.append(
                {
                    "table_id": t.table_id,
                    "page": t.page,
                    "shape": {
                        "rows": t.shape.rows,
                        "cols": t.shape.cols,
                    },
                    "quality": None
                    if t.quality is None
                    else {
                        "passed": t.quality.passed,
                        "action": t.quality.action,
                        "fill_rate": t.quality.fill_rate,
                        "avg_reliability": t.quality.avg_reliability,
                        "low_conf_ratio": t.quality.low_conf_ratio,
                        "header_ok": t.quality.header_ok,
                    },
                    "cells": [
                        {
                            "row": c.row,
                            "col": c.col,
                            "text": c.text,
                            "ocr_text": c.ocr_text,
                            "ocr_confidence": c.ocr_confidence,
                            "reliability": c.reliability,
                        }
                        for c in t.cells
                    ],
                }
            )

        return {
            "document": {
                "source_path": doc.document.source_path,
                "format": doc.document.format,
                "encrypted": doc.document.encrypted,
                "parser": doc.document.parser,
                "page_count": doc.document.page_count,
                "file_size": doc.document.file_size,
            },
            "content": {
                "paragraphs": doc.content.paragraphs,
            },
            "tables": tables_list,
            "assets": {
                "images": [
                    {
                        "image_id": img.image_id,
                        "page": img.page,
                        "ocr_text": img.ocr_text,
                    }
                    for img in doc.assets.images
                ]
            },
        }


# =============================================================================
# YAML Serialization
# =============================================================================


def dump_yaml(data: Dict[str, Any]) -> str:
    """
    Serialize data to YAML string.

    Args:
        data: Dictionary to serialize

    Returns:
        YAML formatted string
    """
    import yaml

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)


def parse_hwp(file_path: Path, **kwargs) -> ParsedDocumentYAML:
    """
    Convenience function to parse HWP file.

    Args:
        file_path: Path to HWP file
        **kwargs: Parser configuration options

    Returns:
        ParsedDocumentYAML instance
    """
    parser = HWPParser(**kwargs)
    return parser.parse(file_path)


def parse_hwp_to_yaml(file_path: Path, **kwargs) -> str:
    """
    Convenience function to parse HWP file to YAML string.

    Args:
        file_path: Path to HWP file
        **kwargs: Parser configuration options

    Returns:
        YAML string
    """
    parser = HWPParser(**kwargs)
    doc = parser.parse(file_path)
    data = parser._to_serializable_dict(doc)
    return dump_yaml(data)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Schemas
    "DocumentMeta",
    "DocumentContent",
    "TableShape",
    "TableCell",
    "TableQuality",
    "TableYAML",
    "ImageAsset",
    "Assets",
    "ParsedDocumentYAML",
    # OCR Schema
    "OCRResult",
    "CellBox",
    # Parser
    "HWPParser",
    # Quality & Reliability
    "calculate_cell_reliability",
    "evaluate_table_quality",
    # RAG
    "table_rows_to_sentences",
    "build_rag_chunks",
    # Utilities
    "parse_hwp",
    "parse_hwp_to_yaml",
    "dump_yaml",
    "calculate_iou",
    "probe_hwp_encryption",
]
