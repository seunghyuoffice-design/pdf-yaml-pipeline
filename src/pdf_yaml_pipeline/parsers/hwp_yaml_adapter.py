"""HWP to YAML adapter.

HWP 파일을 구조화된 YAML로 변환.
암호화 감지 및 graceful degradation 포함.

Fallback: LibreOffice headless (AGPL-free)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from src.pipeline.quality.table_quality import (
    attach_cell_reliability,
    attach_table_quality,
)
from src.pipeline.security.hwp_encryption import probe_hwp_encryption


class HWPYAMLAdapter:
    """HWP -> YAML canonical dict adapter.

    Features:
    - Encryption detection before parsing
    - Graceful degradation for encrypted files
    - Table structure extraction with quality assessment
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        table_extraction: bool = True,
        ocr_engine: str = "paddle",
        overwrite_empty_tables_with_ocr: bool = True,
    ) -> None:
        if ocr_engine != "paddle":
            raise ValueError("Only paddle OCR is supported.")

        self.ocr_enabled = ocr_enabled
        self.table_extraction = table_extraction
        self.ocr_engine = ocr_engine
        self.overwrite_empty_tables_with_ocr = overwrite_empty_tables_with_ocr

        # Try to import HWP parser
        self._hwp_available = False
        try:
            from hwp import HWPDocument  # type: ignore

            self._HWPDocument = HWPDocument
            self._hwp_available = True
        except ImportError:
            logger.warning("hwp library not available. HWP parsing will be limited.")
            self._HWPDocument = None

        # LibreOffice headless fallback (AGPL-free alternative)
        self._libreoffice_path = shutil.which("soffice") or shutil.which("libreoffice")
        if self._libreoffice_path:
            logger.info(f"LibreOffice fallback available: {self._libreoffice_path}")
        else:
            logger.warning("LibreOffice not found.")

        # olefile fallback (MIT license, pure Python)
        self._olefile_available = False
        try:
            import olefile  # type: ignore

            self._olefile = olefile
            self._olefile_available = True
            logger.info("olefile fallback available (for HWP5 OLE format)")
        except ImportError:
            logger.warning("olefile not available. HWP text fallback limited.")
            self._olefile = None

        # zipfile for HWPX format (built-in)
        import zipfile

        self._zipfile = zipfile

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse HWP file to YAML-serializable dict.

        Args:
            file_path: Path to HWP file

        Returns:
            YAML-serializable dict with document, content, tables, assets
        """
        # Check encryption first
        probe = probe_hwp_encryption(file_path)
        if probe["is_encrypted"]:
            logger.warning(f"Encrypted HWP file: {file_path}")
            return {
                "document": {
                    "source_path": str(file_path),
                    "format": "hwp",
                    "encrypted": True,
                    "parser": "hwp",
                    "encryption_reason": probe["reason"],
                },
                "content": {"paragraphs": []},
                "tables": [],
                "assets": {"images": []},
            }

        if not self._hwp_available:
            # Detect file format
            file_format = self._detect_hwp_format(file_path)

            # Try format-specific parser first
            paragraphs: List[str] = []
            parser_used = "unknown"

            if file_format == "hwpx":
                paragraphs = self._extract_hwpx_text(file_path)
                parser_used = "hwpx"
            elif file_format == "ole2" and self._olefile_available:
                paragraphs = self._extract_olefile_text(file_path)
                parser_used = "olefile"

            # Fallback to LibreOffice for old HWP or failed extraction
            if not paragraphs and self._libreoffice_path:
                paragraphs = self._extract_libreoffice_text(file_path)
                parser_used = "libreoffice"

            if paragraphs:
                return {
                    "document": {
                        "source_path": str(file_path),
                        "format": "hwp",
                        "encrypted": False,
                        "parser": parser_used,
                    },
                    "content": {"paragraphs": paragraphs},
                    "tables": [],
                    "assets": {"images": []},
                }

            logger.warning(f"HWP library not available, returning minimal structure: {file_path}")
            return {
                "document": {
                    "source_path": str(file_path),
                    "format": "hwp",
                    "encrypted": False,
                    "parser": "hwp_unavailable",
                },
                "content": {"paragraphs": []},
                "tables": [],
                "assets": {"images": []},
            }

        try:
            doc = self._HWPDocument(file_path)

            paragraphs = self._extract_paragraphs(doc)
            tables = self._extract_tables(doc)

            # Apply quality assessment
            for t in tables:
                attach_cell_reliability(t)
                attach_table_quality(t)

            return {
                "document": {
                    "source_path": str(file_path),
                    "format": "hwp",
                    "encrypted": False,
                    "parser": "sjunepark/hwp",
                },
                "content": {"paragraphs": paragraphs},
                "tables": tables,
                "assets": {"images": []},
            }

        except Exception as e:
            logger.error(f"Failed to parse HWP: {e}")
            return {
                "document": {
                    "source_path": str(file_path),
                    "format": "hwp",
                    "encrypted": False,
                    "parser": "hwp_error",
                    "error": str(e),
                },
                "content": {"paragraphs": []},
                "tables": [],
                "assets": {"images": []},
            }

    def _extract_libreoffice_text(self, file_path: Path) -> List[str]:
        """Extract plain text using LibreOffice headless (AGPL-free fallback).

        Converts HWP to TXT via soffice --headless --convert-to txt.
        """
        if not self._libreoffice_path:
            return []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # LibreOffice converts to txt in output directory
                cmd = [
                    self._libreoffice_path,
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    tmpdir,
                    str(file_path),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )

                if result.returncode != 0:
                    logger.warning(f"LibreOffice conversion failed: {result.stderr.decode()}")
                    return []

                # Find converted file
                txt_path = Path(tmpdir) / (file_path.stem + ".txt")
                if not txt_path.exists():
                    # Sometimes extension differs
                    txt_files = list(Path(tmpdir).glob("*.txt"))
                    if txt_files:
                        txt_path = txt_files[0]
                    else:
                        logger.warning("LibreOffice produced no output")
                        return []

                text = txt_path.read_text(encoding="utf-8", errors="ignore")

        except subprocess.TimeoutExpired:
            logger.warning(f"LibreOffice conversion timed out: {file_path}")
            return []
        except Exception as e:
            logger.warning(f"LibreOffice extraction failed: {e}")
            return []

        return [line.strip() for line in text.splitlines() if line.strip()]

    def _detect_hwp_format(self, file_path: Path) -> str:
        """Detect HWP file format from magic bytes.

        Returns:
            'ole2' for HWP5 OLE format
            'hwpx' for HWPX (ZIP-based XML)
            'old' for HWP 3.0/4.0
            'unknown' for unrecognized format
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)

            if header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                return "ole2"
            elif header[:4] == b"PK\x03\x04":
                return "hwpx"
            elif header[:5] == b"HWP D":
                return "old"
            else:
                return "unknown"
        except Exception:
            return "unknown"

    def _extract_hwpx_text(self, file_path: Path) -> List[str]:
        """Extract text from HWPX (ZIP-based XML format).

        HWPX stores content in Contents/section*.xml files as OOXML-like structure.
        """
        try:
            import xml.etree.ElementTree as ET

            paragraphs: List[str] = []

            with self._zipfile.ZipFile(file_path, "r") as zf:
                for name in zf.namelist():
                    if name.startswith("Contents/section") and name.endswith(".xml"):
                        try:
                            xml_data = zf.read(name).decode("utf-8", errors="ignore")
                            root = ET.fromstring(xml_data)

                            # HWPX uses hp: namespace for text
                            # Extract all text content
                            for elem in root.iter():
                                if elem.text and elem.text.strip():
                                    paragraphs.append(elem.text.strip())
                                if elem.tail and elem.tail.strip():
                                    paragraphs.append(elem.tail.strip())

                        except Exception as e:
                            logger.debug(f"Failed to parse HWPX section {name}: {e}")

            return paragraphs

        except Exception as e:
            logger.warning(f"HWPX extraction failed: {e}")
            return []

    def _extract_olefile_text(self, file_path: Path) -> List[str]:
        """Extract text from HWP using olefile (MIT license, pure Python).

        HWP files are OLE compound documents. Text is stored in 'BodyText/Section*'
        streams. Uses regex pattern matching to extract Korean/ASCII text segments.
        """
        if not self._olefile_available:
            return []

        import re
        import zlib

        try:
            ole = self._olefile.OleFileIO(str(file_path))
            text_parts: List[str] = []

            # Find all BodyText sections
            for stream_path in ole.listdir():
                stream_name = "/".join(stream_path)
                if stream_name.startswith("BodyText/Section"):
                    try:
                        data = ole.openstream(stream_path).read()

                        # HWP uses raw deflate compression (wbits=-15)
                        try:
                            data = zlib.decompress(data, -15)
                        except Exception:
                            pass  # Not compressed or different compression

                        # Try multiple encodings and extract valid text sequences
                        for encoding in ("utf-16-le", "euc-kr", "utf-8"):
                            try:
                                text = data.decode(encoding, errors="ignore")
                                # Extract Korean + ASCII sequences (4+ chars)
                                # Korean syllables: \uac00-\ud7a3
                                # Korean jamo: \u3131-\u318e
                                # ASCII printable: \u0020-\u007e
                                pattern = r"[\uac00-\ud7a3\u3131-\u318e\u0020-\u007e]{4,}"
                                matches = re.findall(pattern, text)
                                if matches:
                                    text_parts.extend(matches)
                                    break
                            except Exception:
                                continue

                    except Exception as e:
                        logger.debug(f"Failed to read stream {stream_name}: {e}")

            ole.close()

            # Deduplicate while preserving order
            seen: set[str] = set()
            unique: List[str] = []
            for part in text_parts:
                part = part.strip()
                if part and part not in seen and len(part) > 3:
                    seen.add(part)
                    unique.append(part)

            return unique

        except Exception as e:
            logger.warning(f"olefile extraction failed: {e}")
            return []

    def _extract_paragraphs(self, doc: Any) -> List[str]:
        """Extract paragraphs from HWP document."""
        out: List[str] = []

        try:
            bodytext = getattr(doc, "bodytext", None)
            if bodytext is None:
                return out

            sections = getattr(bodytext, "sections", None)
            if not isinstance(sections, list):
                return out

            for section in sections:
                paragraphs = getattr(section, "paragraphs", None)
                if not isinstance(paragraphs, list):
                    continue

                for para in paragraphs:
                    text = getattr(para, "text", None)
                    if isinstance(text, str) and text.strip():
                        out.append(text.strip())

        except Exception as e:
            logger.debug(f"Failed to extract paragraphs: {e}")

        return out

    def _extract_tables(self, doc: Any) -> List[Dict[str, Any]]:
        """Extract tables from HWP document."""
        if not self.table_extraction:
            return []

        tables_out: List[Dict[str, Any]] = []
        idx = 0

        try:
            bodytext = getattr(doc, "bodytext", None)
            if bodytext is None:
                return tables_out

            sections = getattr(bodytext, "sections", None)
            if not isinstance(sections, list):
                return tables_out

            for page_i, section in enumerate(sections, start=1):
                section_tables = getattr(section, "tables", None)
                if not isinstance(section_tables, list):
                    continue

                for table in section_tables:
                    idx += 1
                    table_dict = self._convert_table(table, idx, page_i)
                    if table_dict:
                        tables_out.append(table_dict)

        except Exception as e:
            logger.debug(f"Failed to extract tables: {e}")

        return tables_out

    def _convert_table(self, table: Any, idx: int, page: int) -> Dict[str, Any]:
        """Convert HWP table to YAML schema."""
        cells: List[Dict[str, Any]] = []

        try:
            rows = getattr(table, "rows", None)
            if not isinstance(rows, list):
                return {
                    "table_id": f"table_{idx}",
                    "page": page,
                    "shape": {"rows": 0, "cols": 0},
                    "cells": [],
                }

            for r, row in enumerate(rows):
                row_cells = getattr(row, "cells", None)
                if not isinstance(row_cells, list):
                    continue

                for c, cell in enumerate(row_cells):
                    text = getattr(cell, "text", None)
                    cells.append(
                        {
                            "row": r,
                            "col": c,
                            "text": (text or "").strip() if isinstance(text, str) else "",
                            "bbox": None,
                            "coord_origin": None,
                            "row_span": 1,
                            "col_span": 1,
                            "is_header": (r == 0),
                            "ocr_text": None,
                            "ocr_confidence": None,
                            "reliability": None,
                        }
                    )

        except Exception as e:
            logger.debug(f"Failed to convert table {idx}: {e}")

        # Infer shape
        if cells:
            n_rows = max(int(c["row"]) for c in cells) + 1
            n_cols = max(int(c["col"]) for c in cells) + 1
        else:
            n_rows = n_cols = 0

        return {
            "table_id": f"table_{idx}",
            "page": page,
            "shape": {"rows": n_rows, "cols": n_cols},
            "cells": cells,
        }


__all__ = ["HWPYAMLAdapter"]
