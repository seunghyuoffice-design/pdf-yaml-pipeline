"""PDF Validator for integrity checking before parsing.

Provides fast validation without full parsing:
- Header/magic byte verification
- Basic structure check
- Corruption detection

Used by multi_parser.py for pre-validation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# PDF magic bytes
PDF_MAGIC = b"%PDF-"
PDF_MAGIC_OFFSET_MAX = 1024  # PDF spec allows some junk before header


def validate_pdf_integrity(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate PDF file integrity.

    Fast validation checks:
    1. File exists and readable
    2. PDF magic bytes present
    3. Basic structure markers (%%EOF)
    4. File size reasonable

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple[is_valid, error_message]
        - (True, None) if valid
        - (False, "error description") if invalid
    """
    path = Path(file_path)

    # Check 1: File exists
    if not path.exists():
        return False, f"File not found: {file_path}"

    # Check 2: File readable
    try:
        file_size = path.stat().st_size
    except OSError as e:
        return False, f"Cannot access file: {e}"

    # Check 3: File size
    if file_size == 0:
        return False, "File is empty"

    if file_size < 100:  # Minimum viable PDF is ~100 bytes
        return False, f"File too small ({file_size} bytes) to be valid PDF"

    # Check 4: PDF magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(min(PDF_MAGIC_OFFSET_MAX, file_size))

            # Find PDF header (can be offset by some bytes)
            pdf_start = header.find(PDF_MAGIC)
            if pdf_start == -1:
                return False, "PDF magic bytes not found (not a PDF file)"

            if pdf_start > 0:
                logger.debug(f"PDF header offset by {pdf_start} bytes in {path.name}")

            # Check 5: EOF marker (read last 1KB)
            f.seek(max(0, file_size - 1024))
            tail = f.read()

            # Look for %%EOF marker
            if b"%%EOF" not in tail:
                # Not fatal, but suspicious
                logger.warning(f"PDF missing %%EOF marker: {path.name}")
                # Don't fail - some PDFs are truncated but still parseable

    except IOError as e:
        return False, f"Error reading file: {e}"

    except Exception as e:
        return False, f"Unexpected validation error: {e}"

    return True, None


def is_pdf_corrupted(file_path: str) -> Tuple[bool, Optional[str]]:
    """Check if PDF appears corrupted.

    More thorough check using pikepdf if available.

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple[is_corrupted, error_detail]
        - (False, None) if not corrupted
        - (True, "corruption description") if corrupted
    """
    # First do basic validation
    is_valid, error = validate_pdf_integrity(file_path)
    if not is_valid:
        return True, error

    # Try deeper check with pikepdf
    try:
        import pikepdf

        try:
            pdf = pikepdf.open(file_path)
            page_count = len(pdf.pages)
            pdf.close()

            if page_count == 0:
                return True, "PDF has no pages"

            return False, None

        except pikepdf.PasswordError:
            # Encrypted is not corrupted
            return False, None

        except pikepdf.PdfError as e:
            error_str = str(e).lower()

            # Classify corruption type
            if "damaged" in error_str or "invalid" in error_str:
                return True, f"PDF structure damaged: {e}"
            elif "xref" in error_str:
                return True, f"PDF cross-reference table corrupted: {e}"
            elif "stream" in error_str:
                return True, f"PDF stream error: {e}"
            else:
                return True, f"PDF error: {e}"

    except ImportError:
        # pikepdf not available, rely on basic validation
        logger.debug("pikepdf not available for deep corruption check")
        return False, None

    except Exception as e:
        # Unexpected error during check
        logger.warning(f"Unexpected error checking corruption: {e}")
        return True, f"Validation error: {e}"


def get_pdf_info(file_path: str) -> dict:
    """Get basic PDF information without full parsing.

    Args:
        file_path: Path to PDF file

    Returns:
        Dict with PDF info or empty dict on error
    """
    path = Path(file_path)

    info = {
        "file_path": str(path),
        "file_size": 0,
        "page_count": 0,
        "is_encrypted": False,
        "pdf_version": None,
        "is_linearized": False,
        "is_valid": False,
        "error": None,
    }

    # Basic validation first
    is_valid, error = validate_pdf_integrity(file_path)
    if not is_valid:
        info["error"] = error
        return info

    try:
        info["file_size"] = path.stat().st_size

        # Try to get more info with pikepdf
        try:
            import pikepdf

            pdf = pikepdf.open(file_path)
            info["page_count"] = len(pdf.pages)
            info["is_encrypted"] = pdf.is_encrypted
            info["pdf_version"] = pdf.pdf_version
            info["is_linearized"] = pdf.is_linearized
            info["is_valid"] = True
            pdf.close()

        except pikepdf.PasswordError:
            info["is_encrypted"] = True
            info["is_valid"] = True  # Encrypted is still valid

        except pikepdf.PdfError as e:
            info["error"] = str(e)

        except ImportError:
            # pikepdf not available, use basic header parsing
            with open(path, "rb") as f:
                header = f.read(100)
                # Extract version from header (e.g., %PDF-1.7)
                pdf_start = header.find(PDF_MAGIC)
                if pdf_start != -1:
                    version_end = header.find(b"\n", pdf_start)
                    if version_end != -1:
                        version_str = header[pdf_start + 5 : version_end].decode(
                            "ascii", errors="ignore"
                        )
                        info["pdf_version"] = version_str.strip()
                info["is_valid"] = True

    except Exception as e:
        info["error"] = str(e)

    return info
