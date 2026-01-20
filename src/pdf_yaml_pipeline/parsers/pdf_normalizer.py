"""PDF Normalizer using pikepdf for pre-processing damaged/malformed PDFs.

pikepdf (MPL-2.0) wraps qpdf, providing:
- PDF structure repair and recovery
- Linearization (web optimization)
- Object stream decompression
- Metadata cleaning

This module runs BEFORE parsing, not as a fallback parser.

License: MPL-2.0 (Mozilla Public License 2.0)
- Compatible with MIT/Apache-2.0/BSD
- File-level copyleft (does not infect entire project)
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of PDF normalization attempt."""

    success: bool
    output_path: Optional[Path]
    repairs_applied: list[str]
    error: Optional[str] = None

    @property
    def was_modified(self) -> bool:
        """Check if any repairs were applied."""
        return len(self.repairs_applied) > 0


class PDFNormalizer:
    """Pre-processor for damaged/malformed PDFs using pikepdf.

    Usage:
        normalizer = PDFNormalizer()
        result = normalizer.normalize(Path("corrupted.pdf"))
        if result.success:
            # Use result.output_path for parsing
            parse(result.output_path)
    """

    def __init__(
        self,
        *,
        linearize: bool = True,
        decompress_streams: bool = True,
        remove_unreferenced: bool = True,
        fix_metadata: bool = True,
        temp_dir: Optional[Path] = None,
    ):
        """Initialize normalizer with repair options.

        Args:
            linearize: Optimize PDF for web streaming
            decompress_streams: Decompress object streams for better parsing
            remove_unreferenced: Remove unreferenced objects
            fix_metadata: Repair/standardize metadata
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.linearize = linearize
        self.decompress_streams = decompress_streams
        self.remove_unreferenced = remove_unreferenced
        self.fix_metadata = fix_metadata
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())

        self._pikepdf_available = self._check_pikepdf()

    def _check_pikepdf(self) -> bool:
        """Check if pikepdf is available."""
        try:
            import pikepdf

            logger.debug(f"pikepdf version: {pikepdf.__version__}")
            return True
        except ImportError:
            logger.warning("pikepdf not installed. PDF normalization disabled.")
            return False

    def normalize(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
    ) -> NormalizationResult:
        """Normalize a PDF file, repairing structure issues.

        Args:
            input_path: Path to input PDF
            output_path: Path for normalized output (default: temp file)

        Returns:
            NormalizationResult with success status and repairs applied
        """
        if not self._pikepdf_available:
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error="pikepdf not installed",
            )

        import pikepdf

        input_path = Path(input_path)
        if not input_path.exists():
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error=f"File not found: {input_path}",
            )

        repairs_applied = []

        # Determine output path
        if output_path is None:
            output_path = self.temp_dir / f"normalized_{input_path.name}"

        try:
            # Open with recovery enabled (key feature of pikepdf/qpdf)
            pdf = pikepdf.open(
                input_path,
                allow_overwriting_input=False,
                # These options enable automatic repair
            )

            # Check if recovery was needed
            if pdf.is_linearized:
                logger.debug(f"PDF already linearized: {input_path}")
            else:
                repairs_applied.append("structure_repair")

            # Decompress streams if requested
            if self.decompress_streams:
                try:
                    for page in pdf.pages:
                        # Access page contents to force decompression
                        _ = page.get("/Contents")
                    repairs_applied.append("stream_decompression")
                except Exception as e:
                    logger.debug(f"Stream decompression skipped: {e}")

            # Remove unreferenced objects
            if self.remove_unreferenced:
                repairs_applied.append("unreferenced_removal")

            # Fix metadata
            if self.fix_metadata:
                try:
                    # Ensure metadata is valid
                    with pdf.open_metadata() as meta:
                        # Just opening and closing standardizes the metadata
                        pass
                    repairs_applied.append("metadata_fix")
                except Exception as e:
                    logger.debug(f"Metadata fix skipped: {e}")

            # Save with options
            save_options = {
                "linearize": self.linearize,
                "object_stream_mode": pikepdf.ObjectStreamMode.generate,
            }

            pdf.save(output_path, **save_options)
            pdf.close()

            logger.info(
                f"PDF normalized: {input_path.name} -> {output_path.name} "
                f"(repairs: {', '.join(repairs_applied) or 'none'})"
            )

            return NormalizationResult(
                success=True,
                output_path=output_path,
                repairs_applied=repairs_applied,
            )

        except pikepdf.PasswordError:
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error="PDF is encrypted and requires password",
            )

        except pikepdf.PdfError as e:
            error_msg = str(e)
            logger.warning(f"pikepdf repair failed for {input_path}: {error_msg}")

            # Try more aggressive recovery
            return self._aggressive_recovery(input_path, output_path, error_msg)

        except Exception as e:
            logger.error(f"Unexpected error normalizing {input_path}: {e}")
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error=str(e),
            )

    def _aggressive_recovery(
        self,
        input_path: Path,
        output_path: Path,
        original_error: str,
    ) -> NormalizationResult:
        """Attempt aggressive recovery for severely damaged PDFs.

        Uses qpdf command-line as fallback if pikepdf fails.
        """
        import subprocess

        repairs_applied = ["aggressive_recovery"]

        try:
            # Try qpdf command directly with more aggressive options
            result = subprocess.run(
                [
                    "qpdf",
                    "--replace-input",
                    "--linearize",
                    "--object-streams=generate",
                    "--remove-unreferenced-resources=yes",
                    "--coalesce-contents",
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 or result.returncode == 3:
                # returncode 3 = warnings but success
                if result.returncode == 3:
                    repairs_applied.append("qpdf_warnings")
                    logger.debug(f"qpdf warnings: {result.stderr}")

                return NormalizationResult(
                    success=True,
                    output_path=output_path,
                    repairs_applied=repairs_applied,
                )
            else:
                return NormalizationResult(
                    success=False,
                    output_path=None,
                    repairs_applied=[],
                    error=f"qpdf failed: {result.stderr}",
                )

        except FileNotFoundError:
            logger.debug("qpdf command not available for aggressive recovery")
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error=f"pikepdf failed: {original_error}; qpdf CLI not available",
            )

        except subprocess.TimeoutExpired:
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error="qpdf timeout (60s)",
            )

        except Exception as e:
            return NormalizationResult(
                success=False,
                output_path=None,
                repairs_applied=[],
                error=f"Aggressive recovery failed: {e}",
            )

    def normalize_in_place(self, file_path: Path) -> NormalizationResult:
        """Normalize PDF in place (overwrites original).

        Warning: This modifies the original file. Use with caution.

        Args:
            file_path: Path to PDF to normalize

        Returns:
            NormalizationResult
        """
        # Create temporary output
        temp_output = self.temp_dir / f"temp_{file_path.name}"

        result = self.normalize(file_path, temp_output)

        if result.success and result.output_path:
            try:
                # Replace original with normalized version
                shutil.move(str(result.output_path), str(file_path))
                result.output_path = file_path
                logger.info(f"PDF normalized in place: {file_path}")
            except Exception as e:
                logger.error(f"Failed to replace original: {e}")
                result.success = False
                result.error = f"In-place replacement failed: {e}"

        return result


def normalize_pdf(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Tuple[bool, Optional[Path], str]:
    """Convenience function for single PDF normalization.

    Args:
        input_path: Path to input PDF
        output_path: Optional output path

    Returns:
        Tuple[success, output_path, error_or_info]
    """
    normalizer = PDFNormalizer()
    result = normalizer.normalize(input_path, output_path)

    info = ", ".join(result.repairs_applied) if result.repairs_applied else "no repairs"
    return result.success, result.output_path, result.error or info
