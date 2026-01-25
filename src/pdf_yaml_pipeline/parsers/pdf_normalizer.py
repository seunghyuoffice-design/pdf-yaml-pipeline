"""PDF utilities using qpdf CLI for page counting and chunking.

qpdf CLI (Apache-2.0) is used for:
- Memory-efficient page counting
- Large PDF chunking without loading the full file

This module avoids external PDF repair libraries.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of a qpdf operation."""

    success: bool
    output_path: Optional[Path]
    repairs_applied: List[str]
    error: Optional[str] = None

    @property
    def was_modified(self) -> bool:
        return len(self.repairs_applied) > 0


# =============================================================================
# qpdf CLI 기반 대용량 PDF 처리 (메모리 효율적)
# =============================================================================


def count_pages_qpdf(pdf_path: Path) -> int:
    """qpdf CLI로 PDF 페이지 수 확인 (메모리 효율적)."""
    try:
        result = subprocess.run(
            ["qpdf", "--show-npages", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"qpdf failed: {result.stderr}")
        return int(result.stdout.strip())
    except FileNotFoundError as exc:
        raise RuntimeError("qpdf CLI not installed") from exc


def split_pdf_by_pages(
    input_path: Path,
    chunk_size: int = 50,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """qpdf CLI로 대용량 PDF를 페이지 청크로 분할 (메모리 효율적)."""
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="pdf_chunks_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = count_pages_qpdf(input_path)
    logger.info(f"Splitting {input_path.name}: {total_pages} pages, chunk_size={chunk_size}")

    chunks: List[Path] = []
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        output_path = output_dir / f"chunk_{start:05d}-{end:05d}.pdf"

        result = subprocess.run(
            [
                "qpdf",
                str(input_path),
                "--pages",
                ".",
                f"{start}-{end}",
                "--",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode not in (0, 3):
            raise RuntimeError(f"qpdf split failed: {result.stderr}")

        chunks.append(output_path)
        logger.debug(f"Created chunk: {output_path.name} (pages {start}-{end})")

    logger.info(f"Split complete: {len(chunks)} chunks")
    return chunks


def normalize_pdf_chunks(
    input_path: Path,
    chunk_size: int = 50,
    output_dir: Optional[Path] = None,
    *,
    linearize: bool = True,
) -> Generator[Tuple[Path, NormalizationResult], None, None]:
    """대용량 PDF를 청크로 분할+정규화 (qpdf CLI 단독)."""
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="pdf_normalized_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_pages = count_pages_qpdf(input_path)
    except RuntimeError as e:
        logger.error(f"Failed to get page count: {e}")
        return

    logger.info(
        f"Processing {input_path.name}: {total_pages} pages, "
        f"chunk_size={chunk_size}, ~{(total_pages + chunk_size - 1) // chunk_size} chunks"
    )

    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        output_path = output_dir / f"chunk_{start:05d}-{end:05d}.pdf"

        cmd = [
            "qpdf",
            str(input_path),
            "--pages",
            ".",
            f"{start}-{end}",
            "--",
        ]

        if linearize:
            cmd.extend([
                "--linearize",
                "--object-streams=generate",
                "--remove-unreferenced-resources=yes",
            ])

        cmd.append(str(output_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )

            repairs = ["page_extraction"]
            if linearize:
                repairs.append("linearize")

            if result.returncode == 0:
                yield (
                    output_path,
                    NormalizationResult(
                        success=True,
                        output_path=output_path,
                        repairs_applied=repairs,
                    ),
                )
            elif result.returncode == 3:
                repairs.append("qpdf_warnings")
                logger.debug(f"qpdf warnings for pages {start}-{end}: {result.stderr}")
                yield (
                    output_path,
                    NormalizationResult(
                        success=True,
                        output_path=output_path,
                        repairs_applied=repairs,
                    ),
                )
            else:
                logger.error(f"qpdf failed for pages {start}-{end}: {result.stderr}")
                yield (
                    output_path,
                    NormalizationResult(
                        success=False,
                        output_path=None,
                        repairs_applied=[],
                        error=f"qpdf failed: {result.stderr}",
                    ),
                )

        except subprocess.TimeoutExpired:
            logger.error(f"qpdf timeout for pages {start}-{end}")
            yield (
                output_path,
                NormalizationResult(
                    success=False,
                    output_path=None,
                    repairs_applied=[],
                    error="qpdf timeout (180s)",
                ),
            )

        except Exception as e:
            logger.error(f"Error processing pages {start}-{end}: {e}")
            yield (
                output_path,
                NormalizationResult(
                    success=False,
                    output_path=None,
                    repairs_applied=[],
                    error=str(e),
                ),
            )
