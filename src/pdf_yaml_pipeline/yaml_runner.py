"""YAML Pipeline CLI Runner.

PDF/HWP/HWPX 파일을 YAML 정본으로 변환하는 CLI.

Usage:
    python -m pdf_yaml_pipeline.yaml_runner \
        --input /data/input \
        --output /data/output \
        --ocr paddle \
        --safe-mode \
        --overwrite

Safe Mode:
    --safe-mode (기본값) 활성화 시 각 파일을 별도 프로세스에서 파싱합니다.
    SIGSEGV 등 C 레벨 크래시 발생 시 해당 파일만 스킵하고 계속 진행합니다.
    문제 파일은 --skip-dir로 격리됩니다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
    from loguru import logger
except ImportError:
    print("Required: pyyaml, loguru")
    sys.exit(1)

from pdf_yaml_pipeline.parsers.safe_parser import SafeParser
from pdf_yaml_pipeline.parsers.unified_parser import UnifiedParser, UnifiedParserConfig


def setup_logger(log_level: str = "INFO") -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def collect_files(input_dir: Path) -> List[Path]:
    """Collect all supported files from input directory.

    Args:
        input_dir: Directory to search

    Returns:
        Sorted list of supported file paths
    """
    files = []
    for ext in UnifiedParser.SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        prog="yaml_runner",
        description="Convert PDF/HWP/HWPX to YAML canonical format",
    )
    ap.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing documents",
    )
    ap.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for YAML files",
    )
    ap.add_argument(
        "--ocr",
        default="paddle",
        choices=["paddle"],
        help="OCR engine to use (default: paddle)",
    )
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR",
    )
    ap.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table extraction",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate YAML contract for each output",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Write summary.yaml at the end",
    )
    ap.add_argument(
        "--safe-mode",
        action="store_true",
        default=True,
        help="Enable subprocess isolation for SIGSEGV protection (default: True)",
    )
    ap.add_argument(
        "--no-safe-mode",
        action="store_true",
        help="Disable subprocess isolation (faster but no crash protection)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per file in seconds for safe-mode (default: 600)",
    )
    ap.add_argument(
        "--skip-dir",
        default="/tmp/skip_pdf",
        help="Directory to isolate problem files (default: /tmp/skip_pdf)",
    )

    args = ap.parse_args()

    # Handle safe-mode flag
    use_safe_mode = args.safe_mode and not args.no_safe_mode

    setup_logger(args.log_level)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize parser
    config = UnifiedParserConfig(
        ocr_engine=args.ocr,
        ocr_enabled=not args.no_ocr,
        table_extraction=not args.no_tables,
    )

    if use_safe_mode:
        logger.info(f"Safe mode enabled (timeout={args.timeout}s, skip_dir={args.skip_dir})")
        safe_parser = SafeParser(
            timeout=args.timeout,
            skip_dir=args.skip_dir,
            ocr_enabled=not args.no_ocr,
            table_extraction=not args.no_tables,
            ocr_engine=args.ocr,
        )
    else:
        logger.warning("Safe mode disabled - SIGSEGV will crash the entire process")
        parser = UnifiedParser(config=config)

    # Collect files
    files = collect_files(input_dir)
    logger.info(f"Found {len(files)} documents in {input_dir}")

    if not files:
        logger.warning("No supported files found")
        return 0

    # Process files
    stats: Dict[str, Any] = {
        "total": len(files),
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "timeout": 0,
        "crash": 0,
        "error_files": [],
        "safe_mode": use_safe_mode,
    }
    t0 = time.time()

    for i, file_path in enumerate(files, 1):
        # Preserve relative path structure to avoid filename collisions
        try:
            rel_path = file_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(file_path.name)
        out_path = output_dir / rel_path.with_suffix(".yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if exists and not overwriting
        if out_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue

        try:
            if use_safe_mode:
                result = safe_parser.parse(file_path)
                if not result.success:
                    # Track failure reason
                    if result.reason == "timeout":
                        stats["timeout"] += 1
                    elif result.reason == "crash":
                        stats["crash"] += 1
                    stats["errors"] += 1
                    stats["error_files"].append(
                        {
                            "file": str(file_path),
                            "error": result.error,
                            "reason": result.reason,
                        }
                    )
                    logger.error(f"Failed ({result.reason}): {file_path.name}")
                    continue
                doc = result.data
            else:
                doc = parser.parse(file_path)

            # Validate contract if requested
            if args.validate:
                UnifiedParser.validate_yaml_contract(doc)

            # Write YAML
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    doc,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )

            stats["success"] += 1

            # Progress logging
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = stats["success"] / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    f"[{i}/{len(files)}] {rate:.1f}/min | "
                    f"success={stats['success']}, skip={stats['skipped']}, "
                    f"err={stats['errors']} (timeout={stats['timeout']}, crash={stats['crash']})"
                )

        except Exception as e:
            stats["errors"] += 1
            stats["error_files"].append(
                {
                    "file": str(file_path),
                    "error": str(e),
                }
            )
            logger.error(f"Failed: {file_path.name} - {e}")

    # Final stats
    elapsed = time.time() - t0
    stats["duration_seconds"] = round(elapsed, 2)
    stats["rate_per_minute"] = round(stats["success"] / elapsed * 60, 2) if elapsed > 0 else 0

    logger.info(
        f"Done: {stats['success']} success, {stats['skipped']} skipped, "
        f"{stats['errors']} errors (timeout={stats['timeout']}, crash={stats['crash']}) "
        f"in {elapsed/60:.1f} min"
    )

    # Write summary if requested
    if args.summary:
        summary_path = output_dir / "summary.yaml"
        with summary_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(stats, f, allow_unicode=True, sort_keys=False)
        logger.info(f"Summary written to {summary_path}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
