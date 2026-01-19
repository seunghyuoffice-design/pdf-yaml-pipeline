"""Separate completed HWP/HWPX files for structure-preserving reprocessing.

HWP 5.x/HWPX 파일은 네이티브 구조 파싱이 가능하므로,
구조보존률을 높인 별도 파이프라인으로 재처리할 필요가 있음.

Usage:
    # Dry run (확인만)
    python scripts/separate_hwp_done.py --dry-run

    # 실행 (파일 이동)
    python scripts/separate_hwp_done.py --execute

    # 복사 모드 (원본 유지)
    python scripts/separate_hwp_done.py --execute --copy
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import redis
except ImportError:
    print("ERROR: redis 패키지가 필요합니다. pip install redis")
    sys.exit(1)

KEY_DONE = "file:done"


def find_hwp_files_in_done(r: redis.Redis) -> tuple[list[str], list[str]]:
    """file:done에서 HWP/HWPX 파일 찾기."""
    hwp_files = []
    hwpx_files = []

    done_files = r.smembers(KEY_DONE)
    for f in done_files:
        f_lower = f.lower()
        if f_lower.endswith(".hwp"):
            hwp_files.append(f)
        elif f_lower.endswith(".hwpx"):
            hwpx_files.append(f)

    return hwp_files, hwpx_files


def separate_files(
    files: list[str],
    input_dir: Path,
    output_dir: Path,
    copy_mode: bool = False,
    dry_run: bool = True,
) -> dict:
    """파일을 별도 폴더로 분리."""
    result = {"moved": 0, "not_found": 0, "errors": []}

    for rel_path in files:
        src = input_dir / rel_path
        dst = output_dir / rel_path

        if not src.exists():
            result["not_found"] += 1
            continue

        if dry_run:
            print(f"  [DRY] {src} -> {dst}")
            result["moved"] += 1
            continue

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            result["moved"] += 1
        except Exception as e:
            result["errors"].append(f"{rel_path}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Separate HWP/HWPX files for reprocessing")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.getenv("INPUT_DIR", "/data")),
        help="Input directory containing original files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("INPUT_DIR", "/data")) / "_hwp_reprocess",
        help="Output directory for HWP/HWPX files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without moving files",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move/copy files",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (keep originals)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("ERROR: --dry-run 또는 --execute 중 하나를 지정하세요")
        sys.exit(1)

    # Redis 연결
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD")

    print(f"Connecting to Redis at {redis_host}:{redis_port}")
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
    )

    try:
        r.ping()
        print("Redis connected")
    except redis.ConnectionError as e:
        print(f"ERROR: Failed to connect to Redis: {e}")
        sys.exit(1)

    # HWP/HWPX 파일 찾기
    print("\nScanning file:done for HWP/HWPX files...")
    hwp_files, hwpx_files = find_hwp_files_in_done(r)

    print(f"  HWP 5.x files: {len(hwp_files)}")
    print(f"  HWPX files: {len(hwpx_files)}")
    print(f"  Total: {len(hwp_files) + len(hwpx_files)}")

    if not hwp_files and not hwpx_files:
        print("\nNo HWP/HWPX files found in done set.")
        return

    # 분리 실행
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'COPY' if args.copy else 'MOVE'}")
    print(f"Dry run: {args.dry_run}")

    all_files = hwp_files + hwpx_files
    result = separate_files(
        all_files,
        args.input_dir,
        args.output_dir,
        copy_mode=args.copy,
        dry_run=args.dry_run,
    )

    print(f"\n=== Result ===")
    print(f"  {'Would move' if args.dry_run else 'Moved'}: {result['moved']}")
    print(f"  Not found: {result['not_found']}")
    if result["errors"]:
        print(f"  Errors: {len(result['errors'])}")
        for err in result["errors"][:10]:
            print(f"    - {err}")

    if args.dry_run:
        print("\n[DRY RUN] --execute 플래그로 실제 실행하세요")


if __name__ == "__main__":
    main()
