"""Queue Initialization Script.

입력 디렉토리의 파일들을 Redis 큐에 등록합니다.

사용법:
    # Docker 내부
    docker-compose -f docker-compose.pipeline.yml run queue-init

    # 직접 실행
    python -m scripts.queue_init --input /data/input

옵션:
    --input     입력 디렉토리 (기본: $INPUT_DIR 또는 /data/input)
    --reset     기존 큐 초기화 후 재등록
    --dry-run   실제 등록 없이 파일 목록만 출력
    --pattern   파일 패턴 (기본: *.pdf,*.hwp,*.hwpx)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

try:
    import redis
except ImportError:
    print("ERROR: redis 패키지가 필요합니다. pip install redis")
    sys.exit(1)

# Redis 키
KEY_QUEUE = "file:queue"
KEY_QUEUE_SET = "file:queue:set"
KEY_PROCESSING = "file:processing"
KEY_PROCESSING_SET = "file:processing:set"
KEY_DONE = "file:done"
KEY_FAILED = "file:failed"
KEY_LOCK_PREFIX = "file:lock:"
KEY_RETRY_PREFIX = "file:retry:"

# 지원 확장자
SUPPORTED_EXTENSIONS = {".pdf", ".hwp", ".hwpx"}


def get_input_files(input_dir: Path, patterns: List[str]) -> List[Path]:
    """입력 디렉토리에서 처리 대상 파일 목록 수집 (심볼릭 링크 포함)."""
    import fnmatch
    files = []
    # os.walk with followlinks=True to traverse symlinked directories
    for root, dirs, filenames in os.walk(input_dir, followlinks=True):
        root_path = Path(root)
        for filename in filenames:
            for pattern in patterns:
                # pattern에서 **/ 제거하고 파일명만 매칭
                file_pattern = pattern.replace("**/", "")
                if fnmatch.fnmatch(filename, file_pattern):
                    files.append(root_path / filename)
                    break
    return sorted(set(files))


def reset_queue(r: redis.Redis):
    """큐 관련 모든 키 초기화."""
    print("Resetting queue...")

    # 큐 삭제
    r.delete(KEY_QUEUE)
    r.delete(KEY_QUEUE_SET)
    r.delete(KEY_PROCESSING)
    r.delete(KEY_PROCESSING_SET)

    # 락 키 삭제
    lock_keys = list(r.scan_iter(match=f"{KEY_LOCK_PREFIX}*"))
    if lock_keys:
        r.delete(*lock_keys)

    # 재시도 카운터 삭제
    retry_keys = list(r.scan_iter(match=f"{KEY_RETRY_PREFIX}*"))
    if retry_keys:
        r.delete(*retry_keys)

    print(f"Deleted: queue, processing, {len(lock_keys)} locks, {len(retry_keys)} retries")


def main():
    parser = argparse.ArgumentParser(description="Initialize file queue")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(os.getenv("INPUT_DIR", "/data/input")),
        help="Input directory",
    )
    parser.add_argument("--reset", action="store_true", help="Reset existing queue")
    parser.add_argument("--reset-all", action="store_true", help="Reset all including done/failed")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (print files only)")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf,*.hwp,*.hwpx",
        help="File patterns (comma-separated)",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        default=False,
        help="Skip already processed files (deprecated; default behavior)",
    )
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Include already processed files (override default skip)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed files (override default skip)",
    )
    parser.add_argument(
        "--include-queued",
        action="store_true",
        help="Include already queued files (may cause duplicates)",
    )
    parser.add_argument(
        "--include-processing",
        action="store_true",
        help="Include processing files (may conflict with active workers)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", "/data/output")),
        help="Output directory to check for existing YAML files",
    )
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        default=True,
        help="Skip files that already have YAML output (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing-output",
        action="store_true",
        help="Do not skip files with existing YAML output",
    )
    args = parser.parse_args()

    input_dir = args.input
    patterns = [p.strip() for p in args.pattern.split(",")]

    print(f"Input directory: {input_dir}")
    print(f"Patterns: {patterns}")

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # 파일 목록 수집
    files = get_input_files(input_dir, patterns)
    print(f"Found {len(files)} files")

    if args.dry_run:
        for f in files[:20]:
            print(f"  {f.name}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return

    # Redis 연결
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD")

    print(f"Connecting to Redis at {redis_host}:{redis_port}")
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,  # bytes→str 인코딩 일관성 유지
    )

    try:
        r.ping()
        print("Redis connected")
    except redis.ConnectionError as e:
        print(f"ERROR: Failed to connect to Redis: {e}")
        sys.exit(1)

    # 초기화
    if args.reset_all:
        reset_queue(r)
        r.delete(KEY_DONE)
        r.delete(KEY_FAILED)
        print("Deleted: done, failed")
    elif args.reset:
        reset_queue(r)

    skip_done = True
    if args.include_done:
        skip_done = False
    elif args.skip_done:
        skip_done = True

    skip_failed = not args.include_failed
    skip_queued = not args.include_queued
    skip_processing = not args.include_processing
    skip_existing_output = args.skip_existing_output and not args.no_skip_existing_output

    # output 디렉토리에서 기존 YAML 파일 수집
    output_dir = args.output
    existing_outputs = set()
    if skip_existing_output and output_dir.exists():
        print(f"Scanning output directory: {output_dir}")
        for yaml_file in output_dir.rglob("*.yaml"):
            # YAML 파일명에서 원본 파일 경로 추출
            # 예: 생명보험사/KB라이프/filename.pdf → 생명보험사/KB라이프/filename.yaml
            try:
                relative_yaml = yaml_file.relative_to(output_dir).as_posix()
                # .yaml → .pdf 변환하여 원본 파일명 추정
                for ext in [".pdf", ".hwp", ".hwpx"]:
                    original_name = relative_yaml.rsplit(".", 1)[0] + ext
                    existing_outputs.add(original_name)
            except ValueError:
                pass
        print(f"Found {len(existing_outputs)} existing YAML outputs")

    # 이미 완료된 파일 수 출력
    if skip_done:
        done_count = r.scard(KEY_DONE)
        print(f"Already done (Redis): {done_count} files")

    # 큐에 등록
    added = 0
    skipped = 0
    skipped_existing = 0

    batch_size = 500
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]

        names = []
        pipe = r.pipeline()
        for file_path in batch:
            try:
                relative_name = file_path.relative_to(input_dir).as_posix()
            except ValueError:
                relative_name = file_path.name
            names.append(relative_name)
            pipe.sismember(KEY_DONE, relative_name)
            pipe.sismember(KEY_FAILED, relative_name)
            pipe.sismember(KEY_QUEUE_SET, relative_name)
            pipe.sismember(KEY_PROCESSING_SET, relative_name)

        results = pipe.execute()
        for idx, name in enumerate(names):
            base = idx * 4
            done_flag = results[base]
            failed_flag = results[base + 1]
            queued_flag = results[base + 2]
            processing_flag = results[base + 3]

            # output 디렉토리에 이미 YAML이 있으면 건너뛰기
            if skip_existing_output and name in existing_outputs:
                skipped_existing += 1
                # Redis done set에도 추가하여 일관성 유지
                r.sadd(KEY_DONE, name)
                continue

            if (
                (skip_done and done_flag)
                or (skip_failed and failed_flag)
                or (skip_queued and queued_flag)
                or (skip_processing and processing_flag)
            ):
                skipped += 1
                continue

            if queued_flag and args.include_queued:
                r.lrem(KEY_QUEUE, 0, name)
                r.srem(KEY_QUEUE_SET, name)

            if processing_flag and args.include_processing:
                r.lrem(KEY_PROCESSING, 0, name)
                r.srem(KEY_PROCESSING_SET, name)

            r.sadd(KEY_QUEUE_SET, name)
            r.lpush(KEY_QUEUE, name)
            added += 1

    print(f"Added to queue: {added}")
    print(f"Skipped (Redis done/failed/queued): {skipped}")
    print(f"Skipped (existing YAML output): {skipped_existing}")

    # 현재 상태 출력
    queue_len = r.llen(KEY_QUEUE)
    done_count = r.scard(KEY_DONE)
    failed_count = r.scard(KEY_FAILED)

    print("\n=== Queue Status ===")
    print(f"Pending:    {queue_len}")
    print(f"Processing: {r.llen(KEY_PROCESSING)}")
    print(f"Done:       {done_count}")
    print(f"Failed:     {failed_count}")


if __name__ == "__main__":
    main()
