"""Rebuild done set from existing output files.

AOF 손상 등으로 Redis 데이터가 유실된 경우,
출력 디렉토리의 YAML 파일을 기반으로 file:done 세트를 재구축합니다.

Usage:
    # Via Docker Compose
    docker compose --profile recovery run --rm rebuild-done

    # Direct execution
    python scripts/rebuild_done_set.py --output /data/output
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import redis
except ImportError:
    print("ERROR: redis 패키지가 필요합니다. pip install redis")
    sys.exit(1)

KEY_DONE = "file:done"
KEY_QUEUE = "file:queue"
KEY_QUEUE_SET = "file:queue:set"
KEY_PROCESSING = "file:processing"
KEY_PROCESSING_SET = "file:processing:set"
KEY_FAILED = "file:failed"


def find_output_files(output_dir: Path) -> list[str]:
    """출력 디렉토리에서 YAML 파일 목록 수집 및 원본 파일명 변환."""
    done_files = []

    for yaml_path in output_dir.rglob("*.yaml"):
        # 출력 경로에서 상대 경로 추출
        try:
            relative = yaml_path.relative_to(output_dir)
        except ValueError:
            continue

        # .yaml → 원본 확장자 (.pdf, .hwp, .hwpx) 변환
        # 출력: 생명보험사/DB생명/파일.yaml
        # 원본: 생명보험사/DB생명/파일.pdf
        base_path = relative.with_suffix("")

        # 원본 확장자 추정 (대부분 PDF)
        for ext in [".pdf", ".hwp", ".hwpx"]:
            original_name = str(base_path) + ext
            done_files.append(original_name)
            break  # 첫 번째 확장자만 (PDF 우선)

    return done_files


def recover_processing_list(r: redis.Redis, output_dir: Path) -> dict:
    """file:processing 리스트 정리.

    출력 파일이 존재하면 → done으로 이동
    출력 파일이 없으면 → queue로 재삽입

    Returns:
        dict: {moved_to_done: int, moved_to_queue: int}
    """
    result = {"moved_to_done": 0, "moved_to_queue": 0}

    processing_items = r.lrange(KEY_PROCESSING, 0, -1)
    if not processing_items:
        print("file:processing is empty, nothing to recover")
        return result

    print(f"Found {len(processing_items)} items in file:processing")

    for file_name in processing_items:
        # 출력 파일 경로 추정
        base_path = Path(file_name).with_suffix("")
        output_path = output_dir / f"{base_path}.yaml"

        if output_path.exists():
            # 출력 존재 → done으로 이동
            r.sadd(KEY_DONE, file_name)
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            result["moved_to_done"] += 1
        else:
            # 출력 미존재 → queue로 재삽입
            r.lrem(KEY_PROCESSING, 0, file_name)
            r.srem(KEY_PROCESSING_SET, file_name)
            # done이나 failed에 없으면 재큐잉
            if not r.sismember(KEY_DONE, file_name) and not r.sismember(KEY_FAILED, file_name):
                r.sadd(KEY_QUEUE_SET, file_name)
                r.lpush(KEY_QUEUE, file_name)
                result["moved_to_queue"] += 1

    return result


def verify_queue_consistency(r: redis.Redis) -> dict:
    """file:queue와 file:queue:set 일관성 검증 및 복구.

    Returns:
        dict: {in_list_not_set: int, in_set_not_list: int, fixed: int}
    """
    result = {"in_list_not_set": 0, "in_set_not_list": 0, "fixed": 0}

    # 리스트에 있으나 세트에 없는 항목
    queue_list = r.lrange(KEY_QUEUE, 0, -1)
    queue_set = r.smembers(KEY_QUEUE_SET)

    queue_list_set = set(queue_list)

    # 리스트에 있으나 세트에 없음 → 세트에 추가
    for item in queue_list_set:
        if item not in queue_set:
            r.sadd(KEY_QUEUE_SET, item)
            result["in_list_not_set"] += 1
            result["fixed"] += 1

    # 세트에 있으나 리스트에 없음 → 리스트에 추가
    for item in queue_set:
        if item not in queue_list_set:
            r.lpush(KEY_QUEUE, item)
            result["in_set_not_list"] += 1
            result["fixed"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Rebuild done set from output files")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", "/output")),
        help="Output directory containing YAML files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (print files only)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge with existing done set (default: True)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing done set entirely",
    )
    parser.add_argument(
        "--full-recovery",
        action="store_true",
        help="Full recovery: done + processing cleanup + queue consistency",
    )
    args = parser.parse_args()

    output_dir = args.output
    print(f"Output directory: {output_dir}")

    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)

    # YAML 파일 목록 수집
    print("Scanning output files...")
    done_files = find_output_files(output_dir)
    print(f"Found {len(done_files)} output files")

    if args.dry_run:
        for f in done_files[:20]:
            print(f"  {f}")
        if len(done_files) > 20:
            print(f"  ... and {len(done_files) - 20} more")
        return

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

    # 기존 done 세트 상태
    existing_count = r.scard(KEY_DONE)
    print(f"Existing done count: {existing_count}")

    if args.replace:
        print("Replacing existing done set...")
        r.delete(KEY_DONE)

    # 배치로 done 세트에 추가
    batch_size = 1000
    added = 0

    for i in range(0, len(done_files), batch_size):
        batch = done_files[i : i + batch_size]
        if batch:
            added += r.sadd(KEY_DONE, *batch)

        if (i // batch_size) % 50 == 0:
            print(f"Processed {i + len(batch)} files...")

    # 결과 출력
    final_count = r.scard(KEY_DONE)
    print("\n=== Done Set Rebuild Complete ===")
    print(f"Files from output: {len(done_files)}")
    print(f"Actually added: {added}")
    print(f"Final done count: {final_count}")

    # Full recovery 모드: processing 정리 + 큐 일관성 검증
    if args.full_recovery:
        print("\n=== Full Recovery Mode ===")

        # 1. Processing 리스트 정리
        print("\n[Step 1] Recovering file:processing...")
        proc_result = recover_processing_list(r, output_dir)
        print(f"  Moved to done: {proc_result['moved_to_done']}")
        print(f"  Moved to queue: {proc_result['moved_to_queue']}")

        # 2. 큐 일관성 검증
        print("\n[Step 2] Verifying queue consistency...")
        queue_result = verify_queue_consistency(r)
        print(f"  In list not set: {queue_result['in_list_not_set']}")
        print(f"  In set not list: {queue_result['in_set_not_list']}")
        print(f"  Total fixed: {queue_result['fixed']}")

        # 최종 상태 출력
        print("\n=== Final State ===")
        print(f"  file:queue: {r.llen(KEY_QUEUE)}")
        print(f"  file:processing: {r.llen(KEY_PROCESSING)}")
        print(f"  file:done: {r.scard(KEY_DONE)}")
        print(f"  file:failed: {r.scard(KEY_FAILED)}")


if __name__ == "__main__":
    main()
