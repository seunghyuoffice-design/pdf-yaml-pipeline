"""Auto validation utilities for JSONL quality checking.

JSONL 파일의 품질을 자동으로 검증하고 필터링.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def validate_jsonl_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    min_score: float = 0.7,
    include_scores: bool = True,
) -> dict[str, Any]:
    """JSONL 파일 검증 및 필터링.

    Args:
        input_path: 입력 JSONL 파일 경로
        output_path: 출력 파일 경로 (None이면 필터링 안 함)
        min_score: 최소 품질 점수
        include_scores: 출력에 점수 포함 여부

    Returns:
        dict: 검증 통계
    """
    input_path = Path(input_path)
    stats = {
        "total_records": 0,
        "valid_records": 0,
        "filtered_records": 0,
        "invalid_records": 0,
        "scores": {
            "structure": [],
            "completeness": [],
            "quality": [],
            "overall": [],
        },
        "issues": [],
    }

    records_to_write = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total_records"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                stats["invalid_records"] += 1
                stats["issues"].append(
                    {
                        "line": line_num,
                        "type": "json_decode_error",
                        "message": str(e),
                    }
                )
                continue

            # 품질 점수 계산
            scores = calculate_quality_scores(record)
            overall_score = scores["overall"]

            stats["scores"]["structure"].append(scores["structure"])
            stats["scores"]["completeness"].append(scores["completeness"])
            stats["scores"]["quality"].append(scores["quality"])
            stats["scores"]["overall"].append(overall_score)

            if overall_score >= min_score:
                stats["valid_records"] += 1
                if include_scores:
                    record["_quality_score"] = overall_score
                records_to_write.append(record)
            else:
                stats["filtered_records"] += 1

    # 출력 파일 작성
    if output_path and records_to_write:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records_to_write:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 평균 점수 계산
    for key in ["structure", "completeness", "quality", "overall"]:
        scores_list = stats["scores"][key]
        if scores_list:
            stats[f"avg_{key}_score"] = sum(scores_list) / len(scores_list)
        else:
            stats[f"avg_{key}_score"] = 0.0

    return stats


def calculate_quality_scores(record: dict[str, Any]) -> dict[str, float]:
    """레코드 품질 점수 계산.

    Args:
        record: 검사할 레코드

    Returns:
        dict: 각 카테고리별 점수
    """
    scores = {
        "structure": 0.0,
        "completeness": 0.0,
        "quality": 0.0,
        "overall": 0.0,
    }

    # 1. 구조 점수
    structure_score = calculate_structure_score(record)

    # 2. 완성도 점수
    completeness_score = calculate_completeness_score(record)

    # 3. 텍스트 품질 점수
    quality_score = calculate_text_quality_score(record)

    scores["structure"] = structure_score
    scores["completeness"] = completeness_score
    scores["quality"] = quality_score

    # 전체 점수 (가중 평균)
    scores["overall"] = structure_score * 0.3 + completeness_score * 0.3 + quality_score * 0.4

    return scores


def calculate_structure_score(record: dict) -> float:
    """구조 점수 계산."""
    score = 0.0
    max_score = 0.0

    # OpenAI format 체크
    if "messages" in record:
        max_score += 1.0
        messages = record["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            score += 1.0

        # 역할 다양성 체크
        max_score += 0.5
        roles = set(m.get("role") for m in messages if isinstance(m, dict))
        if len(roles) >= 2:
            score += 0.5

    # Alpaca format 체크
    elif "instruction" in record:
        max_score += 1.0
        if record.get("instruction"):
            score += 0.5
        if record.get("output"):
            score += 0.5

    # ShareGPT format 체크
    elif "conversations" in record:
        max_score += 1.0
        convs = record["conversations"]
        if isinstance(convs, list) and len(convs) > 0:
            score += 1.0

    else:
        max_score += 1.0
        # 기본 텍스트 필드 존재 여부
        for key in ["text", "content", "input", "output"]:
            if key in record and record[key]:
                score += 0.25

    return score / max_score if max_score > 0 else 0.0


def calculate_completeness_score(record: dict) -> float:
    """완성도 점수 계산."""
    scores = []

    # 텍스트 추출
    texts = extract_texts(record)

    for text in texts:
        if not text:
            scores.append(0.0)
            continue

        text_score = 0.0

        # 길이 체크 (최소 50자)
        if len(text) >= 50:
            text_score += 0.3
        elif len(text) >= 20:
            text_score += 0.15

        # 단어 수 체크 (최소 10단어)
        word_count = len(text.split())
        if word_count >= 10:
            text_score += 0.3
        elif word_count >= 5:
            text_score += 0.15

        # 문장 완성도 체크
        if text.rstrip().endswith((".", "!", "?", "다.", "요.", "습니다.")):
            text_score += 0.4

        scores.append(text_score)

    return sum(scores) / len(scores) if scores else 0.0


def calculate_text_quality_score(record: dict) -> float:
    """텍스트 품질 점수 계산."""
    texts = extract_texts(record)
    scores = []

    for text in texts:
        if not text:
            scores.append(0.0)
            continue

        text_score = 1.0

        # 1. 가비지 패턴 감지
        garbage_patterns = [
            r"[!@#$%^&*()]{4,}",  # 과도한 특수문자
            r"(.)\1{5,}",  # 5회 이상 반복 문자
            r"[\x00-\x08\x0b\x0c\x0e-\x1f]",  # 제어 문자
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                text_score -= 0.2

        # 2. 반복 패턴 감지
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # 70% 이상 반복
                text_score -= 0.3

        # 3. 한글 비율 체크 (한국어 문서의 경우)
        korean_chars = len(re.findall(r"[가-힣]", text))
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            # 한글 문서로 추정되면 한글 비율 체크
            if korean_ratio > 0.1 and korean_ratio < 0.3:
                text_score -= 0.1

        # 4. 빈 줄 과다 체크
        empty_line_ratio = text.count("\n\n") / max(len(text), 1)
        if empty_line_ratio > 0.1:
            text_score -= 0.1

        scores.append(max(0.0, text_score))

    return sum(scores) / len(scores) if scores else 0.0


def extract_texts(record: dict) -> list[str]:
    """레코드에서 텍스트 추출."""
    texts = []

    # OpenAI format
    if "messages" in record:
        for msg in record["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                texts.append(msg["content"])

    # Alpaca format
    elif "instruction" in record:
        texts.append(record.get("instruction", ""))
        texts.append(record.get("input", ""))
        texts.append(record.get("output", ""))

    # ShareGPT format
    elif "conversations" in record:
        for conv in record["conversations"]:
            if isinstance(conv, dict) and "value" in conv:
                texts.append(conv["value"])

    # 기타
    else:
        for key in ["text", "content", "input", "output"]:
            if key in record:
                texts.append(str(record[key]))

    return [t for t in texts if t]


def print_validation_report(stats: dict[str, Any]) -> None:
    """검증 리포트 출력."""
    print("\n" + "=" * 50)
    print("Quality Validation Report")
    print("=" * 50)
    print(f"Total records:    {stats['total_records']:,}")
    print(f"Valid records:    {stats['valid_records']:,}")
    print(f"Filtered records: {stats['filtered_records']:,}")
    print(f"Invalid records:  {stats['invalid_records']:,}")
    print()
    print("Average Scores:")
    print(f"  Structure:    {stats.get('avg_structure_score', 0):.2%}")
    print(f"  Completeness: {stats.get('avg_completeness_score', 0):.2%}")
    print(f"  Quality:      {stats.get('avg_quality_score', 0):.2%}")
    print(f"  Overall:      {stats.get('avg_overall_score', 0):.2%}")

    if stats["issues"]:
        print()
        print(f"Issues ({len(stats['issues'])}):")
        for issue in stats["issues"][:5]:
            print(f"  Line {issue['line']}: {issue['type']} - {issue['message'][:50]}")


__all__ = [
    "validate_jsonl_file",
    "calculate_quality_scores",
    "print_validation_report",
]
