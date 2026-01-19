"""Auto-fix utilities for JSONL quality issues.

자동으로 품질 문제를 감지하고 수정.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.pipeline.converters.factory import OutputFormat


class AutoFixer:
    """JSONL 품질 문제 자동 수정기.

    Args:
        output_format: 대상 출력 형식
        config: 수정 설정

    Example:
        >>> fixer = AutoFixer(output_format=OutputFormat.OPENAI)
        >>> stats = fixer.fix_jsonl_file("input.jsonl", "output.jsonl")
    """

    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.OPENAI,
        config: dict[str, Any] | None = None,
    ):
        self.output_format = output_format
        self.config = config or {
            "remove_empty_fields": True,
            "clean_special_chars": True,
            "fix_encoding": True,
            "normalize_whitespace": True,
            "remove_control_chars": True,
        }

    def fix_record(self, record: dict[str, Any]) -> tuple[dict[str, Any] | None, bool, list[str]]:
        """단일 레코드 수정.

        Args:
            record: 수정할 레코드

        Returns:
            tuple: (수정된 레코드 또는 None, 수정됨 여부, 수정 내용 목록)
        """
        fixes_applied = []
        fixed = record.copy()
        was_modified = False

        # 1. 인코딩 수정
        if self.config.get("fix_encoding"):
            fixed, encoding_fixes = self._fix_encoding(fixed)
            if encoding_fixes:
                fixes_applied.extend(encoding_fixes)
                was_modified = True

        # 2. 제어 문자 제거
        if self.config.get("remove_control_chars"):
            fixed, control_fixes = self._remove_control_chars(fixed)
            if control_fixes:
                fixes_applied.extend(control_fixes)
                was_modified = True

        # 3. 특수 문자 정리
        if self.config.get("clean_special_chars"):
            fixed, special_fixes = self._clean_special_chars(fixed)
            if special_fixes:
                fixes_applied.extend(special_fixes)
                was_modified = True

        # 4. 공백 정규화
        if self.config.get("normalize_whitespace"):
            fixed, ws_fixes = self._normalize_whitespace(fixed)
            if ws_fixes:
                fixes_applied.extend(ws_fixes)
                was_modified = True

        # 5. 빈 필드 제거
        if self.config.get("remove_empty_fields"):
            fixed, empty_fixes = self._remove_empty_fields(fixed)
            if empty_fixes:
                fixes_applied.extend(empty_fixes)
                was_modified = True

        # 6. 형식별 유효성 검사
        is_valid = self._validate_format(fixed)
        if not is_valid:
            return None, False, ["Invalid format - cannot fix"]

        return fixed, was_modified, fixes_applied

    # Common mojibake patterns (UTF-8 misinterpreted as cp1252/latin1)
    MOJIBAKE_PATTERNS = [
        "â€",  # Common UTF-8 multibyte misread
        "Ã",  # Latin characters with diacritics
        "ì",  # Korean mojibake patterns
        "í",
        "ë",
        "ê",
        "\ufffd",  # Replacement character
    ]

    def _fix_encoding(self, record: dict) -> tuple[dict, list[str]]:
        """인코딩 문제 수정."""
        fixes = []

        def has_mojibake_pattern(s: str) -> bool:
            """Check if string contains mojibake patterns."""
            return any(pattern in s for pattern in self.MOJIBAKE_PATTERNS)

        def try_fix_mojibake(s: str) -> str:
            """Try to fix mojibake with multiple encoding combinations."""
            if not has_mojibake_pattern(s):
                return s

            # Encoding combinations to try (source → decode)
            encoding_pairs = [
                ("cp1252", "utf-8"),  # Most common
                ("latin1", "utf-8"),  # ISO-8859-1
                ("cp949", "utf-8"),  # Korean
                ("euc-kr", "utf-8"),  # Korean legacy
            ]

            best_result = s
            best_score = 0

            for src_enc, dest_enc in encoding_pairs:
                try:
                    fixed = s.encode(src_enc).decode(dest_enc)
                    # Score: fewer replacement chars and mojibake patterns is better
                    score = len(fixed) - fixed.count("\ufffd") * 10
                    score -= sum(fixed.count(p) for p in self.MOJIBAKE_PATTERNS) * 5

                    if score > best_score and not has_mojibake_pattern(fixed):
                        best_score = score
                        best_result = fixed
                except (UnicodeDecodeError, UnicodeEncodeError):
                    continue

            return best_result

        def fix_string(s: str) -> str:
            # UTF-8 정규화
            normalized = unicodedata.normalize("NFC", s)
            # mojibake 복구 시도
            return try_fix_mojibake(normalized)

        def process_value(v: Any, path: str) -> Any:
            if isinstance(v, str):
                fixed = fix_string(v)
                if fixed != v:
                    fixes.append(f"Fixed encoding at {path}")
                return fixed
            elif isinstance(v, dict):
                return {k: process_value(val, f"{path}.{k}") for k, val in v.items()}
            elif isinstance(v, list):
                return [process_value(item, f"{path}[{i}]") for i, item in enumerate(v)]
            return v

        return process_value(record, "root"), fixes

    def _remove_control_chars(self, record: dict) -> tuple[dict, list[str]]:
        """제어 문자 제거."""
        fixes = []
        # 허용되는 제어 문자: 탭, 줄바꿈, 캐리지 리턴
        control_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

        def clean_string(s: str) -> str:
            cleaned = control_pattern.sub("", s)
            if cleaned != s:
                fixes.append("Removed control characters")
            return cleaned

        def process_value(v: Any) -> Any:
            if isinstance(v, str):
                return clean_string(v)
            elif isinstance(v, dict):
                return {k: process_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [process_value(item) for item in v]
            return v

        return process_value(record), fixes

    def _clean_special_chars(self, record: dict) -> tuple[dict, list[str]]:
        """과도한 특수 문자 정리."""
        fixes = []

        def clean_string(s: str) -> str:
            original = s
            # 연속된 특수 문자 정리
            s = re.sub(r"[!@#$%^&*()]{3,}", "", s)
            # 연속된 구두점 정리
            s = re.sub(r"\.{4,}", "...", s)
            s = re.sub(r"-{4,}", "---", s)
            s = re.sub(r"_{4,}", "___", s)
            # 연속된 물결표 정리
            s = re.sub(r"~{3,}", "~~", s)

            if s != original:
                fixes.append("Cleaned special characters")
            return s

        def process_value(v: Any) -> Any:
            if isinstance(v, str):
                return clean_string(v)
            elif isinstance(v, dict):
                return {k: process_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [process_value(item) for item in v]
            return v

        return process_value(record), fixes

    def _normalize_whitespace(self, record: dict) -> tuple[dict, list[str]]:
        """공백 정규화."""
        fixes = []

        def normalize_string(s: str) -> str:
            original = s
            # 연속 공백을 단일 공백으로
            s = re.sub(r" {2,}", " ", s)
            # 연속 줄바꿈을 최대 2개로
            s = re.sub(r"\n{3,}", "\n\n", s)
            # 줄 끝 공백 제거
            s = re.sub(r" +\n", "\n", s)
            # 줄 시작 공백 제거 (들여쓰기 제외)
            s = re.sub(r"\n +", "\n", s)
            # 앞뒤 공백 제거
            s = s.strip()

            if s != original:
                fixes.append("Normalized whitespace")
            return s

        def process_value(v: Any) -> Any:
            if isinstance(v, str):
                return normalize_string(v)
            elif isinstance(v, dict):
                return {k: process_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [process_value(item) for item in v]
            return v

        return process_value(record), fixes

    def _remove_empty_fields(self, record: dict) -> tuple[dict, list[str]]:
        """빈 필드 제거."""
        fixes = []

        def is_empty(v: Any) -> bool:
            if v is None:
                return True
            if isinstance(v, str) and not v.strip():
                return True
            if isinstance(v, (list, dict)) and len(v) == 0:
                return True
            return False

        def clean_dict(d: dict, path: str = "") -> dict:
            cleaned = {}
            for k, v in d.items():
                current_path = f"{path}.{k}" if path else k
                if isinstance(v, dict):
                    v = clean_dict(v, current_path)
                    if v:  # 빈 dict 제외
                        cleaned[k] = v
                elif isinstance(v, list):
                    v = [
                        clean_dict(item, f"{current_path}[{i}]") if isinstance(item, dict) else item
                        for i, item in enumerate(v)
                        if not is_empty(item)
                    ]
                    if v:  # 빈 list 제외
                        cleaned[k] = v
                elif not is_empty(v):
                    cleaned[k] = v
                else:
                    fixes.append(f"Removed empty field: {current_path}")
            return cleaned

        return clean_dict(record), fixes

    def _validate_format(self, record: dict) -> bool:
        """형식별 유효성 검사."""
        if self.output_format == OutputFormat.OPENAI:
            if "messages" not in record:
                return False
            if not isinstance(record["messages"], list):
                return False
            for msg in record["messages"]:
                if "role" not in msg or "content" not in msg:
                    return False
            return True

        elif self.output_format == OutputFormat.ALPACA:
            return "instruction" in record and "output" in record

        elif self.output_format == OutputFormat.SHAREGPT:
            if "conversations" not in record:
                return False
            for conv in record.get("conversations", []):
                if "from" not in conv or "value" not in conv:
                    return False
            return True

        elif self.output_format == OutputFormat.QWEN3:
            return "messages" in record

        return True

    def fix_jsonl_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sampling_rate: float = 1.0,
    ) -> dict[str, int]:
        """JSONL 파일 수정.

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로
            sampling_rate: 샘플링 비율 (0.0-1.0)

        Returns:
            dict: 통계
        """
        import random

        stats = {
            "total_records": 0,
            "processed_records": 0,
            "valid_records": 0,
            "fixed_records": 0,
            "unfixable_records": 0,
        }

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            open(input_path, "r", encoding="utf-8") as f_in,
            open(output_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                stats["total_records"] += 1
                line = line.strip()
                if not line:
                    continue

                # 샘플링
                if sampling_rate < 1.0 and random.random() > sampling_rate:
                    continue

                stats["processed_records"] += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["unfixable_records"] += 1
                    continue

                fixed, was_modified, fixes = self.fix_record(record)

                if fixed is None:
                    stats["unfixable_records"] += 1
                    continue

                if was_modified:
                    stats["fixed_records"] += 1
                else:
                    stats["valid_records"] += 1

                f_out.write(json.dumps(fixed, ensure_ascii=False) + "\n")

        return stats


__all__ = ["AutoFixer"]
