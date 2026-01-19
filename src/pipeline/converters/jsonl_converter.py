"""JSONL conversion utilities for training data.

다양한 형식으로 JSONL 변환, 병합, 분할 기능 제공.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from src.pipeline.converters.factory import OutputFormat


class JSONLConverter:
    """JSONL 형식 변환기.

    TrainingExample을 다양한 LLM 학습 형식으로 변환.

    Args:
        output_format: 출력 형식
        include_metadata: 메타데이터 포함 여부

    Example:
        >>> converter = JSONLConverter(output_format=OutputFormat.OPENAI)
        >>> converter.export(examples, "output.jsonl")
    """

    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.OPENAI,
        include_metadata: bool = True,
    ):
        self.output_format = output_format
        self.include_metadata = include_metadata

    def convert(self, example: Any) -> dict[str, Any]:
        """단일 예제 변환.

        Args:
            example: TrainingExample 또는 dict

        Returns:
            dict: 변환된 데이터
        """
        if hasattr(example, "model_dump"):
            data = example.model_dump()
        elif isinstance(example, dict):
            data = example
        else:
            raise ValueError(f"Unsupported example type: {type(example)}")

        if self.output_format == OutputFormat.OPENAI:
            return self._to_openai(data)
        elif self.output_format == OutputFormat.ALPACA:
            return self._to_alpaca(data)
        elif self.output_format == OutputFormat.SHAREGPT:
            return self._to_sharegpt(data)
        elif self.output_format == OutputFormat.QWEN3:
            return self._to_qwen3(data)
        elif self.output_format == OutputFormat.YAML:
            return data
        else:
            return data

    def _to_openai(self, data: dict) -> dict:
        """OpenAI Chat format."""
        messages = []
        if data.get("instruction"):
            messages.append({"role": "system", "content": data["instruction"]})
        if data.get("input"):
            messages.append({"role": "user", "content": data["input"]})
        messages.append({"role": "assistant", "content": data.get("output", "")})

        result = {"messages": messages}
        if self.include_metadata:
            result["_meta"] = {k: v for k, v in data.items() if k not in ("instruction", "input", "output", "messages")}
        return result

    def _to_alpaca(self, data: dict) -> dict:
        """Alpaca instruction format."""
        result = {
            "instruction": data.get("instruction", ""),
            "input": data.get("input", ""),
            "output": data.get("output", ""),
        }
        if self.include_metadata:
            result["_meta"] = {k: v for k, v in data.items() if k not in ("instruction", "input", "output")}
        return result

    def _to_sharegpt(self, data: dict) -> dict:
        """ShareGPT conversation format."""
        conversations = []
        if data.get("instruction"):
            conversations.append({"from": "system", "value": data["instruction"]})
        if data.get("input"):
            conversations.append({"from": "human", "value": data["input"]})
        conversations.append({"from": "gpt", "value": data.get("output", "")})

        result = {"conversations": conversations}
        if self.include_metadata:
            result["_meta"] = {
                k: v for k, v in data.items() if k not in ("instruction", "input", "output", "conversations")
            }
        return result

    def _to_qwen3(self, data: dict) -> dict:
        """Qwen3 ChatML format with thinking support."""
        messages = []
        if data.get("instruction"):
            messages.append({"role": "system", "content": data["instruction"]})
        if data.get("input"):
            messages.append({"role": "user", "content": data["input"]})
        messages.append({"role": "assistant", "content": data.get("output", "")})

        result = {
            "type": "chatml",
            "messages": messages,
        }
        if self.include_metadata:
            result["metadata"] = {
                k: v for k, v in data.items() if k not in ("instruction", "input", "output", "messages", "type")
            }
        return result

    def convert_batch(self, examples: list[Any]) -> list[dict[str, Any]]:
        """배치 변환."""
        return [self.convert(ex) for ex in examples]

    def export(self, examples: list[Any], output_path: str | Path) -> int:
        """JSONL 파일로 내보내기.

        Args:
            examples: 내보낼 예제 목록
            output_path: 출력 파일 경로

        Returns:
            int: 내보낸 레코드 수
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                converted = self.convert(example)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1

        return count


class MultiFormatConverter:
    """다중 형식 동시 변환기.

    여러 형식으로 동시에 내보내기.

    Args:
        formats: 출력 형식 목록
        include_metadata: 메타데이터 포함 여부
    """

    def __init__(
        self,
        formats: list[OutputFormat],
        include_metadata: bool = True,
    ):
        self.formats = formats
        self.converters = {fmt: JSONLConverter(output_format=fmt, include_metadata=include_metadata) for fmt in formats}

    def export_all(
        self,
        examples: list[Any],
        output_dir: str | Path,
        base_name: str = "data",
    ) -> dict[OutputFormat, int]:
        """모든 형식으로 내보내기.

        Args:
            examples: 예제 목록
            output_dir: 출력 디렉터리
            base_name: 기본 파일명

        Returns:
            dict[OutputFormat, int]: 형식별 내보낸 레코드 수
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for fmt, converter in self.converters.items():
            output_path = output_dir / f"{base_name}_{fmt.value}.jsonl"
            count = converter.export(examples, output_path)
            results[fmt] = count

        return results


class JSONLReader:
    """JSONL 파일 읽기 유틸리티."""

    @staticmethod
    def read(file_path: str | Path) -> list[dict[str, Any]]:
        """전체 파일 읽기.

        Args:
            file_path: 파일 경로

        Returns:
            list[dict]: 레코드 목록
        """
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def read_streaming(file_path: str | Path) -> Iterator[dict[str, Any]]:
        """스트리밍 읽기.

        Args:
            file_path: 파일 경로

        Yields:
            dict: 각 레코드
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def count_records(file_path: str | Path) -> int:
        """레코드 수 카운트.

        Args:
            file_path: 파일 경로

        Returns:
            int: 레코드 수
        """
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class JSONLMerger:
    """JSONL 파일 병합 유틸리티."""

    @staticmethod
    def merge(
        input_files: list[str | Path],
        output_file: str | Path,
        deduplicate: bool = True,
        shuffle: bool = False,
        seed: int = 42,
    ) -> int:
        """여러 JSONL 파일 병합.

        Args:
            input_files: 입력 파일 목록
            output_file: 출력 파일 경로
            deduplicate: 중복 제거 여부
            shuffle: 셔플 여부
            seed: 랜덤 시드

        Returns:
            int: 병합된 레코드 수
        """
        records = []
        seen_hashes = set()

        for file_path in input_files:
            for record in JSONLReader.read_streaming(file_path):
                if deduplicate:
                    record_hash = hash(json.dumps(record, sort_keys=True))
                    if record_hash in seen_hashes:
                        continue
                    seen_hashes.add(record_hash)
                records.append(record)

        if shuffle:
            random.seed(seed)
            random.shuffle(records)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return len(records)


class JSONLSplitter:
    """JSONL 파일 분할 유틸리티."""

    @staticmethod
    def split(
        input_file: str | Path,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> dict[str, int]:
        """Train/Val/Test 분할.

        Args:
            input_file: 입력 파일 경로
            output_dir: 출력 디렉터리
            train_ratio: 훈련 세트 비율
            val_ratio: 검증 세트 비율
            test_ratio: 테스트 세트 비율
            seed: 랜덤 시드
            shuffle: 셔플 여부

        Returns:
            dict[str, int]: 분할별 레코드 수
        """
        records = JSONLReader.read(input_file)

        if shuffle:
            random.seed(seed)
            random.shuffle(records)

        n = len(records)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            "train": records[:train_end],
            "val": records[train_end:val_end],
            "test": records[val_end:],
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        counts = {}
        for split_name, split_records in splits.items():
            output_path = output_dir / f"{split_name}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for record in split_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split_name] = len(split_records)

        return counts


__all__ = [
    "JSONLConverter",
    "MultiFormatConverter",
    "JSONLReader",
    "JSONLMerger",
    "JSONLSplitter",
]
