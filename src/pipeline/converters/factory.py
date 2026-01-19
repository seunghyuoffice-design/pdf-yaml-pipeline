"""Output format factory for training data conversion.

다양한 LLM 학습 형식으로 변환하기 위한 팩토리.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class OutputFormat(str, Enum):
    """지원하는 출력 형식.

    Attributes:
        OPENAI: OpenAI Chat format (messages array)
        ALPACA: Alpaca instruction format
        SHAREGPT: ShareGPT conversation format
        QWEN3: Qwen3 ChatML format with thinking support
        YAML: YAML 형식
        CUSTOM: 사용자 정의 형식
    """

    OPENAI = "openai"
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    QWEN3 = "qwen3"
    YAML = "yaml"
    CUSTOM = "custom"


@runtime_checkable
class FormatConverter(Protocol):
    """형식 변환기 프로토콜."""

    output_format: OutputFormat

    def convert(self, example: Any) -> dict[str, Any]:
        """단일 예제 변환."""
        ...

    def convert_batch(self, examples: list[Any]) -> list[dict[str, Any]]:
        """배치 변환."""
        ...


def get_format_schema(format_type: OutputFormat) -> dict[str, Any]:
    """형식별 JSON 스키마 반환.

    Args:
        format_type: 출력 형식

    Returns:
        dict: JSON Schema
    """
    schemas = {
        OutputFormat.OPENAI: {
            "type": "object",
            "required": ["messages"],
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["system", "user", "assistant"],
                            },
                            "content": {"type": "string"},
                        },
                    },
                },
            },
        },
        OutputFormat.ALPACA: {
            "type": "object",
            "required": ["instruction", "output"],
            "properties": {
                "instruction": {"type": "string"},
                "input": {"type": "string"},
                "output": {"type": "string"},
            },
        },
        OutputFormat.SHAREGPT: {
            "type": "object",
            "required": ["conversations"],
            "properties": {
                "conversations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["from", "value"],
                        "properties": {
                            "from": {
                                "type": "string",
                                "enum": ["human", "gpt", "system"],
                            },
                            "value": {"type": "string"},
                        },
                    },
                },
            },
        },
        OutputFormat.QWEN3: {
            "type": "object",
            "required": ["type", "messages"],
            "properties": {
                "type": {"type": "string", "const": "chatml"},
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["system", "user", "assistant"],
                            },
                            "content": {"type": "string"},
                        },
                    },
                },
            },
        },
        OutputFormat.YAML: {
            "type": "object",
            "required": ["instruction", "output"],
            "properties": {
                "instruction": {"type": "string"},
                "input": {"type": "string"},
                "output": {"type": "string"},
                "source": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    }
    return schemas.get(format_type, {"type": "object"})


__all__ = [
    "OutputFormat",
    "FormatConverter",
    "get_format_schema",
]
