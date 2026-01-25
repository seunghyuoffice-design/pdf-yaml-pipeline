"""Shared format conversion utilities."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

DEFAULT_QWEN3_THINKING = (
    "사용자 질문에 답변하기 위해 관련 정보를 체계적으로 정리해보겠습니다.\n"
    "1. 질문의 핵심 파악\n"
    "2. 관련 규정과 내용 검토\n"
    "3. 논리적인 답변 구성\n\n"
)

_META_EXCLUDE_DEFAULT = {"instruction", "input", "output", "messages", "conversations", "type"}


def _resolve_metadata(
    data: Mapping[str, Any],
    *,
    include_metadata: bool,
    explicit_metadata: Mapping[str, Any] | None = None,
    exclude_keys: Iterable[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, Any] | None:
    if not include_metadata:
        return None

    if explicit_metadata is not None:
        if explicit_metadata or allow_empty:
            return dict(explicit_metadata)
        return None

    if exclude_keys is None:
        exclude_keys = _META_EXCLUDE_DEFAULT

    meta = {k: v for k, v in data.items() if k not in exclude_keys}
    if meta or allow_empty:
        return meta
    return None


def to_openai(
    data: Mapping[str, Any],
    *,
    include_metadata: bool,
    explicit_metadata: Mapping[str, Any] | None = None,
    exclude_keys: Iterable[str] | None = None,
    meta_key: str = "_meta",
    allow_empty_meta: bool = False,
) -> dict[str, Any]:
    messages = []
    if data.get("instruction"):
        messages.append({"role": "system", "content": data["instruction"]})
    if data.get("input"):
        messages.append({"role": "user", "content": data["input"]})
    output = data.get("output", "")
    if output:
        messages.append({"role": "assistant", "content": output})

    result: dict[str, Any] = {"messages": messages}
    meta = _resolve_metadata(
        data,
        include_metadata=include_metadata,
        explicit_metadata=explicit_metadata,
        exclude_keys=exclude_keys,
        allow_empty=allow_empty_meta,
    )
    if meta is not None:
        result[meta_key] = meta
    return result


def to_alpaca(
    data: Mapping[str, Any],
    *,
    include_metadata: bool,
    explicit_metadata: Mapping[str, Any] | None = None,
    exclude_keys: Iterable[str] | None = None,
    meta_key: str = "_meta",
    allow_empty_meta: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "instruction": data.get("instruction", ""),
        "input": data.get("input", ""),
        "output": data.get("output", ""),
    }

    meta = _resolve_metadata(
        data,
        include_metadata=include_metadata,
        explicit_metadata=explicit_metadata,
        exclude_keys=exclude_keys,
        allow_empty=allow_empty_meta,
    )
    if meta is not None:
        result[meta_key] = meta
    return result


def to_sharegpt(
    data: Mapping[str, Any],
    *,
    include_metadata: bool,
    explicit_metadata: Mapping[str, Any] | None = None,
    exclude_keys: Iterable[str] | None = None,
    meta_key: str = "_meta",
    allow_empty_meta: bool = False,
) -> dict[str, Any]:
    conversations = []
    if data.get("instruction"):
        conversations.append({"from": "system", "value": data["instruction"]})
    if data.get("input"):
        conversations.append({"from": "human", "value": data["input"]})
    output = data.get("output", "")
    if output:
        conversations.append({"from": "gpt", "value": output})

    result: dict[str, Any] = {"conversations": conversations}
    meta = _resolve_metadata(
        data,
        include_metadata=include_metadata,
        explicit_metadata=explicit_metadata,
        exclude_keys=exclude_keys,
        allow_empty=allow_empty_meta,
    )
    if meta is not None:
        result[meta_key] = meta
    return result


def to_qwen3(
    data: Mapping[str, Any],
    *,
    include_metadata: bool,
    explicit_metadata: Mapping[str, Any] | None = None,
    exclude_keys: Iterable[str] | None = None,
    meta_key: str = "metadata",
    allow_empty_meta: bool = False,
    add_thinking: bool = False,
    thinking_threshold: int = 100,
    thinking_text: str | None = None,
) -> dict[str, Any]:
    messages = []
    if data.get("instruction"):
        messages.append({"role": "system", "content": data["instruction"]})
    if data.get("input"):
        messages.append({"role": "user", "content": data["input"]})

    output = data.get("output", "")
    if not output:
        output = ""
    if output and add_thinking and len(output) > thinking_threshold:
        thinking = thinking_text or DEFAULT_QWEN3_THINKING
        messages.append({"role": "assistant", "content": f"<think>\n{thinking}\n</think>\n{output}"})
    elif output:
        messages.append({"role": "assistant", "content": output})

    result: dict[str, Any] = {"type": "chatml", "messages": messages}
    meta = _resolve_metadata(
        data,
        include_metadata=include_metadata,
        explicit_metadata=explicit_metadata,
        exclude_keys=exclude_keys,
        allow_empty=allow_empty_meta,
    )
    if meta is not None:
        result[meta_key] = meta
    return result


__all__ = [
    "DEFAULT_QWEN3_THINKING",
    "to_openai",
    "to_alpaca",
    "to_sharegpt",
    "to_qwen3",
]
