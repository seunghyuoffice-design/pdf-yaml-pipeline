"""format_utils regression tests.

These tests lock down conversion invariants so refactors don't silently change
training data formats.
"""

from pdf_yaml_pipeline.converters.format_utils import to_openai, to_qwen3, to_sharegpt


def test_openai_does_not_emit_empty_assistant_message():
    data = {"instruction": "SYS", "input": "USER", "output": ""}
    result = to_openai(data, include_metadata=False)

    assert "messages" in result
    assert [m["role"] for m in result["messages"]] == ["system", "user"]


def test_sharegpt_does_not_emit_empty_gpt_message():
    data = {"instruction": "SYS", "input": "USER", "output": ""}
    result = to_sharegpt(data, include_metadata=False)

    assert "conversations" in result
    assert [m["from"] for m in result["conversations"]] == ["system", "human"]


def test_message_order_is_stable():
    data = {"instruction": "SYS", "input": "USER", "output": "ASSISTANT"}

    openai = to_openai(data, include_metadata=False)
    assert [m["role"] for m in openai["messages"]] == ["system", "user", "assistant"]

    sharegpt = to_sharegpt(data, include_metadata=False)
    assert [m["from"] for m in sharegpt["conversations"]] == ["system", "human", "gpt"]


def test_metadata_inclusion_respects_include_metadata_and_explicit_metadata():
    data = {"instruction": "SYS", "input": "USER", "output": "OK", "foo": "bar"}

    disabled = to_openai(data, include_metadata=False)
    assert "_meta" not in disabled

    auto_meta = to_openai(data, include_metadata=True)
    assert auto_meta["_meta"] == {"foo": "bar"}

    explicit_meta = to_openai(data, include_metadata=True, explicit_metadata={"source": "unit-test"})
    assert explicit_meta["_meta"] == {"source": "unit-test"}


def test_qwen3_chatml_and_thinking_insertion_rules():
    data = {"instruction": "SYS", "input": "USER", "output": "x" * 101, "foo": "bar"}

    no_meta = to_qwen3(data, include_metadata=False)
    assert no_meta["type"] == "chatml"
    assert "metadata" not in no_meta

    with_meta = to_qwen3(data, include_metadata=True, add_thinking=True, thinking_threshold=100, thinking_text="THINK")
    assert with_meta["type"] == "chatml"
    assert with_meta["metadata"] == {"foo": "bar"}
    assert "<think>" in with_meta["messages"][-1]["content"]
    assert "THINK" in with_meta["messages"][-1]["content"]
    assert with_meta["messages"][-1]["content"].endswith("x" * 101)

    short = {"instruction": "SYS", "input": "USER", "output": "x" * 100}
    no_think = to_qwen3(short, include_metadata=False, add_thinking=True, thinking_threshold=100, thinking_text="THINK")
    assert "<think>" not in no_think["messages"][-1]["content"]
