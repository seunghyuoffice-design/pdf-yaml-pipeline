"""YAMLConverter unit tests.

Tests for document-to-training-example conversion.
"""

import pytest
from pdf_yaml_pipeline.converters.factory import OutputFormat
from pdf_yaml_pipeline.converters.yaml_converter import YAMLConverter


class TestYAMLConverter:
    """Test YAMLConverter."""

    @pytest.fixture
    def converter(self):
        return YAMLConverter(output_format=OutputFormat.YAML)

    def test_summary_example_has_document_in_input(self, converter):
        """Test that summary examples include document text in input.

        Regression test: Previously input was just "이 문서의 주요 내용을 요약해주세요."
        without the actual document content, making the training pair meaningless.
        """
        doc = {
            "document": {"source_path": "test.pdf"},
            "content": {
                "paragraphs": [
                    "This is the first paragraph with some content.",
                    "This is the second paragraph with more content.",
                    "Third paragraph continues with important information.",
                ] * 50  # Make it > 500 chars
            },
            "tables": [],
        }

        examples = converter.convert(doc)

        # Find summary example
        summary_examples = [ex for ex in examples if ex.get("metadata", {}).get("type") == "document_summary"]

        assert len(summary_examples) > 0, "Should generate summary example for long document"

        summary = summary_examples[0]

        # Input should contain document text, not just the request
        assert "first paragraph" in summary["input"], "Input should contain document content"
        assert "주요 내용을 요약" in summary["input"], "Input should contain summary request"

        # Output should be empty for Teacher to fill
        assert summary["output"] == "", "Output should be empty for Teacher model to generate"

    def test_summary_not_generated_for_short_document(self, converter):
        """Test that summary is not generated for short documents."""
        doc = {
            "document": {"source_path": "short.pdf"},
            "content": {
                "paragraphs": ["Short text."]  # < 500 chars
            },
            "tables": [],
        }

        examples = converter.convert(doc)

        summary_examples = [ex for ex in examples if ex.get("metadata", {}).get("type") == "document_summary"]
        assert len(summary_examples) == 0, "Should not generate summary for short document"

    def test_summary_truncation_metadata(self, converter):
        """Test that truncation is recorded in metadata."""
        # Create very long document (> 8000 chars)
        long_paragraph = "A" * 1000
        doc = {
            "document": {"source_path": "long.pdf"},
            "content": {
                "paragraphs": [long_paragraph] * 20  # 20000 chars
            },
            "tables": [],
        }

        examples = converter.convert(doc)

        summary_examples = [ex for ex in examples if ex.get("metadata", {}).get("type") == "document_summary"]
        assert len(summary_examples) > 0

        metadata = summary_examples[0]["metadata"]
        assert metadata.get("truncated") is True, "Should mark as truncated"
        assert metadata.get("original_length") == 20000 + (19 * 2), "Should record original length"  # +newlines

    def test_table_example_generation(self, converter):
        """Test that table examples are generated correctly."""
        doc = {
            "document": {"source_path": "table.pdf"},
            "content": {"paragraphs": []},
            "tables": [
                {
                    "cells": [
                        {"text": "Header1", "row": 0, "col": 0},
                        {"text": "Header2", "row": 0, "col": 1},
                        {"text": "Value1", "row": 1, "col": 0},
                        {"text": "Value2", "row": 1, "col": 1},
                    ],
                    "page": 1,
                }
            ],
        }

        examples = converter.convert(doc)

        # Should have table examples
        table_examples = [ex for ex in examples if "table_index" in ex.get("metadata", {})]
        assert len(table_examples) > 0, "Should generate table example"

        # Table example should have table content in input
        table_ex = table_examples[0]
        assert "분석" in table_ex["input"], "Input should request table analysis"
        assert table_ex["output"] == "", "Output should be empty for Teacher"

    def test_empty_document(self, converter):
        """Test handling of empty document."""
        doc = {
            "document": {"source_path": "empty.pdf"},
            "content": {"paragraphs": []},
            "tables": [],
        }

        examples = converter.convert(doc)

        # Should not crash, may return empty list
        assert isinstance(examples, list)

    def test_document_type_classification(self, converter):
        """Test that document type is classified and used."""
        doc = {
            "document": {"source_path": "insurance_policy.pdf"},
            "content": {
                "paragraphs": ["보험 약관 내용입니다. 보장 범위와 특약 사항을 설명합니다."] * 20
            },
            "tables": [],
        }

        examples = converter.convert(doc)

        if examples:
            # Source should reflect document classification
            assert "source" in examples[0]

    def test_instruction_has_system_prompt(self, converter):
        """Test that examples have appropriate system prompts."""
        doc = {
            "document": {"source_path": "test.pdf"},
            "content": {
                "paragraphs": ["Content " * 100]  # > 500 chars
            },
            "tables": [],
        }

        examples = converter.convert(doc)

        for ex in examples:
            assert "instruction" in ex, "Each example should have instruction"
            assert len(ex["instruction"]) > 0, "Instruction should not be empty"
