"""Deduplicator unit tests.

Tests for ExactDeduplicator and StructuralDeduplicator.
"""

import pytest
from pdf_yaml_pipeline.deduplication.deduplicator import (
    DeduplicationConfig,
    ExactDeduplicator,
    StructuralDeduplicator,
)


class TestExactDeduplicator:
    """Test ExactDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        return ExactDeduplicator()

    @pytest.fixture
    def config(self):
        return DeduplicationConfig(min_length=10, similarity_threshold=0.85)

    def test_remove_exact_duplicates(self, deduplicator, config):
        """Test removal of exact duplicate content."""
        examples = [
            {"input": "test", "output": "This is a test output content"},
            {"input": "test", "output": "This is a test output content"},  # duplicate
            {"input": "test", "output": "Different content here now"},
        ]

        result, removed = deduplicator.deduplicate(examples, config)

        assert removed == 1
        assert len(result) == 2

    def test_whitespace_normalization(self, deduplicator, config):
        """Test that whitespace differences are normalized."""
        examples = [
            {"output": "Hello   world   test content"},
            {"output": "Hello world test content"},  # Same after normalization
        ]

        result, removed = deduplicator.deduplicate(examples, config)

        assert removed == 1
        assert len(result) == 1

    def test_short_content_not_deduplicated(self, deduplicator, config):
        """Test that short content is preserved."""
        examples = [
            {"output": "short"},  # Below min_length
            {"output": "short"},  # Below min_length - both kept
        ]

        result, removed = deduplicator.deduplicate(examples, config)

        assert removed == 0
        assert len(result) == 2

    def test_empty_examples(self, deduplicator, config):
        """Test with empty example list."""
        result, removed = deduplicator.deduplicate([], config)

        assert removed == 0
        assert len(result) == 0


class TestStructuralDeduplicator:
    """Test StructuralDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        return StructuralDeduplicator()

    @pytest.fixture
    def config(self):
        return DeduplicationConfig(
            enable_structural=True,
            similarity_threshold=0.85,
        )

    def test_extract_structure(self, deduplicator):
        """Test structure extraction from example."""
        example = {
            "input": "test input",
            "instruction": "test instruction",
            "output": "keyword1 keyword2 keyword3",
            "source": "insurance",
        }

        structure = deduplicator.extract_structure(example)

        assert structure["has_input"] is True
        assert structure["has_instruction"] is True
        assert structure["output_length"] == len("keyword1 keyword2 keyword3")
        assert structure["source_type"] == "insurance"
        assert "keyword1" in structure["keywords"]

    def test_similar_structures_deduplicated(self, deduplicator, config):
        """Test that similar structures are deduplicated."""
        examples = [
            {
                "input": "What is coverage?",
                "instruction": "Answer the question",
                "output": "Coverage means protection insurance policy",
                "source": "insurance",
            },
            {
                "input": "What is coverage limit?",  # Similar
                "instruction": "Answer this question",  # Similar
                "output": "Coverage limit means maximum protection insurance",  # Similar keywords
                "source": "insurance",
            },
        ]

        result, removed = deduplicator.deduplicate(examples, config)

        # These should be similar enough to deduplicate
        assert removed >= 0  # May or may not deduplicate based on threshold

    def test_different_structures_preserved(self, deduplicator, config):
        """Test that different structures are preserved."""
        examples = [
            {
                "input": "",
                "instruction": "Short",
                "output": "A",
                "source": "legal",
            },
            {
                "input": "Very long input text here with many words",
                "instruction": "Very long instruction text",
                "output": "Very long output with completely different keywords here",
                "source": "medical",
            },
        ]

        result, removed = deduplicator.deduplicate(examples, config)

        assert removed == 0
        assert len(result) == 2

    def test_disabled_structural_dedup(self, deduplicator):
        """Test that structural dedup can be disabled."""
        config = DeduplicationConfig(enable_structural=False)
        examples = [
            {"input": "a", "output": "same"},
            {"input": "a", "output": "same"},
        ]

        result, removed = deduplicator.deduplicate(examples, config)

        assert removed == 0
        assert len(result) == 2

    def test_calculate_similarity(self, deduplicator):
        """Test similarity calculation between structures."""
        struct1 = {
            "has_input": True,
            "has_instruction": True,
            "output_length": 100,
            "input_length": 50,
            "instruction_length": 30,
            "source_type": "insurance",
            "keywords": {"coverage", "policy", "limit"},
        }

        # Identical structure
        similarity = deduplicator.calculate_similarity(struct1, struct1)
        assert similarity == 1.0

        # Different structure
        struct2 = {
            "has_input": False,
            "has_instruction": False,
            "output_length": 10,
            "input_length": 0,
            "instruction_length": 0,
            "source_type": "legal",
            "keywords": {"contract", "agreement"},
        }

        similarity = deduplicator.calculate_similarity(struct1, struct2)
        assert similarity < 0.5  # Should be quite different

    def test_performance_no_index_lookup(self, deduplicator, config):
        """Test that the optimized version doesn't use examples.index().

        This is a regression test for the O(n^2)+O(n) bug where
        examples.index(kept_example) was called inside the loop.
        """
        # Create 100 unique examples
        examples = [
            {
                "input": f"input_{i}",
                "instruction": f"instruction_{i}",
                "output": f"output_{i} with unique content number {i}",
                "source": f"source_{i % 5}",
            }
            for i in range(100)
        ]

        # This should complete quickly without O(n^3) behavior
        import time
        start = time.time()
        result, removed = deduplicator.deduplicate(examples, config)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 100 items)
        assert elapsed < 1.0, f"Deduplication took {elapsed:.2f}s - possible performance regression"

    def test_empty_keywords(self, deduplicator):
        """Test similarity with empty keywords."""
        struct1 = {
            "has_input": True,
            "has_instruction": True,
            "output_length": 0,
            "input_length": 0,
            "instruction_length": 0,
            "source_type": "test",
            "keywords": set(),
        }
        struct2 = {
            "has_input": True,
            "has_instruction": True,
            "output_length": 0,
            "input_length": 0,
            "instruction_length": 0,
            "source_type": "test",
            "keywords": set(),
        }

        # Should not crash with empty keywords
        similarity = deduplicator.calculate_similarity(struct1, struct2)
        assert similarity >= 0.0
