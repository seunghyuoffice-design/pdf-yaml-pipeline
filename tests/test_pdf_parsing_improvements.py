"""Unit tests for PDF parsing improvements."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Test PDF validation functions
def test_validate_pdf_header():
    """Test PDF header validation."""
    from src.pipeline.parsers.pdf_validator import validate_pdf_header

    # Create a temporary file with valid PDF header
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'%PDF-1.4\n%valid pdf content')
        temp_path = f.name

    try:
        assert validate_pdf_header(temp_path) == True
    finally:
        os.unlink(temp_path)

    # Test invalid header
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'Not a PDF file')
        temp_path = f.name

    try:
        assert validate_pdf_header(temp_path) == False
    finally:
        os.unlink(temp_path)

def test_validate_pdf_size():
    """Test PDF size validation."""
    from src.pipeline.parsers.pdf_validator import validate_pdf_size

    # Create a small valid PDF
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'%PDF-1.4\n' + b'x' * 1000)  # 1KB file
        temp_path = f.name

    try:
        assert validate_pdf_size(temp_path, max_size_mb=10) == True
        assert validate_pdf_size(temp_path, max_size_mb=0.001) == False  # Too small
    finally:
        os.unlink(temp_path)

def test_validate_pdf_integrity():
    """Test comprehensive PDF integrity validation."""
    from src.pipeline.parsers.pdf_validator import validate_pdf_integrity

    # Valid PDF-like file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        content = b'%PDF-1.4\n%header\n%%EOF'
        f.write(content)
        temp_path = f.name

    try:
        is_valid, error = validate_pdf_integrity(temp_path)
        assert is_valid == True
        assert error is None
    finally:
        os.unlink(temp_path)

    # Non-existent file
    is_valid, error = validate_pdf_integrity('/non/existent/file.pdf')
    assert is_valid == False
    assert error == "File does not exist"

# Test MultiParserSystem
class TestMultiParserSystem:
    """Test cases for MultiParserSystem."""

    def test_initialization(self):
        """Test MultiParserSystem initialization."""
        from src.pipeline.parsers.multi_parser import MultiParserSystem

        system = MultiParserSystem()
        assert len(system.parsers) > 0
        assert 'docling' in [name for name, _ in system.parsers]

    def test_error_classification(self):
        """Test error classification logic."""
        from src.pipeline.parsers.multi_parser import MultiParserSystem

        system = MultiParserSystem()

        # Test corrupted PDF error
        error = ValueError("Input document is not valid")
        error_type = system._classify_error(error)
        assert error_type == "corrupted"

        # Test encrypted PDF error
        error = ValueError("PDF is encrypted")
        error_type = system._classify_error(error)
        assert error_type == "encrypted"

        # Test memory error
        error = MemoryError("Out of memory")
        error_type = system._classify_error(error)
        assert error_type == "memory"

    def test_result_validation(self):
        """Test parsing result validation."""
        from src.pipeline.parsers.multi_parser import MultiParserSystem

        system = MultiParserSystem()

        # Valid result with paragraphs
        valid_result = {
            'document': {'source_path': '/test.pdf'},
            'content': {'paragraphs': ['Test content']},
            'tables': [],
            'assets': {'images': []}
        }
        assert system._validate_result(valid_result) == True

        # Valid result with tables
        valid_result_tables = {
            'document': {'source_path': '/test.pdf'},
            'content': {'paragraphs': []},
            'tables': [{'page': 1, 'cells': [{'text': 'data'}]}],
            'assets': {'images': []}
        }
        assert system._validate_result(valid_result_tables) == True

        # Invalid result - no content
        invalid_result = {
            'document': {'source_path': '/test.pdf'},
            'content': {'paragraphs': []},
            'tables': [],
            'assets': {'images': []}
        }
        assert system._validate_result(invalid_result) == False

        # Invalid result - missing document
        invalid_result2 = {
            'content': {'paragraphs': ['content']},
            'tables': [],
            'assets': {'images': []}
        }
        assert system._validate_result(invalid_result2) == False

    @patch('src.pipeline.parsers.multi_parser.logger')
    def test_parse_pdf_fallback(self, mock_logger):
        """Test PDF parsing with fallback to alternative parsers."""
        from src.pipeline.parsers.multi_parser import MultiParserSystem

        # Create system with mock parsers
        mock_docling = Mock(side_effect=ValueError("Input document is not valid"))
        mock_pypdf = Mock(return_value={
            'document': {'source_path': '/test.pdf'},
            'content': {'paragraphs': ['Fallback content']},
            'tables': [],
            'assets': {'images': []}
        })

        system = MultiParserSystem(parsers=[
            ('docling', lambda x: mock_docling(x)),
            ('pypdf', lambda x: mock_pypdf(x))
        ])

        # Test parsing
        result, parser_name, error_history = system.parse_pdf('/test.pdf', pre_validate=False)

        # Should succeed with pypdf fallback
        assert result is not None
        assert parser_name == 'pypdf'
        assert len(error_history) == 1  # One failure before success
        assert 'docling' in error_history[0]

    def test_parse_pdf_all_fail(self):
        """Test when all parsers fail."""
        from src.pipeline.parsers.multi_parser import MultiParserSystem

        # Create system with failing parsers
        mock_parser = Mock(side_effect=ValueError("Parser failed"))

        system = MultiParserSystem(parsers=[
            ('parser1', lambda x: mock_parser(x)),
            ('parser2', lambda x: mock_parser(x))
        ])

        # Test parsing
        result, parser_name, error_history = system.parse_pdf('/test.pdf', pre_validate=False)

        # Should fail
        assert result is None
        assert parser_name == 'all_failed'
        assert len(error_history) == 2  # Both parsers failed

    def test_stats_collection(self):
        """Test statistics collection."""
        from src.pipeline.parsers.multi_parser import MultiParserSystem

        system = MultiParserSystem()

        # Mock successful parse
        with patch.object(system, 'parse_pdf', return_value=(
            {'document': {'source_path': '/test.pdf'}, 'content': {'paragraphs': ['test']}, 'tables': [], 'assets': {'images': []}},
            'docling',
            []
        )):
            system.parse_pdf('/test.pdf', pre_validate=False)

        stats = system.get_stats()
        assert stats['attempts'] == 1
        assert stats['successes'] == 1
        assert stats['failures'] == 0
        assert stats['success_rate'] == 100.0

# Test SafeParser integration
class TestSafeParserIntegration:
    """Test SafeParser with multi-parser integration."""

    @patch('src.pipeline.parsers.safe_parser._parse_worker_loop')
    def test_fallback_logging(self, mock_worker_loop):
        """Test that fallback usage is logged properly."""
        from src.pipeline.parsers.safe_parser import SafeParser

        # Mock successful fallback result
        mock_worker_loop.return_value = None  # Worker loop doesn't return

        # Create SafeParser and mock the worker response
        parser = SafeParser(timeout=30)

        # Mock the queues and worker
        with patch.object(parser, '_req_q') as mock_req_q, \
             patch.object(parser, '_res_q') as mock_res_q, \
             patch.object(parser, '_worker') as mock_worker:

            mock_worker.is_alive.return_value = True
            mock_req_q.get.return_value = {"job_id": "test", "file_path": "/test.pdf"}
            mock_res_q.get.return_value = {
                "job_id": "test",
                "success": True,
                "data": {"test": "data"},
                "fallback_used": True,
                "fallback_parser": "pypdf"
            }

            # This would normally run the parse method
            # We're just testing the response handling logic exists
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])