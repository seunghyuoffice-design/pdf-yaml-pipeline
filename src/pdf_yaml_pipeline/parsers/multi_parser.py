"""Multi-parser fallback system for robust PDF processing."""

import logging
from typing import Any, Dict, Optional, Tuple, Callable, List
from pathlib import Path
import traceback

try:
    from .pdf_validator import validate_pdf_integrity, is_pdf_corrupted
except ImportError:
    # Fallback if validator not available
    def validate_pdf_integrity(file_path: str) -> Tuple[bool, Optional[str]]:
        return True, None

    def is_pdf_corrupted(file_path: str) -> Tuple[bool, Optional[str]]:
        return False, None

logger = logging.getLogger(__name__)

class ParserError(Exception):
    """Custom exception for parser failures."""
    def __init__(self, parser_name: str, original_error: Exception, error_type: str = "unknown"):
        self.parser_name = parser_name
        self.original_error = original_error
        self.error_type = error_type
        super().__init__(f"{parser_name} failed: {str(original_error)}")

class MultiParserSystem:
    """Robust PDF parsing with multiple fallback parsers."""

    def __init__(self, parsers: Optional[List[Tuple[str, Callable]]] = None):
        self.parsers = parsers or self._get_default_parsers()
        self.stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'parser_usage': {}
        }

    def _get_default_parsers(self) -> List[Tuple[str, Callable]]:
        """Get default parser chain with fallbacks."""
        parsers = []

        # Primary: docling (if available)
        try:
            from .docling_yaml_adapter import DoclingYAMLAdapter
            parsers.append(('docling', self._docling_parser))
        except ImportError:
            logger.warning("docling not available, skipping")

        # Fallback 1: pypdf
        try:
            import pypdf
            parsers.append(('pypdf', self._pypdf_parser))
        except ImportError:
            logger.warning("pypdf not available, skipping")

        # Fallback 2: pdfplumber
        try:
            import pdfplumber
            parsers.append(('pdfplumber', self._pdfplumber_parser))
        except ImportError:
            logger.warning("pdfplumber not available, skipping")

        # Note: PyMuPDF (fitz) removed due to AGPL license (Dyarchy policy: MIT/Apache-2.0/BSD only)
        # PDF normalization/repair is handled by pikepdf in pre-processing stage, not here.

        return parsers

    def _docling_parser(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF using docling."""
        from .docling_yaml_adapter import DoclingYAMLAdapter
        adapter = DoclingYAMLAdapter()
        return adapter.parse(file_path)

    def _pypdf_parser(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF using pypdf."""
        import pypdf

        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)

            # Extract text from all pages
            text_content = []
            for page in pdf.pages:
                text = page.extract_text()
                if text.strip():
                    text_content.append(text.strip())

            return {
                'document': {
                    'source_path': file_path,
                    'format': 'pdf',
                    'parser': 'pypdf',
                    'page_count': len(pdf.pages),
                    'original_pages': len(pdf.pages),
                    'truncated': False,
                    'ocr_enabled': False,
                    'table_extraction': False,
                    'encrypted': pdf.is_encrypted
                },
                'content': {
                    'paragraphs': text_content
                },
                'tables': [],
                'assets': {'images': []}
            }

    def _pdfplumber_parser(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF using pdfplumber."""
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            text_content = []
            tables = []

            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(text.strip())

                # Extract tables
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    if table:
                        # Convert table to structured format
                        table_data = {
                            'page': page_num + 1,
                            'bbox': None,  # pdfplumber doesn't provide bbox easily
                            'cells': []
                        }

                        for row_idx, row in enumerate(table):
                            for col_idx, cell in enumerate(row):
                                if cell and str(cell).strip():
                                    table_data['cells'].append({
                                        'text': str(cell).strip(),
                                        'row': row_idx,
                                        'col': col_idx,
                                        'bbox': None,
                                        'confidence': 1.0  # pdfplumber is text-based
                                    })

                        if table_data['cells']:
                            tables.append(table_data)

            return {
                'document': {
                    'source_path': file_path,
                    'format': 'pdf',
                    'parser': 'pdfplumber',
                    'page_count': len(pdf.pages),
                    'original_pages': len(pdf.pages),
                    'truncated': False,
                    'ocr_enabled': False,
                    'table_extraction': True,
                    'encrypted': False  # pdfplumber handles encryption differently
                },
                'content': {
                    'paragraphs': text_content
                },
                'tables': tables,
                'assets': {'images': []}
            }

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for better handling."""
        error_str = str(error).lower()

        if any(keyword in error_str for keyword in ['not valid', 'invalid', 'corrupt', 'damaged']):
            return 'corrupted'
        elif any(keyword in error_str for keyword in ['encrypted', 'password', 'permission']):
            return 'encrypted'
        elif any(keyword in error_str for keyword in ['version', 'format', 'unsupported']):
            return 'version_unsupported'
        elif any(keyword in error_str for keyword in ['memory', 'out of memory', 'oom']):
            return 'memory'
        elif any(keyword in error_str for keyword in ['timeout', 'time']):
            return 'timeout'
        else:
            return 'unknown'

    def parse_pdf(self, file_path: str, pre_validate: bool = True) -> Tuple[Optional[Dict[str, Any]], str, List[str]]:
        """
        Parse PDF with multi-parser fallback.

        Args:
            file_path: Path to PDF file
            pre_validate: Whether to validate PDF before parsing

        Returns:
            Tuple[result, successful_parser, error_history]
        """
        self.stats['attempts'] += 1

        # Pre-validation
        if pre_validate:
            is_valid, error_msg = validate_pdf_integrity(file_path)
            if not is_valid:
                self.stats['failures'] += 1
                return None, "pre_validation_failed", [error_msg] if error_msg else ["Validation failed"]

        error_history = []

        # Try each parser in order
        for parser_name, parser_func in self.parsers:
            try:
                logger.debug(f"Trying parser: {parser_name} for {file_path}")
                result = parser_func(file_path)

                # Basic validation of result
                if self._validate_result(result):
                    self.stats['successes'] += 1
                    self.stats['parser_usage'][parser_name] = self.stats['parser_usage'].get(parser_name, 0) + 1
                    return result, parser_name, error_history
                else:
                    error_msg = f"{parser_name} produced invalid result"
                    error_history.append(error_msg)
                    logger.warning(error_msg)

            except Exception as e:
                error_type = self._classify_error(e)
                error_msg = f"{parser_name} failed ({error_type}): {str(e)}"
                error_history.append(error_msg)
                logger.warning(error_msg)

                # Log full traceback for debugging
                logger.debug(f"Parser {parser_name} traceback: {traceback.format_exc()}")

        # All parsers failed
        self.stats['failures'] += 1
        return None, "all_failed", error_history

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate parsing result structure."""
        try:
            # Check basic structure
            if not isinstance(result, dict):
                return False

            # Check document info
            doc = result.get('document', {})
            if not isinstance(doc, dict):
                return False

            if not doc.get('source_path'):
                return False

            # Check content
            content = result.get('content', {})
            if not isinstance(content, dict):
                return False

            # At least one of paragraphs or tables should have content
            paragraphs = content.get('paragraphs', [])
            tables = result.get('tables', [])

            has_content = (
                (isinstance(paragraphs, list) and len(paragraphs) > 0) or
                (isinstance(tables, list) and len(tables) > 0)
            )

            return has_content

        except Exception as e:
            logger.error(f"Result validation error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        total_attempts = self.stats['attempts']
        success_rate = (self.stats['successes'] / total_attempts * 100) if total_attempts > 0 else 0

        # Calculate failure rate and parser effectiveness
        failure_rate = (self.stats['failures'] / total_attempts * 100) if total_attempts > 0 else 0

        # Parser usage percentages
        parser_stats = {}
        for parser_name, usage_count in self.stats['parser_usage'].items():
            parser_stats[parser_name] = {
                'usage_count': usage_count,
                'usage_percentage': round(usage_count / total_attempts * 100, 2) if total_attempts > 0 else 0
            }

        return {
            **self.stats,
            'success_rate': round(success_rate, 2),
            'failure_rate': round(failure_rate, 2),
            'parser_effectiveness': parser_stats,
            'total_attempts': total_attempts,
            'parsers_available': len(self.parsers)
        }

    def log_summary(self) -> None:
        """Log a summary of parsing statistics."""
        stats = self.get_stats()

        logger.info("Multi-Parser System Summary:")
        logger.info(f"  Total attempts: {stats['total_attempts']}")
        logger.info(f"  Success rate: {stats['success_rate']}%")
        logger.info(f"  Failure rate: {stats['failure_rate']}%")
        logger.info(f"  Parsers available: {stats['parsers_available']}")

        if stats['parser_effectiveness']:
            logger.info("  Parser effectiveness:")
            for parser_name, parser_data in stats['parser_effectiveness'].items():
                logger.info(f"    {parser_name}: {parser_data['usage_count']} uses ({parser_data['usage_percentage']}%)")