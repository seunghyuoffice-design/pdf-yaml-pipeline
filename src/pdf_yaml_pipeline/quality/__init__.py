"""Quality module - 품질 검증 및 자동 수정."""

from src.pipeline.quality.auto_validator import (
    calculate_quality_scores,
    print_validation_report,
    validate_jsonl_file,
)
from src.pipeline.quality.autofix import AutoFixer
from src.pipeline.quality.ocr_remap import remap_ocr_lines_to_table_cells_iou
from src.pipeline.quality.table_quality import (
    attach_cell_reliability,
    attach_table_quality,
)

# Re-export for backward compatibility
QualityValidator = AutoFixer  # Alias


class ReportGenerator:
    """품질 리포트 생성기.

    다양한 형식(JSON, HTML, Markdown)으로 리포트 생성.
    """

    def generate_all(
        self,
        validation_report: dict,
        output_dir,
        formats: list[str] | None = None,
        base_name: str = "quality_report",
    ) -> dict[str, str]:
        """모든 형식으로 리포트 생성.

        Args:
            validation_report: 검증 결과
            output_dir: 출력 디렉터리
            formats: 생성할 형식 목록
            base_name: 기본 파일명

        Returns:
            dict: 형식별 파일 경로
        """
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = formats or ["json", "html"]
        paths = {}

        if "json" in formats:
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(validation_report, f, ensure_ascii=False, indent=2, default=str)
            paths["json"] = str(json_path)

        if "html" in formats:
            html_path = output_dir / f"{base_name}.html"
            html_content = self._generate_html(validation_report)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            paths["html"] = str(html_path)

        if "markdown" in formats:
            md_path = output_dir / f"{base_name}.md"
            md_content = self._generate_markdown(validation_report)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            paths["markdown"] = str(md_path)

        return paths

    def _generate_html(self, report: dict) -> str:
        """HTML 리포트 생성."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Quality Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
    </style>
</head>
<body>
    <h1>Quality Validation Report</h1>
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Records</td><td>{report.get('total_records', 'N/A')}</td></tr>
        <tr><td>Valid Records</td><td>{report.get('valid_records', 'N/A')}</td></tr>
        <tr><td>Filtered Records</td><td>{report.get('filtered_records', 'N/A')}</td></tr>
        <tr><td>Invalid Records</td><td>{report.get('invalid_records', 'N/A')}</td></tr>
    </table>
</body>
</html>"""

    def _generate_markdown(self, report: dict) -> str:
        """Markdown 리포트 생성."""
        return f"""# Quality Validation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Records | {report.get('total_records', 'N/A')} |
| Valid Records | {report.get('valid_records', 'N/A')} |
| Filtered Records | {report.get('filtered_records', 'N/A')} |
| Invalid Records | {report.get('invalid_records', 'N/A')} |

## Scores

- Structure: {report.get('avg_structure_score', 0):.2%}
- Completeness: {report.get('avg_completeness_score', 0):.2%}
- Quality: {report.get('avg_quality_score', 0):.2%}
- Overall: {report.get('avg_overall_score', 0):.2%}
"""


__all__ = [
    "AutoFixer",
    "QualityValidator",
    "ReportGenerator",
    "validate_jsonl_file",
    "calculate_quality_scores",
    "print_validation_report",
    # Table quality (YAML pipeline)
    "attach_cell_reliability",
    "attach_table_quality",
    "remap_ocr_lines_to_table_cells_iou",
]
