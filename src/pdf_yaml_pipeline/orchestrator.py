"""Pipeline orchestrator for document conversion.

PDF/HWP/HWPX → YAML → JSONL 파이프라인 오케스트레이션.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from src.pipeline.converters.factory import OutputFormat
from src.pipeline.converters.jsonl_converter import JSONLConverter, MultiFormatConverter
from src.pipeline.converters.yaml_converter import YAMLConverter
from src.pipeline.parsers.unified_parser import UnifiedParser
from src.schemas.document import DocumentMetadata, DocumentType, TrainingExample


@dataclass
class PipelineConfig:
    """파이프라인 설정.

    Args:
        input_dir: 입력 디렉터리
        output_dir: 출력 디렉터리
        output_format: 출력 형식
        output_formats: 다중 출력 형식 (지정 시 output_format 무시)
        chunk_size: 청크 크기 (문자 수)
        create_splits: Train/Val/Test 분할 생성 여부
        skip_errors: 오류 발생 시 건너뛰기
    """

    input_dir: Path
    output_dir: Path
    output_format: OutputFormat = OutputFormat.OPENAI
    output_formats: list[OutputFormat] | None = None
    chunk_size: int = 1000
    create_splits: bool = True
    skip_errors: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class PipelineStats:
    """파이프라인 실행 통계.

    Args:
        files_processed: 처리된 파일 수
        files_failed: 실패한 파일 수
        total_examples: 생성된 총 예제 수
        examples_by_type: 문서 유형별 예제 수
        errors: 오류 목록
        duration_seconds: 총 소요 시간
    """

    files_processed: int = 0
    files_failed: int = 0
    total_examples: int = 0
    examples_by_type: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


class Pipeline:
    """문서 변환 파이프라인.

    PDF/DOCX 파일을 파싱하고 JSONL 형식으로 변환.

    Args:
        config: 파이프라인 설정

    Example:
        >>> pipeline = create_pipeline(
        ...     input_dir="./pdfs",
        ...     output_dir="./output",
        ...     output_format="openai",
        ... )
        >>> stats = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # 컨버터 초기화
        if config.output_format == OutputFormat.YAML:
            self.converter = None
            self.yaml_converter = YAMLConverter(
                output_format=config.output_format,
                include_metadata=True,
                validate_integrity=True,
            )
            self.multi_converter = None
        elif config.output_formats:
            self.multi_converter = MultiFormatConverter(
                formats=config.output_formats,
                include_metadata=True,
            )
            self.converter = None
            self.yaml_converter = None
        else:
            self.converter = JSONLConverter(
                output_format=config.output_format,
                include_metadata=True,
            )
            self.yaml_converter = None
            self.multi_converter = None

    def run(self) -> PipelineStats:
        """파이프라인 실행.

        Returns:
            PipelineStats: 실행 통계
        """
        start_time = time.time()
        stats = PipelineStats()

        # 파일 목록 수집
        files = self._collect_files()
        logger.info(f"Found {len(files)} files to process")

        all_examples: list[TrainingExample] = []

        for file_path in files:
            try:
                examples = self.process_file(file_path)
                all_examples.extend(examples)
                stats.files_processed += 1

                # 문서 유형별 카운트
                for ex in examples:
                    doc_type = ex.document_type.value if hasattr(ex.document_type, "value") else str(ex.document_type)
                    stats.examples_by_type[doc_type] = stats.examples_by_type.get(doc_type, 0) + 1

                logger.info(f"Processed: {file_path.name} -> {len(examples)} examples")

            except Exception as e:
                stats.files_failed += 1
                error_info = {"file": str(file_path), "error": str(e)}
                stats.errors.append(error_info)
                logger.error(f"Failed: {file_path.name} - {e}")

                if not self.config.skip_errors:
                    raise

        stats.total_examples = len(all_examples)

        # 내보내기
        if all_examples:
            self._export(all_examples)

            # 분할 생성
            if self.config.create_splits:
                self._create_splits(all_examples)

        stats.duration_seconds = time.time() - start_time
        return stats

    def process_file(self, file_path: Path) -> list[TrainingExample]:
        """단일 파일 처리.

        Args:
            file_path: 처리할 파일 경로

        Returns:
            list[TrainingExample]: 생성된 예제 목록
        """
        # UnifiedParser로 직접 파싱 (YAML dict 반환)
        parser = UnifiedParser()
        parsed_doc = parser.parse(file_path)

        # YAML 파이프라인의 경우 바로 변환
        if self.yaml_converter:
            examples_data = self.yaml_converter.convert_document(parsed_doc)
            return self._data_to_training_examples(examples_data, file_path)

        # 기존 방식으로 계속 (JSONL)
        # 메타데이터 생성
        metadata = DocumentMetadata(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            file_size_bytes=file_path.stat().st_size,
            document_type=self._classify_document(file_path, parsed_doc),
        )

        # 청킹 및 예제 생성
        examples = self._create_examples(parsed_doc, metadata)

        return examples

    def _data_to_training_examples(self, examples_data: list[dict], file_path: Path) -> list[TrainingExample]:
        """YAML 변환 데이터를 TrainingExample로 변환."""
        examples = []

        if not examples_data or not isinstance(examples_data, list):
            return examples

        for i, data in enumerate(examples_data):
            if not isinstance(data, dict):
                continue
            example = TrainingExample(
                id=f"{file_path.name}_{i:04d}",
                source_document_id=file_path.name,
                document_type=DocumentType.UNKNOWN,  # YAML에서 이미 source 정보 있음
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", ""),
                example_type="qa",
                metadata=data.get("metadata", {}),
            )
            examples.append(example)

        return examples

    def _collect_files(self) -> list[Path]:
        """처리할 파일 목록 수집."""
        files = []
        for ext in UnifiedParser.SUPPORTED_EXTENSIONS:
            ext_clean = ext.lstrip(".")
            files.extend(self.config.input_dir.glob(f"*.{ext_clean}"))
            files.extend(self.config.input_dir.glob(f"**/*.{ext_clean}"))
        return sorted(set(files))

    def _classify_document(self, file_path: Path, parsed_doc: Any) -> DocumentType:
        """문서 유형 분류."""
        name_lower = file_path.name.lower()

        if "약관" in name_lower or "terms" in name_lower:
            return DocumentType.TERMS_AND_CONDITIONS
        elif "사업방법서" in name_lower:
            return DocumentType.BUSINESS_METHOD
        elif "요약서" in name_lower or "summary" in name_lower:
            return DocumentType.PRODUCT_SUMMARY
        elif "보도자료" in name_lower or "press" in name_lower:
            return DocumentType.FSS_PRESS_RELEASE
        elif "판례" in name_lower or "판결" in name_lower:
            return DocumentType.COURT_PRECEDENT
        else:
            return DocumentType.UNKNOWN

    def _create_examples(
        self,
        parsed_doc: Any,
        metadata: DocumentMetadata,
    ) -> list[TrainingExample]:
        """파싱된 문서에서 학습 예제 생성."""
        examples = []

        # YAML dict에서 텍스트 추출 (UnifiedParser 출력)
        if isinstance(parsed_doc, dict):
            content = parsed_doc.get("content", {})
            paragraphs = content.get("paragraphs", [])
            text = "\n\n".join(paragraphs) if paragraphs else ""
        # 레거시 호환 (ParsedDocument)
        elif hasattr(parsed_doc, "markdown"):
            text = parsed_doc.markdown
        elif hasattr(parsed_doc, "raw_text"):
            text = parsed_doc.raw_text
        elif hasattr(parsed_doc, "text"):
            text = parsed_doc.text
        else:
            text = str(parsed_doc)

        # 청킹
        chunks = self._chunk_text(text, self.config.chunk_size)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            example = TrainingExample(
                id=f"{metadata.file_name}_{i:04d}",
                source_document_id=metadata.file_name,
                document_type=metadata.document_type,
                instruction="다음 보험 문서를 분석하고 핵심 내용을 설명하세요.",
                input=chunk,
                output="",  # Teacher가 채울 부분
                example_type="extraction",
                metadata={
                    "chunk_index": i,
                    "source_path": metadata.source_path,
                },
            )
            examples.append(example)

        return examples

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """텍스트를 청크로 분할."""
        if not text:
            return []

        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _export(self, examples: list[TrainingExample]) -> None:
        """예제 내보내기."""
        if self.yaml_converter:
            # TrainingExample을 다시 ParsedDocument로 변환
            # (임시 구현 - 추후 개선 필요)
            output_path = self.config.output_dir / "data.yaml"
            with open(output_path, "w", encoding="utf-8") as f:
                # 간단한 형식으로 저장
                data = []
                for ex in examples:
                    item = {
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": ex.output,
                        "metadata": ex.metadata,
                    }
                    data.append(item)

                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        elif self.multi_converter:
            self.multi_converter.export_all(
                examples,
                self.config.output_dir,
                base_name="data",
            )
        elif self.converter:
            output_path = self.config.output_dir / f"data_{self.config.output_format.value}.jsonl"
            self.converter.export(examples, output_path)

    def _create_splits(self, examples: list[TrainingExample]) -> None:
        """Train/Val/Test 분할 생성."""
        import random

        random.seed(42)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        converter = JSONLConverter(
            output_format=self.config.output_format,
            include_metadata=True,
        )

        for split_name, split_examples in splits.items():
            if split_examples:
                output_path = self.config.output_dir / f"{split_name}.jsonl"
                converter.export(split_examples, output_path)
                logger.info(f"Created {split_name}.jsonl with {len(split_examples)} examples")


def create_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    output_format: str = "openai",
    output_formats: list[str] | None = None,
    chunk_size: int = 1000,
    create_splits: bool = True,
    skip_errors: bool = True,
) -> Pipeline:
    """파이프라인 생성 헬퍼.

    Args:
        input_dir: 입력 디렉터리
        output_dir: 출력 디렉터리
        output_format: 출력 형식
        output_formats: 다중 출력 형식
        chunk_size: 청크 크기
        create_splits: 분할 생성 여부
        skip_errors: 오류 시 건너뛰기

    Returns:
        Pipeline: 설정된 파이프라인
    """
    config = PipelineConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        output_format=OutputFormat(output_format),
        output_formats=[OutputFormat(f) for f in output_formats] if output_formats else None,
        chunk_size=chunk_size,
        create_splits=create_splits,
        skip_errors=skip_errors,
    )
    return Pipeline(config)


__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineStats",
    "create_pipeline",
]
