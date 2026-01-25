"""YAML conversion utilities for training data.

다양한 형식으로 YAML 변환, 병합, 분할 기능 제공.
OpenAI, Qwen3, Alpaca, ShareGPT 형식을 지원하며 Integrity validation 포함.

NOTE: 이 모듈은 레거시 ParsedDocument와 새 YAML dict 형식 모두 지원합니다.
"""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from pdf_yaml_pipeline.converters.factory import OutputFormat
from pdf_yaml_pipeline.converters.format_utils import (
    DEFAULT_QWEN3_THINKING,
    to_alpaca,
    to_openai,
    to_qwen3,
    to_sharegpt,
)
from pdf_yaml_pipeline.utils.doc_classifier import classify_document_label
from pdf_yaml_pipeline.utils.splitter import split_items
from pdf_yaml_pipeline.parsers.base import ParsedDocument


class YAMLConverter:
    """YAML 형식 변환기.

    ParsedDocument을 다양한 LLM 학습 형식으로 변환.

    Args:
        output_format: 출력 형식
        include_metadata: 메타데이터 포함 여부
        validate_integrity: 무결성 검증 여부

    Example:
        >>> converter = YAMLConverter(output_format=OutputFormat.OPENAI)
        >>> converter.export(examples, "output.yaml")
    """

    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.OPENAI,
        include_metadata: bool = True,
        validate_integrity: bool = True,
    ):
        self.output_format = output_format
        self.include_metadata = include_metadata
        self.validate_integrity = validate_integrity

    def convert_document(self, doc: Union[ParsedDocument, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ParsedDocument 또는 YAML dict를 학습 데이터로 변환.

        Args:
            doc: 파싱된 문서 (ParsedDocument 또는 UnifiedParser의 YAML dict)

        Returns:
            List[Dict]: 변환된 학습 데이터 목록
        """
        # YAML dict 형식 처리 (UnifiedParser 출력)
        if isinstance(doc, dict):
            return self._convert_yaml_dict(doc)

        # 레거시 ParsedDocument 처리
        warnings.warn(
            "ParsedDocument is deprecated. Use UnifiedParser which returns YAML dict.",
            DeprecationWarning,
            stacklevel=2,
        )

        examples = []

        # 문서 타입에 따른 분류
        doc_type = self._classify_document(doc)

        # 구조화된 데이터 생성
        if doc.structure.headings:
            # 헤딩 기반 예제 생성
            examples.extend(self._create_heading_examples(doc, doc_type))

        # 표 데이터 예제 생성
        if doc.tables:
            examples.extend(self._create_table_examples(doc, doc_type))

        # 전체 문서 예제 생성
        text = getattr(doc, "markdown", "") or ""
        if len(text) > 500:  # 의미 있는 내용만
            examples.extend(self._create_document_examples(doc, doc_type))

        # 각 예제를 지정된 형식으로 변환
        return [self._convert_example(example) for example in examples]

    def convert(self, doc: Union[ParsedDocument, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """convert_document 호환 래퍼."""
        return self.convert_document(doc)

    def _convert_yaml_dict(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """YAML dict (UnifiedParser 출력)를 학습 데이터로 변환.

        Args:
            doc: UnifiedParser의 YAML dict 출력
                {
                    "document": {...},
                    "content": {"paragraphs": [...]},
                    "tables": [...],
                    "assets": {...}
                }

        Returns:
            List[Dict]: 변환된 학습 데이터 목록
        """
        examples = []

        source_path = doc.get("document", {}).get("source_path", "unknown")
        paragraphs = doc.get("content", {}).get("paragraphs", [])
        tables = doc.get("tables", [])

        # 문서 타입 분류
        doc_type = self._classify_yaml_doc(doc)

        # 문단 기반 예제 생성
        full_text = "\n\n".join(paragraphs)
        if len(full_text) > 500:
            max_summary_chars = 8000
            truncated = len(full_text) > max_summary_chars
            summary_text = full_text[:max_summary_chars]
            examples.append(
                {
                    "instruction": self._get_system_prompt(doc_type),
                    "input": f"이 문서의 주요 내용을 요약해주세요.\n\n{summary_text}",
                    "output": "",  # 요약은 Teacher가 채울 예정
                    "source": doc_type,
                    "metadata": {
                        "file": source_path,
                        "type": "document_summary",
                        "truncated": truncated,
                        "original_length": len(full_text),
                    },
                }
            )

        # 테이블 기반 예제 생성
        for i, table in enumerate(tables):
            cells = table.get("cells", [])
            if not cells:
                continue

            # 테이블 텍스트 추출
            table_text = self._extract_table_text(table)
            if table_text:
                examples.append(
                    {
                        "instruction": self._get_system_prompt(doc_type),
                        "input": f"다음 표의 내용을 분석해주세요:\n\n{table_text}",
                        "output": "",  # Teacher가 채울 예정
                        "source": doc_type,
                        "metadata": {
                            "file": source_path,
                            "table_index": i,
                            "page": table.get("page", 1),
                        },
                    }
                )

        # 각 예제를 지정된 형식으로 변환
        return [self._convert_example(example) for example in examples]

    def _classify_yaml_doc(self, doc: Dict[str, Any]) -> str:
        """YAML dict에서 문서 타입 분류."""
        source_path = doc.get("document", {}).get("source_path", "").lower()
        paragraphs = doc.get("content", {}).get("paragraphs", [])
        content = " ".join(paragraphs)
        return classify_document_label(source_path, content)

    def _extract_table_text(self, table: Dict[str, Any]) -> str:
        """테이블에서 텍스트 추출."""
        cells = table.get("cells", [])
        if not cells:
            return ""

        shape = table.get("shape", {})
        n_rows = shape.get("rows", 0)
        n_cols = shape.get("cols", 0)

        if n_rows == 0 or n_cols == 0:
            max_row = max((cell.get("row", 0) for cell in cells), default=0)
            max_col = max((cell.get("col", 0) for cell in cells), default=0)
            n_rows = max_row + 1
            n_cols = max_col + 1

        if n_rows == 0 or n_cols == 0:
            return ""

        # 그리드 구성
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for cell in cells:
            r = cell.get("row", 0)
            c = cell.get("col", 0)
            if 0 <= r < n_rows and 0 <= c < n_cols:
                grid[r][c] = cell.get("text", "")

        # 마크다운 테이블로 변환
        lines = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines.append("|" + "---|" * n_cols)

        return "\n".join(lines)

    def _classify_document(self, doc: ParsedDocument) -> str:
        """문서 타입 분류."""
        content = doc.markdown
        return classify_document_label(doc.source_path, content)

    def _create_heading_examples(self, doc: ParsedDocument, doc_type: str) -> List[Dict[str, Any]]:
        """헤딩 기반 예제 생성."""
        examples = []

        for heading in doc.structure.headings:
            # 헤딩 관련 질문 생성
            questions = self._generate_heading_questions(heading, doc_type)

            for question in questions:
                examples.append(
                    {
                        "instruction": self._get_system_prompt(doc_type),
                        "input": question,
                        "output": self._generate_heading_answer(heading, doc),
                        "source": doc_type,
                        "metadata": {
                            "file": doc.source_path,
                            "heading": heading,
                            "page": heading.get("page", 1),
                        },
                    }
                )

        return examples

    def _create_table_examples(self, doc: ParsedDocument, doc_type: str) -> List[Dict[str, Any]]:
        """표 데이터 기반 예제 생성."""
        examples = []

        for i, table in enumerate(doc.tables):
            if not table.headers:
                continue

            # 표 관련 질문 생성
            questions = self._generate_table_questions(table, doc_type)

            for question in questions:
                answer = self._generate_table_answer(table, question)
                examples.append(
                    {
                        "instruction": self._get_system_prompt(doc_type),
                        "input": question,
                        "output": answer,
                        "source": doc_type,
                        "metadata": {
                            "file": doc.source_path,
                            "table_index": i,
                            "page": table.page_number,
                        },
                    }
                )

        return examples

    def _create_document_examples(self, doc: ParsedDocument, doc_type: str) -> List[Dict[str, Any]]:
        """전체 문서 기반 예제 생성."""
        examples = []

        # 문서 전체에 대한 일반적인 질문
        general_questions = self._generate_document_questions(doc_type)

        for question in general_questions:
            answer = self._generate_document_answer(doc, question)
            examples.append(
                {
                    "instruction": self._get_system_prompt(doc_type),
                    "input": question,
                    "output": answer,
                    "source": doc_type,
                    "metadata": {
                        "file": doc.source_path,
                        "page_count": doc.page_count,
                        "text_length": len(doc.markdown),
                    },
                }
            )

        return examples

    def _generate_heading_questions(self, heading: Dict, doc_type: str) -> List[str]:
        """헤딩에 대한 질문 생성."""
        questions = []
        text = heading.get("text", "")

        # 기본 질문 패턴
        base_questions = [
            f"'{text}'에 대해 설명해 주세요.",
            f"'{text}'의 주요 내용은 무엇인가요?",
            f"'{text}' 관련하여 알아야 할 사항은 무엇인가요?",
        ]

        # 문서 타입별 특화 질문
        if doc_type == "terms_and_conditions":
            base_questions.extend(
                [
                    f"'{text}' 조항의 적용 조건은 무엇인가요?",
                    f"'{text}'에서 규정하는 권리와 의무는 무엇인가요?",
                ]
            )
        elif doc_type == "business_method":
            base_questions.extend(
                [
                    f"'{text}'의 계산 방법은 무엇인가요?",
                    f"'{text}'에 적용되는 요율은 어떻게 되나요?",
                ]
            )

        return questions[:3]  # 최대 3개 질문

    def _generate_table_questions(self, table, doc_type: str) -> List[str]:
        """표에 대한 질문 생성."""
        questions = []

        if doc_type == "terms_and_conditions":
            questions.extend(
                [
                    "이 표에 나타난 보험료율을 설명해 주세요.",
                    "표의 기준금액과 보험료 관계를 설명해 주세요.",
                ]
            )
        elif doc_type == "business_method":
            questions.extend(
                [
                    "이 표의 계산식을 설명해 주세요.",
                    "표에 사용된 변수와 계산 방법을 설명해 주세요.",
                ]
            )
        else:
            questions.extend(
                [
                    "이 표의 내용을 설명해 주세요.",
                    "표의 주요 특징은 무엇인가요?",
                ]
            )

        return questions[:2]

    def _generate_document_questions(self, doc_type: str) -> List[str]:
        """문서 전체에 대한 질문."""
        questions = []

        if doc_type == "terms_and_conditions":
            questions.extend(
                [
                    "이 약관의 목적과 적용 범위는 무엇인가요?",
                    "가입자가 알아야 할 주요 권리와 의무는 무엇인가요?",
                    "보험금 청구 절차와 필요 서류는 무엇인가요?",
                ]
            )
        elif doc_type == "business_method":
            questions.extend(
                [
                    "보험료 산출 방법의 기본 원리는 무엇인가요?",
                    "순보험료와 영업비용의 관계는 어떻게 되나요?",
                    "책임준비금 산정 기준은 무엇인가요?",
                ]
            )
        else:
            questions.extend(
                [
                    "이 문서의 주요 내용은 무엇인가요?",
                    "문서의 핵심 정보를 요약해 주세요.",
                ]
            )

        return questions

    def _generate_heading_answer(self, heading: Dict, doc: ParsedDocument) -> str:
        """헤딩에 대한 답변 생성."""
        text = heading.get("text", "")
        level = heading.get("level", 1)

        # 관련 내용 찾기

        # 헤딩 다음 내용 추출 (단순화된 방식)
        answer = f"'{text}'에 대한 설명입니다.\n\n"

        # 문서 타입별 답변 패턴
        if "약관" in doc.source_path.lower():
            answer += f"이 조항은 보험계약의 {'일반적인' if level == 1 else '구체적인'} 내용을 규정합니다. "
            answer += "계약 당사자의 권리와 의무, 보험료 납부, 보험금 청구 등의 중요한 사항이 포함됩니다."
        elif "사업방법서" in doc.source_path.lower():
            answer += f"이 부분은 {'기본적인' if level == 1 else '상세한'} 계산 방법을 설명합니다. "
            answer += "순보험료, 부가보험료, 영업비용 등의 산정 기준과 적용 요율이 명시됩니다."
        else:
            answer += f"이 {'섹션' if level == 1 else '항목'}은 문서의 주요 내용 중 하나입니다. "
            answer += "관련된 상세 정보와 적용 사항이 포함되어 있습니다."

        return answer

    def _generate_table_answer(self, table, question: str) -> str:
        """표에 대한 답변 생성."""
        if not table.headers:
            return "표 데이터가 충분하지 않습니다."

        answer = "표의 내용을 설명해 드리겠습니다.\n\n"

        # 표 구조 설명
        answer += "**표 구조**:\n"
        answer += f"- 헤더: {', '.join(table.headers[:3])}{'...' if len(table.headers) > 3 else ''}\n"
        answer += f"- 데이터 행: {len(table.rows)}개\n\n"

        # 첫 2개 행 예시
        if table.rows:
            answer += "**주요 내용**:\n"
            for i, (_, row_data) in enumerate(table.rows[:2]):
                answer += f"{i+1}. {', '.join(row_data[:2])}{'...' if len(row_data) > 2 else ''}\n"

        answer += "\n이 표는 관련 기준과 계산 방법을 체계적으로 정리한 것입니다."

        return answer

    def _generate_document_answer(self, doc: ParsedDocument, question: str) -> str:
        """문서에 대한 답변 생성."""
        answer = "문서의 주요 내용을 요약해 드리겠습니다.\n\n"

        # 기본 정보
        answer += "**문서 정보**:\n"
        answer += f"- 파일명: {Path(doc.source_path).name}\n"
        answer += f"- 페이지: {doc.page_count}쪽\n"
        answer += f"- 표 수: {len(doc.tables)}개\n"
        answer += f"- 주요 섹션: {len(doc.structure.headings)}개\n\n"

        # 주요 헤딩
        if doc.structure.headings:
            answer += "**주요 내용**:\n"
            for heading in doc.structure.headings[:5]:
                level_prefix = "  " * (heading.get("level", 1) - 1)
                answer += f"{level_prefix}- {heading.get('text', '')}\n"

        answer += "\n이 문서는 관련 규정과 절차를 상세히 기술한 자료입니다."

        return answer

    def _get_system_prompt(self, doc_type: str) -> str:
        """문서 타입별 시스템 프롬프트."""
        prompts = {
            "terms_and_conditions": "당신은 보험 약관 전문가입니다. 약관 조항을 명확하고 정확하게 설명해 주세요.",
            "business_method": "당신은 보험 요율 산정 전문가입니다. 보험료 계산 방법을 논리적으로 설명해 주세요.",
            "product_summary": "당신은 보험 상품 설명 전문가입니다. 상품 특징을 알기 쉽게 설명해 주세요.",
            "fss_release": "당신은 금융감독원 자료 분석 전문가입니다. 정책과 규정을 정확하게 설명해 주세요.",
            "court_precedent": "당신은 보험 관련 법률 전문가입니다. 판례의 법리를 명확하게 설명해 주세요.",
            "claims_process": "당신은 보험금 청구 절차 전문가입니다. 처리 과정을 단계별로 설명해 주세요.",
            "general": "당신은 보험 전문가입니다. 관련 질문에 정확하고 상세하게 답변해 주세요.",
        }

        return prompts.get(doc_type, prompts["general"])

    def _convert_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """예제를 지정된 형식으로 변환."""
        if self.output_format == OutputFormat.OPENAI:
            return self._to_openai(example)
        elif self.output_format == OutputFormat.ALPACA:
            return self._to_alpaca(example)
        elif self.output_format == OutputFormat.SHAREGPT:
            return self._to_sharegpt(example)
        elif self.output_format == OutputFormat.QWEN3:
            return self._to_qwen3(example)
        elif self.output_format == OutputFormat.YAML:
            return example
        else:
            return example

    def _to_openai(self, data: Dict) -> Dict:
        """OpenAI Chat format."""
        return to_openai(
            data,
            include_metadata=self.include_metadata,
            explicit_metadata=data.get("metadata"),
            meta_key="_meta",
        )

    def _to_alpaca(self, data: Dict) -> Dict:
        """Alpaca instruction format."""
        return to_alpaca(
            data,
            include_metadata=self.include_metadata,
            explicit_metadata=data.get("metadata"),
            meta_key="_meta",
        )

    def _to_sharegpt(self, data: Dict) -> Dict:
        """ShareGPT conversation format."""
        return to_sharegpt(
            data,
            include_metadata=self.include_metadata,
            explicit_metadata=data.get("metadata"),
            meta_key="_meta",
        )

    def _to_qwen3(self, data: Dict) -> Dict:
        """Qwen3 ChatML format with thinking support."""
        return to_qwen3(
            data,
            include_metadata=self.include_metadata,
            explicit_metadata=data.get("metadata"),
            meta_key="metadata",
            add_thinking=self.output_format == OutputFormat.QWEN3,
            thinking_threshold=100,
            thinking_text=DEFAULT_QWEN3_THINKING,
        )

    def convert_batch(self, documents: List[ParsedDocument]) -> List[Dict[str, Any]]:
        """배치 변환."""
        all_examples = []
        for doc in documents:
            examples = self.convert_document(doc)
            all_examples.extend(examples)

        # 무결성 검증
        if self.validate_integrity:
            all_examples = self._validate_integrity(all_examples)

        return all_examples

    def _validate_integrity(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 무결성 검증."""
        validated = []

        for example in examples:
            # 필수 필드 확인
            if not example.get("output") or len(example["output"].strip()) < 10:
                continue

            # 중복 내용 확인
            content_hash = hashlib.md5(f"{example.get('input', '')}{example.get('output', '')}".encode()).hexdigest()

            example["content_hash"] = content_hash

            # 유효한 예제만 추가
            if self._is_valid_example(example):
                validated.append(example)

        # 중복 제거
        seen_hashes = set()
        unique_examples = []
        for example in validated:
            if example["content_hash"] not in seen_hashes:
                seen_hashes.add(example["content_hash"])
                unique_examples.append(example)

        return unique_examples

    def _is_valid_example(self, example: Dict[str, Any]) -> bool:
        """예제 유효성 검사."""
        output = example.get("output", "")

        # 최소 길이 확인
        if len(output) < 20:
            return False

        # 반복 패턴 확인
        if output.count(output[:10]) > len(output) / 20:
            return False

        # 특수문자 과다 확인
        special_char_ratio = sum(1 for c in output if not c.isalnum() and c not in " \n\t.,;:!") / len(output)
        if special_char_ratio > 0.3:
            return False

        return True

    def export(self, documents: List[ParsedDocument], output_path: str | Path) -> int:
        """YAML 파일로 내보내기.

        Args:
            documents: 내보낼 문서 목록
            output_path: 출력 파일 경로

        Returns:
            int: 내보낸 예제 수
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 문서들을 예제로 변환
        examples = self.convert_batch(documents)

        # YAML 형식으로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                examples,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )

        return len(examples)


class YAMLReader:
    """YAML 파일 읽기 유틸리티."""

    @staticmethod
    def read(file_path: str | Path) -> List[Dict[str, Any]]:
        """전체 파일 읽기.

        Args:
            file_path: 파일 경로

        Returns:
            List[Dict]: 예제 목록
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data if isinstance(data, list) else [data]

    @staticmethod
    def count_examples(file_path: str | Path) -> int:
        """예제 수 카운트.

        Args:
            file_path: 파일 경로

        Returns:
            int: 예제 수
        """
        examples = YAMLReader.read(file_path)
        return len(examples)


class YAMLMerger:
    """YAML 파일 병합 유틸리티."""

    @staticmethod
    def merge(
        input_files: List[str | Path],
        output_file: str | Path,
        deduplicate: bool = True,
        shuffle: bool = False,
        seed: int = 42,
    ) -> int:
        """여러 YAML 파일 병합.

        Args:
            input_files: 입력 파일 목록
            output_file: 출력 파일 경로
            deduplicate: 중복 제거 여부
            shuffle: 셔플 여부
            seed: 랜덤 시드

        Returns:
            int: 병합된 예제 수
        """
        all_examples = []
        seen_hashes = set()

        for file_path in input_files:
            examples = YAMLReader.read(file_path)

            for example in examples:
                if deduplicate:
                    # 해시 기반 중복 확인
                    content = json.dumps(example, sort_keys=True, ensure_ascii=False)
                    example_hash = hashlib.md5(content.encode()).hexdigest()

                    if example_hash in seen_hashes:
                        continue
                    seen_hashes.add(example_hash)

                all_examples.append(example)

        if shuffle:
            random.seed(seed)
            random.shuffle(all_examples)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                all_examples,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )

        return len(all_examples)


class YAMLSplitter:
    """YAML 파일 분할 유틸리티."""

    @staticmethod
    def split(
        input_file: str | Path,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> Dict[str, int]:
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
            Dict[str, int]: 분할별 예제 수
        """
        examples = YAMLReader.read(input_file)

        splits = split_items(
            examples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            shuffle=shuffle,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        counts = {}
        for split_name, split_examples in splits.items():
            output_path = output_dir / f"{split_name}.yaml"
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    split_examples,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2,
                )
            counts[split_name] = len(split_examples)

        return counts


__all__ = [
    "YAMLConverter",
    "YAMLReader",
    "YAMLMerger",
    "YAMLSplitter",
]
