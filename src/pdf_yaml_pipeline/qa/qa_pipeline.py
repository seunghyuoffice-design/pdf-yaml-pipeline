"""QA 파이프라인 (전체 흐름).

파이프라인 흐름:
  (1) ChunkParser → 특약/담보 chunks
  (2) 질문 분류 → 필요한 scope 후보
  (3) Scope별 QA (스코프 고정 프롬프트)
  (4) Scope별 결과 저장
  (5) Merge Engine → 최종 답 + 충돌/근거 묶음

설계 원칙:
  - 파일 단위 병렬화: ✅
  - 페이지 단위 병렬화: ❌
  - 스코프 간 참조: ❌ (파이프라인으로만)
  - 조건부 출력: ✅ (단정 금지)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pdf_yaml_pipeline.parsers.special_clause_parser import (
    ClauseChunk,
    SpecialClauseParser,
)
from pdf_yaml_pipeline.qa.merge_engine import (
    Decision,
    MergedResult,
    MergeEngine,
    ScopeQAResult,
)
from pdf_yaml_pipeline.qa.qa_prompt import QAResult
from pdf_yaml_pipeline.qa.scope_guard import (
    BatchScopeGuard,
    ScopeGuard,
    ScopeGuardConfig,
)
from pdf_yaml_pipeline.qa.scope_validator import CrossScopeRouter

# ============================================================
# 설정
# ============================================================


@dataclass
class QAPipelineConfig:
    """QA 파이프라인 설정."""

    max_scope_chars: int = 12000  # 스코프 최대 문자 수
    max_retries: int = 2  # 검증 실패 시 최대 재시도
    strict_mode: bool = True  # 엄격 모드
    allow_cross_scope: bool = True  # 교차특약 질문 허용


# ============================================================
# QA 파이프라인
# ============================================================


class QAPipeline:
    """약관 QA 파이프라인."""

    def __init__(
        self,
        config: Optional[QAPipelineConfig] = None,
    ):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config or QAPipelineConfig()
        self.parser = SpecialClauseParser(max_chars=self.config.max_scope_chars)
        self.merge_engine = MergeEngine()

    def process_document(
        self,
        doc_id: str,
        doc_text: str,
        question: str,
        llm_call: Callable[[List[Dict[str, str]]], str],
    ) -> MergedResult:
        """문서에서 QA 처리.

        Args:
            doc_id: 문서 ID
            doc_text: 문서 텍스트 (Markdown)
            question: 사용자 질문
            llm_call: LLM 호출 함수

        Returns:
            MergedResult
        """
        # 1. 청크 파싱
        chunks = self.parser.parse(doc_text)

        if not chunks:
            return MergedResult(
                doc_id=doc_id,
                question=question,
                final_decision=Decision.INSUFFICIENT,
                final_answer="문서에서 특약/담보를 찾을 수 없습니다.",
                scope_results=[],
                warnings=["No chunks found in document"],
            )

        # 2. 스코프 준비
        scopes = [
            {
                "scope_id": f"{doc_id}:{c.chunk_type}:{i}",
                "scope_title": c.title,
                "scope_text": c.text,
            }
            for i, c in enumerate(chunks)
        ]

        all_titles = {s["scope_title"] for s in scopes}

        # 3. 교차특약 라우팅
        router = CrossScopeRouter(list(all_titles))
        needed_scopes = router.detect_needed_scopes(question)

        if needed_scopes:
            # 명시된 스코프만 처리
            target_scopes = [s for s in scopes if s["scope_title"] in needed_scopes]
        else:
            # 전체 스코프 처리 (또는 관련 스코프 자동 탐지)
            target_scopes = scopes

        # 4. 배치 QA 실행
        guard_config = ScopeGuardConfig(
            max_retries=self.config.max_retries,
            strict_mode=self.config.strict_mode,
        )

        batch_guard = BatchScopeGuard(target_scopes, guard_config)
        qa_results = batch_guard.execute_all(question, llm_call)

        # 5. 결과 변환
        scope_results = [self._convert_to_scope_result(r) for r in qa_results]

        # 6. 병합
        merged = self.merge_engine.merge(doc_id, question, scope_results)

        # 위반 로그 추가
        violations = batch_guard.get_all_violations()
        if violations:
            merged.warnings.append(f"Scope violations: {len(violations)}")

        return merged

    def process_chunks(
        self,
        doc_id: str,
        chunks: List[ClauseChunk],
        question: str,
        llm_call: Callable[[List[Dict[str, str]]], str],
    ) -> MergedResult:
        """이미 파싱된 청크에서 QA 처리.

        Args:
            doc_id: 문서 ID
            chunks: 청크 리스트
            question: 사용자 질문
            llm_call: LLM 호출 함수

        Returns:
            MergedResult
        """
        scopes = [
            {
                "scope_id": f"{doc_id}:{c.chunk_type}:{i}",
                "scope_title": c.title,
                "scope_text": c.text,
            }
            for i, c in enumerate(chunks)
        ]

        guard_config = ScopeGuardConfig(
            max_retries=self.config.max_retries,
            strict_mode=self.config.strict_mode,
        )

        batch_guard = BatchScopeGuard(scopes, guard_config)
        qa_results = batch_guard.execute_all(question, llm_call)

        scope_results = [self._convert_to_scope_result(r) for r in qa_results]

        return self.merge_engine.merge(doc_id, question, scope_results)

    def _convert_to_scope_result(self, qa: QAResult) -> ScopeQAResult:
        """QAResult → ScopeQAResult 변환."""
        return ScopeQAResult(
            scope_id=qa.scope_id,
            scope_title=qa.scope_title,
            decision=Decision.from_str(qa.decision),
            answer=qa.answer,
            evidence=[{"article": e.article, "quote": e.quote} for e in qa.evidence],
            conditions=qa.conditions,
            exclusions_checked=qa.exclusions_checked,
            exclusions_found=qa.exclusions_found,
            confidence=qa.confidence,
            meta={"warnings": qa.warnings},
        )


# ============================================================
# 단일 스코프 QA
# ============================================================


def single_scope_qa(
    scope_id: str,
    scope_title: str,
    scope_text: str,
    question: str,
    llm_call: Callable[[List[Dict[str, str]]], str],
    strict: bool = True,
) -> QAResult:
    """단일 스코프 QA (편의 함수).

    Args:
        scope_id: 스코프 ID
        scope_title: 스코프 제목
        scope_text: 스코프 텍스트
        question: 질문
        llm_call: LLM 호출 함수
        strict: 엄격 모드

    Returns:
        QAResult
    """
    guard = ScopeGuard(
        scope_id=scope_id,
        scope_title=scope_title,
        scope_text=scope_text,
        config=ScopeGuardConfig(strict_mode=strict),
    )
    return guard.execute_with_guard(question, llm_call)


# ============================================================
# 파일 기반 QA
# ============================================================


def qa_from_file(
    file_path: Path,
    question: str,
    llm_call: Callable[[List[Dict[str, str]]], str],
    config: Optional[QAPipelineConfig] = None,
) -> MergedResult:
    """파일에서 QA 처리 (편의 함수).

    Args:
        file_path: 문서 파일 경로 (Markdown)
        question: 질문
        llm_call: LLM 호출 함수
        config: 파이프라인 설정

    Returns:
        MergedResult
    """
    doc_id = file_path.stem
    doc_text = file_path.read_text(encoding="utf-8")

    pipeline = QAPipeline(config)
    return pipeline.process_document(doc_id, doc_text, question, llm_call)
