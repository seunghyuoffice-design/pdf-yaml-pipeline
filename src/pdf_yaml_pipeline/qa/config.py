"""QA 시스템 운영 정책 설정.

이 파일은 운영 기준을 고정합니다. 변경 시 반드시 리뷰 필요.

확정된 정책:
  - 출력 형태: 조건부 출력 (단정 금지)
  - JSON 스키마: 고정 (QAResult 표준)
  - 교차특약: 라우팅 + 병합 허용
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ============================================================
# 운영 정책 (LOCKED)
# ============================================================


@dataclass(frozen=True)
class OutputPolicy:
    """출력 정책 (변경 금지)."""

    # 조건부 출력 고정 - 단정 금지
    allow_definitive_answer: bool = False

    # 필수 출력 항목
    require_evidence: bool = True  # 근거 인용 필수
    require_conditions: bool = True  # 조건 명시 필수
    require_exclusions_check: bool = True  # 면책 확인 필수

    # 금지 문구
    forbidden_phrases: tuple = (
        "보장됩니다",
        "지급됩니다",
        "확실합니다",
        "틀림없이",
        "반드시",
        "100%",
    )


@dataclass(frozen=True)
class SchemaPolicy:
    """JSON 스키마 정책 (변경 금지)."""

    # 스키마 고정 - 모델별 어댑터 불허
    fixed_schema: bool = True

    # 필수 필드
    required_fields: tuple = (
        "scope_title",
        "question",
        "answer",
        "decision",
        "evidence",
        "exclusions_checked",
        "confidence",
    )

    # 결정 유형
    valid_decisions: tuple = (
        "보장",
        "비보장",
        "조건부",
        "근거부족",
    )


@dataclass(frozen=True)
class CrossScopePolicy:
    """교차특약 정책 (변경 금지)."""

    # 교차특약 허용 - 라우팅 + 병합
    allow_cross_scope: bool = True

    # 라우팅 방식
    auto_detect: bool = True  # 자동 감지
    require_explicit_scopes: bool = False  # 명시적 스코프 요구 안 함

    # 병합 방식
    merge_results: bool = True  # 결과 병합
    show_per_scope: bool = True  # 스코프별 결과 표시


@dataclass(frozen=True)
class ValidationPolicy:
    """검증 정책 (변경 금지)."""

    # 검증 실패 시 처리
    max_retries: int = 1  # 재시도 1회
    fallback_decision: str = "근거부족"  # 실패 시 기본 결정

    # 자동 강등 규칙
    downgrade_unchecked_exclusions: bool = True  # 면책 미확인 → 조건부
    downgrade_no_evidence: bool = True  # 근거 없음 → 신뢰도 하향

    # 차단 대상
    block_external_reference: bool = True  # 외부 참조 차단
    block_speculation: bool = True  # 추측 표현 차단


@dataclass(frozen=True)
class MergePolicy:
    """병합 정책 (변경 금지)."""

    # 우선순위 (높을수록 우선)
    priority_order: tuple = (
        "비보장",  # 3 - 최우선
        "조건부",  # 2
        "보장",  # 1
        "근거부족",  # 0
    )

    # 충돌 처리
    flag_decision_conflict: bool = True  # 보장 vs 비보장 충돌 표시
    flag_exclusion_found: bool = True  # 면책 발견 표시


# ============================================================
# 통합 설정
# ============================================================


@dataclass(frozen=True)
class QAOperationConfig:
    """QA 운영 설정 (통합)."""

    output: OutputPolicy = field(default_factory=OutputPolicy)
    schema: SchemaPolicy = field(default_factory=SchemaPolicy)
    cross_scope: CrossScopePolicy = field(default_factory=CrossScopePolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    merge: MergePolicy = field(default_factory=MergePolicy)


# 기본 설정 (싱글톤)
DEFAULT_CONFIG = QAOperationConfig()


# ============================================================
# 검증 함수
# ============================================================


def validate_output_text(text: str, config: QAOperationConfig = DEFAULT_CONFIG) -> List[str]:
    """출력 텍스트에서 금지 문구 검사.

    Returns:
        발견된 금지 문구 목록
    """
    violations = []
    for phrase in config.output.forbidden_phrases:
        if phrase in text:
            violations.append(phrase)
    return violations


def validate_decision(decision: str, config: QAOperationConfig = DEFAULT_CONFIG) -> bool:
    """결정 유형 유효성 검사."""
    return decision in config.schema.valid_decisions


def validate_response_schema(
    response: dict,
    config: QAOperationConfig = DEFAULT_CONFIG,
) -> List[str]:
    """응답 스키마 유효성 검사.

    Returns:
        누락된 필드 목록
    """
    missing = []
    for field_name in config.schema.required_fields:
        if field_name not in response:
            missing.append(field_name)
    return missing


# ============================================================
# 운영 기준 문서
# ============================================================

OPERATION_RULES = """
# QA 시스템 운영 기준 (v1.0)

## 1. 출력 정책
- 조건부 출력 고정 (단정 금지)
- 근거 + 조건 + 면책 가능성 항상 병기
- 금지 문구: 보장됩니다, 지급됩니다, 확실합니다, 틀림없이, 반드시, 100%

## 2. JSON 스키마
- QAResult 스키마 고정
- 모델별 어댑터 불허
- 필수 필드: scope_title, question, answer, decision, evidence, exclusions_checked, confidence

## 3. 교차특약
- 자동 감지 + 라우팅 + 병합 허용
- 스코프별 결과 항상 표시
- 충돌 시 명시적 표시

## 4. 검증
- LLM 출력은 반드시 검증기 통과
- 검증 실패 시 재시도 1회, 이후 근거부족 처리
- 면책 미확인 보장 → 자동 조건부 강등

## 5. 병합
- 우선순위: 비보장 > 조건부 > 보장 > 근거부족
- 보장 vs 비보장 충돌 시 conflict 표시
- 별표/부칙은 판단 근거로 사용 안 함

---
확정일: 2026-01-10
변경 시 반드시 리뷰 필요
"""


def print_operation_rules():
    """운영 기준 출력."""
    print(OPERATION_RULES)
