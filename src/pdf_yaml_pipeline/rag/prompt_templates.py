# SPDX-License-Identifier: MIT
"""
보험 전용 LLM 프롬프트 템플릿.

핵심 원칙:
- 약관(canonical) = 사실의 근거
- 상품요약서(summary) = 설명 보조
- 사업방법서(operational) = 절차/운영
"""

from __future__ import annotations

from typing import List, Optional

from pdf_yaml_pipeline.rag.query_classifier import QueryType

SYSTEM_PROMPT = """너는 보험 약관을 기반으로 답변하는 전문 AI다.

규칙:
- 모든 사실적 답변은 반드시 '약관(canonical)'을 근거로 한다.
- 상품요약서(summary)는 이해를 돕는 설명으로만 사용한다.
- 사업방법서(operational)는 절차·운영 질문에만 사용한다.
- 약관에 명시되지 않은 내용은 추측하거나 단정하지 않는다.
- 불확실한 경우 "약관에 명시되어 있지 않습니다"라고 답한다.
- 법적 효력이 있는 표현은 약관 문구를 우선한다."""


# 질문 유형별 프롬프트
DEFINITION_TEMPLATE = """아래 약관 내용을 기준으로 질문에 답변하라.

- 약관에 명시된 내용만 사실로 사용한다.
- 요약서나 사업방법서의 내용은 사용하지 않는다.

[약관]
{canonical_context}

질문:
{user_question}"""


EXPLANATION_TEMPLATE = """아래 약관 내용을 기준으로 사실을 유지하되,
상품요약서의 설명을 참고하여 쉽게 풀어서 설명하라.

- 사실 판단은 약관을 기준으로 한다.
- 요약서는 이해를 돕는 표현으로만 사용한다.

[약관]
{canonical_context}

[상품요약서]
{summary_context}

질문:
{user_question}"""


OPERATION_TEMPLATE = """아래 약관을 기준으로 원칙을 설명하고,
사업방법서를 참고하여 실제 처리 절차를 설명하라.

- 권리·의무는 약관 기준
- 절차·운영은 사업방법서 참고

[약관]
{canonical_context}

[사업방법서]
{operational_context}

질문:
{user_question}"""


ANSWER_SUFFIX = """

[출처]
{sources}"""


def build_prompt(
    question: str,
    qtype: QueryType,
    canonical_context: List[str],
    summary_context: Optional[List[str]] = None,
    operational_context: Optional[List[str]] = None,
) -> str:
    """
    질문 유형에 맞는 프롬프트 생성.

    Args:
        question: 사용자 질문
        qtype: 질문 유형
        canonical_context: 약관 컨텍스트
        summary_context: 상품요약서 컨텍스트
        operational_context: 사업방법서 컨텍스트

    Returns:
        완성된 프롬프트
    """
    canonical_str = "\n".join(canonical_context) if canonical_context else "(없음)"
    summary_str = "\n".join(summary_context) if summary_context else "(없음)"
    operational_str = "\n".join(operational_context) if operational_context else "(없음)"

    if qtype == QueryType.DEFINITION:
        return DEFINITION_TEMPLATE.format(
            canonical_context=canonical_str,
            user_question=question,
        )

    if qtype == QueryType.EXPLANATION:
        return EXPLANATION_TEMPLATE.format(
            canonical_context=canonical_str,
            summary_context=summary_str,
            user_question=question,
        )

    if qtype == QueryType.OPERATION:
        return OPERATION_TEMPLATE.format(
            canonical_context=canonical_str,
            operational_context=operational_str,
            user_question=question,
        )

    # fallback
    return DEFINITION_TEMPLATE.format(
        canonical_context=canonical_str,
        user_question=question,
    )


def format_answer_with_sources(
    answer: str,
    used_canonical: bool = True,
    used_summary: bool = False,
    used_operational: bool = False,
) -> str:
    """답변에 출처 정보 추가."""
    sources = []
    if used_canonical:
        sources.append("- 약관(canonical)")
    if used_summary:
        sources.append("- 상품요약서(summary)")
    if used_operational:
        sources.append("- 사업방법서(operational)")

    return answer + ANSWER_SUFFIX.format(sources="\n".join(sources))


# 금지 규칙 (LLM 파인튜닝 또는 가드레일용)
FORBIDDEN_BEHAVIORS = """
- 약관에 없는 내용을 일반적인 보험 상식으로 추론하지 마라.
- 요약서 문구를 법적 근거처럼 사용하지 마라.
- 사업방법서를 권리·의무의 근거로 사용하지 마라.
- 여러 상품의 내용을 섞지 마라.
"""
