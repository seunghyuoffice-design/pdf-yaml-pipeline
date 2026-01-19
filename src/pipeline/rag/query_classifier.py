# SPDX-License-Identifier: MIT
"""질문 유형 분류기."""

from enum import Enum


class QueryType(str, Enum):
    DEFINITION = "definition"  # 정의/보장/면책 질문 → canonical만
    EXPLANATION = "explanation"  # 쉽게 설명 → canonical + summary
    OPERATION = "operation"  # 절차/처리 → canonical + operational


def classify_query(q: str) -> QueryType:
    q = q.lower()
    if any(k in q for k in ["절차", "방법", "처리", "운영", "심사", "프로세스"]):
        return QueryType.OPERATION
    if any(k in q for k in ["쉽게", "요약", "설명", "간단히", "핵심"]):
        return QueryType.EXPLANATION
    return QueryType.DEFINITION
