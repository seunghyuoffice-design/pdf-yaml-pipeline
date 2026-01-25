# SPDX-License-Identifier: MIT
"""역할 기반 컨텍스트 조합."""

from __future__ import annotations

from typing import Any, Dict, List

from pdf_yaml_pipeline.rag.query_classifier import QueryType


def assemble_context(
    canonical_hits: List[Dict[str, Any]],
    extra_chunks: List[Dict[str, Any]],
    qtype: QueryType,
    max_extra: int = 3,
) -> List[str]:
    """
    역할 기반 컨텍스트 조합.

    Args:
        canonical_hits: FAISS 검색 결과 (약관)
        extra_chunks: 같은 product_id/version의 summary/operational
        qtype: 질문 유형
        max_extra: 보조 컨텍스트 최대 개수

    Returns:
        컨텍스트 문자열 리스트
    """
    ctx: List[str] = []

    # 1) 항상 약관(canonical) 먼저
    for h in canonical_hits:
        text = h.get("meta", {}).get("text") or h.get("text", "")
        if text:
            ctx.append(text)

    # 2) 질문 유형에 따라 보조 컨텍스트 추가
    if qtype == QueryType.EXPLANATION:
        extras = [c["text"] for c in extra_chunks if c.get("role") == "paraphrase"]
        ctx.extend(extras[:max_extra])

    if qtype == QueryType.OPERATION:
        extras = [c["text"] for c in extra_chunks if c.get("role") == "operational"]
        ctx.extend(extras[:max_extra])

    return [c for c in ctx if c]
