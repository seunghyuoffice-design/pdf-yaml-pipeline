# SPDX-License-Identifier: MIT
"""
Distillation 데이터 품질 규칙.

핵심:
- canonical 없으면 fail
- CoT 누수 탐지
- 근거성(grounding) 검증
- 법적 과단정 경고
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Chain-of-Thought 누수 패턴
COT_LEAK_PATTERNS = [
    r"\b생각(해보면|하면|해 봅시다)\b",
    r"\b추론\b",
    r"\b단계(별|적으로)\b",
    r"\b먼저\b.*\b다음\b",
    r"\blet'?s think\b",
    r"\bchain of thought\b",
    r"\bstep\s*by\s*step\b",
]

# 환각 위험 패턴
HALLUCINATION_RISK_PATTERNS = [
    r"\b일반적으로\b",
    r"\b보통\b",
    r"\b대개\b",
    r"\b통상\b",
    r"\b대부분\b",
]

# 법적 과단정 패턴
LEGAL_OVERASSERT_PATTERNS = [
    r"\b반드시\b",
    r"\b무조건\b",
    r"\b항상\b",
    r"\b확실히\b",
]

REQUIRED_FIELDS = ["instruction", "context", "output"]
REQUIRED_CONTEXT_KEYS = ["canonical"]


@dataclass
class Finding:
    code: str
    level: str  # "fail" | "warn" | "info"
    message: str


def _has_pattern(text: str, patterns: List[str]) -> Optional[str]:
    t = text or ""
    for p in patterns:
        if re.search(p, t, flags=re.IGNORECASE | re.MULTILINE):
            return p
    return None


def validate_schema(rec: Dict[str, Any]) -> List[Finding]:
    """스키마 검증."""
    f: List[Finding] = []

    for k in REQUIRED_FIELDS:
        if k not in rec:
            f.append(Finding("schema.missing_field", "fail", f"Missing field: {k}"))

    ctx = rec.get("context")
    if not isinstance(ctx, dict):
        f.append(Finding("schema.context_not_object", "fail", "context must be an object"))
        return f

    for k in REQUIRED_CONTEXT_KEYS:
        if not (ctx.get(k) or "").strip():
            f.append(Finding("schema.missing_canonical", "fail", "context.canonical must be non-empty"))

    if not (rec.get("instruction") or "").strip():
        f.append(Finding("schema.empty_instruction", "fail", "instruction must be non-empty"))

    if not (rec.get("output") or "").strip():
        f.append(Finding("schema.empty_output", "fail", "output must be non-empty"))

    return f


def validate_meta(rec: Dict[str, Any]) -> List[Finding]:
    """메타데이터 검증 (product_id, version) - 필수."""
    f: List[Finding] = []
    meta = rec.get("meta") or {}

    if not meta.get("product_id"):
        f.append(Finding("meta.missing_product_id", "fail", "meta.product_id is required"))

    if not meta.get("version"):
        f.append(Finding("meta.missing_version", "fail", "meta.version is required"))

    return f


def validate_cot_leak(rec: Dict[str, Any]) -> List[Finding]:
    """Chain-of-Thought 누수 검사."""
    out = rec.get("output") or ""
    p = _has_pattern(out, COT_LEAK_PATTERNS)
    if p:
        return [Finding("safety.cot_leak", "fail", f"Chain-of-thought leak: {p}")]
    return []


def validate_length(rec: Dict[str, Any], min_chars: int = 20, max_chars: int = 1200) -> List[Finding]:
    """답변 길이 검사."""
    out = (rec.get("output") or "").strip()
    n = len(out)
    f: List[Finding] = []

    if n < min_chars:
        f.append(Finding("quality.too_short", "warn", f"output too short: {n} chars"))

    if n > max_chars:
        f.append(Finding("quality.too_long", "warn", f"output too long: {n} chars"))

    return f


def validate_role_policy(rec: Dict[str, Any]) -> List[Finding]:
    """
    역할 정책 검증.

    canonical 없으면 fail (스키마에서 처리)
    output이 summary/operational에만 의존하면 warn
    """
    ctx = rec.get("context") or {}
    canonical = (ctx.get("canonical") or "").strip()
    summary = (ctx.get("summary") or "").strip()
    operational = (ctx.get("operational") or "").strip()
    out = (rec.get("output") or "").strip()

    f: List[Finding] = []

    if not canonical:
        return f

    # 토큰 기반 검사
    def token_set(t: str) -> set:
        toks = re.findall(r"[A-Za-z0-9가-힣]{2,}", t)
        return set(toks)

    out_toks = token_set(out)
    can_toks = token_set(canonical)
    sum_toks = token_set(summary) if summary else set()
    op_toks = token_set(operational) if operational else set()

    if out_toks:
        overlap_can = len(out_toks & can_toks) / max(1, len(out_toks))
        overlap_sum = len(out_toks & sum_toks) / max(1, len(out_toks)) if summary else 0.0
        overlap_op = len(out_toks & op_toks) / max(1, len(out_toks)) if operational else 0.0

        if overlap_can < 0.18 and (overlap_sum > 0.25 or overlap_op > 0.25):
            f.append(
                Finding("grounding.summary_operational_dominant", "warn", f"Low canonical overlap ({overlap_can:.2f})")
            )

    return f


def validate_grounding_minimal(rec: Dict[str, Any], min_quote_hits: int = 1) -> List[Finding]:
    """
    최소 근거성 검사.

    output의 4-gram이 canonical에 존재하는지 검사.
    """
    ctx = rec.get("context") or {}
    canonical = ctx.get("canonical") or ""
    out = rec.get("output") or ""

    f: List[Finding] = []

    if not canonical.strip() or not out.strip():
        return f

    words = [w for w in re.findall(r"[A-Za-z0-9가-힣]+", out) if len(w) >= 2]
    if len(words) < 8:
        return f

    # 4-gram 매칭
    hits = 0
    for i in range(0, min(len(words) - 3, 40), 3):
        phrase = " ".join(words[i : i + 4])
        if phrase and phrase in canonical:
            hits += 1
            if hits >= min_quote_hits:
                break

    if hits < min_quote_hits:
        f.append(Finding("grounding.weak", "warn", "Weak grounding: no phrase match"))

    return f


def validate_tone_risk(rec: Dict[str, Any]) -> List[Finding]:
    """톤 위험 검사."""
    out = rec.get("output") or ""
    f: List[Finding] = []

    p1 = _has_pattern(out, HALLUCINATION_RISK_PATTERNS)
    if p1:
        f.append(Finding("style.generic_assumption", "warn", f"Generic assumption: {p1}"))

    p2 = _has_pattern(out, LEGAL_OVERASSERT_PATTERNS)
    if p2:
        f.append(Finding("style.overassert", "warn", f"Over-assertive: {p2}"))

    return f


def validate_cross_document(rec: Dict[str, Any]) -> List[Finding]:
    """교차 문서 혼합 방지."""
    f: List[Finding] = []
    meta = rec.get("meta") or {}
    ctx = rec.get("context") or {}

    product_id = meta.get("product_id", "")
    if not product_id:
        return f

    # 컨텍스트의 source_path가 product_id와 일치하는지 검사
    for role in ["canonical", "summary", "operational"]:
        text = ctx.get(role) or ""
        if text and product_id not in text:
            # source_path 패턴 검사
            if "[" in text and "]" in text:
                # [source_path] 형태가 있으면 검사
                pass  # 상세 검사는 실제 데이터 형태에 따라

    return f


def validate_canonical_quote(rec: Dict[str, Any]) -> List[Finding]:
    """
    Canonical 인용 검증.

    - 따옴표로 감싼 인용이 canonical에 실제 존재하는지 확인
    - 인용이 없으면 권장 위반 (definition/operation/multimodal)
    - 인용이 canonical에 없으면 환각 → FAIL
    """
    f: List[Finding] = []

    ctx = rec.get("context") or {}
    canonical = (ctx.get("canonical") or "").strip()
    output = (rec.get("output") or "").strip()
    qtype = (rec.get("meta") or {}).get("qtype", "definition")

    if not canonical or not output:
        return f

    # 따옴표로 감싼 인용 추출 (6자 이상)
    quotes = re.findall(r'["""\'\u2018\u2019]([^"""\'\u2018\u2019]{6,})["""\'\u2018\u2019]', output)

    if not quotes:
        # 인용 없음 → 권장 위반 (definition/operation/multimodal에서)
        if qtype in ("definition", "operation", "multimodal"):
            f.append(
                Finding("grounding.no_canonical_quote", "warn", "No canonical quote found (recommended for grounding)")
            )
        return f

    # 인용이 canonical에 실제 존재하는지 확인
    for q in quotes:
        q_clean = q.strip()
        if q_clean and q_clean not in canonical:
            f.append(Finding("grounding.invalid_quote", "fail", f"Quoted text not in canonical: '{q_clean[:30]}...'"))

    return f
