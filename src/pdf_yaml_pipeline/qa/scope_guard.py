"""LLM 출력 검증기: 스코프 위반 탐지.

스코프 위반 탐지 전략:
  1) evidence.quote가 스코프 텍스트에 실제로 존재하는지
  2) article(제N조)가 스코프 텍스트에 존재하는지
  3) 답변에 다른 특약명을 언급하는지
  4) '단정 표현'이 있는데 근거가 없으면 위반

사용 방식:
  - 위반이 1개라도 있으면 재시도 또는 근거부족으로 강제
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class ScopeViolation:
    """스코프 위반."""

    kind: str  # missing_quote | unknown_article | out_of_scope_name | unsupported_claim
    detail: str
    severity: str = "error"  # error | warning

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "severity": self.severity,
        }


class ScopeGuard:
    """LLM 출력 스코프 위반 탐지기."""

    RE_ARTICLE = re.compile(r"(제\s*\d+\s*조)")
    RE_DECISIVE = re.compile(
        r"(반드시|항상|전부|무조건|확실히|명백히|" r"지급됩니다|지급되지 않습니다|보장됩니다|보장되지 않습니다)"
    )
    RE_EXTERNAL = re.compile(r"(다른\s*(특약|약관|담보)|일반적으로|통상적으로|" r"보험\s*상식|별도\s*확인|추가\s*문의)")

    def __init__(
        self,
        scope_text: str,
        scope_title: str,
        known_other_scope_titles: Optional[Set[str]] = None,
    ):
        self.scope_text = scope_text
        self.scope_title = scope_title
        self.known_other_titles = known_other_scope_titles or set()
        self.scope_articles = set(self.RE_ARTICLE.findall(scope_text))

    def validate(self, llm_json: Dict) -> List[ScopeViolation]:
        """LLM 응답 검증."""
        violations: List[ScopeViolation] = []

        answer = (llm_json.get("answer") or "").strip()
        evidence = llm_json.get("evidence") or []
        decision = llm_json.get("decision", "")

        # 1. evidence 필수 검사
        if not evidence:
            violations.append(
                ScopeViolation(
                    kind="missing_quote",
                    detail="evidence가 비어 있습니다.",
                    severity="error",
                )
            )

        # 2. evidence 검증
        for ev in evidence:
            violations.extend(self._validate_evidence(ev))

        # 3. 다른 특약명 언급 검사
        for title in self.known_other_titles:
            if title and title in answer:
                violations.append(
                    ScopeViolation(
                        kind="out_of_scope_name",
                        detail=f"다른 특약명 언급 감지: {title}",
                        severity="error",
                    )
                )
                break

        # 4. 외부 참조 표현 검사
        external_matches = self.RE_EXTERNAL.findall(answer)
        if external_matches:
            violations.append(
                ScopeViolation(
                    kind="external_reference",
                    detail=f"외부 참조 표현 감지: {external_matches[0]}",
                    severity="warning",
                )
            )

        # 5. 단정 표현 + 근거 없음
        if self.RE_DECISIVE.search(answer) and not evidence:
            violations.append(
                ScopeViolation(
                    kind="unsupported_claim",
                    detail="단정적 표현이 있으나 근거 인용이 없습니다.",
                    severity="error",
                )
            )

        # 6. 보장/비보장 판단인데 면책 미확인
        if decision in ("보장", "비보장"):
            if not llm_json.get("exclusions_checked", False):
                violations.append(
                    ScopeViolation(
                        kind="exclusions_not_checked",
                        detail="보장/비보장 판단인데 면책 확인이 완료되지 않았습니다.",
                        severity="warning",
                    )
                )

        return violations

    def _validate_evidence(self, ev: Dict) -> List[ScopeViolation]:
        """개별 evidence 검증."""
        violations = []

        quote = (ev.get("quote") or "").strip()
        article = (ev.get("article") or "").strip()

        # quote가 원문에 없음
        if quote:
            normalized_quote = re.sub(r"\s+", "", quote)
            normalized_scope = re.sub(r"\s+", "", self.scope_text)

            if normalized_quote not in normalized_scope:
                if not self._fuzzy_match(quote, self.scope_text):
                    violations.append(
                        ScopeViolation(
                            kind="unsupported_claim",
                            detail=f"quote가 스코프 원문에 존재하지 않습니다: {quote[:40]}...",
                            severity="error",
                        )
                    )

        # article이 원문에 없음
        if article:
            article_matches = self.RE_ARTICLE.findall(article)
            for art in article_matches:
                if art not in self.scope_text:
                    violations.append(
                        ScopeViolation(
                            kind="unknown_article",
                            detail=f"article이 스코프 원문에 존재하지 않습니다: {art}",
                            severity="error",
                        )
                    )

        return violations

    def _fuzzy_match(self, quote: str, text: str, threshold: float = 0.5) -> bool:
        """부분 매칭."""
        quote_words = set(quote.split())
        if not quote_words:
            return False
        text_words = set(text.split())
        overlap = len(quote_words & text_words)
        return overlap / len(quote_words) >= threshold

    def has_critical_violations(self, violations: List[ScopeViolation]) -> bool:
        """심각한 위반 여부."""
        return any(v.severity == "error" for v in violations)


def force_insufficient(llm_json: Dict, violations: List[ScopeViolation]) -> Dict:
    """위반 발생 시 근거부족으로 강제."""
    return {
        **llm_json,
        "decision": "근거부족",
        "confidence": 0.0,
        "answer": (
            "스코프 위반으로 인해 판단할 수 없습니다.\n\n"
            "위반 사항:\n" + "\n".join(f"- [{v.kind}] {v.detail}" for v in violations)
        ),
        "warnings": [v.to_dict() for v in violations],
    }


def build_retry_prompt(violations: List[ScopeViolation]) -> str:
    """재시도 프롬프트 생성."""
    lines = [
        "이전 답변에서 다음 문제가 발견되었습니다:",
    ]
    for v in violations:
        lines.append(f"- {v.detail}")

    lines.extend(
        [
            "",
            "다음 규칙을 지켜 다시 답변하세요:",
            "1. 반드시 <SCOPE> 텍스트에서만 근거를 인용하세요.",
            "2. 인용문(quote)은 원문 그대로 작성하세요.",
            "3. 다른 특약/약관을 언급하지 마세요.",
            "4. 근거가 없으면 '근거부족'으로 답하세요.",
        ]
    )

    return "\n".join(lines)


def validate_llm_response(
    llm_json: Dict,
    scope_text: str,
    scope_title: str,
    known_other_scope_titles: Optional[Set[str]] = None,
) -> List[ScopeViolation]:
    """LLM 응답 검증 (편의 함수)."""
    guard = ScopeGuard(
        scope_text=scope_text,
        scope_title=scope_title,
        known_other_scope_titles=known_other_scope_titles,
    )
    return guard.validate(llm_json)


# ============================================================
# 호환성 인터페이스 (기존 __init__.py 지원)
# ============================================================


@dataclass
class ScopeGuardConfig:
    """ScopeGuard 설정."""

    max_retries: int = 2
    strict_mode: bool = True
    fuzzy_threshold: float = 0.5


@dataclass
class BatchQAConfig:
    """배치 QA 설정."""

    max_concurrent: int = 4
    timeout_per_scope: int = 60
    retry_on_violation: bool = True


class BatchScopeGuard:
    """배치 스코프 가드 (여러 스코프 동시 검증)."""

    def __init__(self, config: Optional[BatchQAConfig] = None):
        self.config = config or BatchQAConfig()

    def validate_batch(
        self,
        results: List[Dict],
        scope_texts: Dict[str, str],
        scope_titles: Dict[str, str],
    ) -> Dict[str, List[ScopeViolation]]:
        """배치 검증."""
        violations_map = {}
        for result in results:
            scope_id = result.get("scope_id", "")
            text = scope_texts.get(scope_id, "")
            title = scope_titles.get(scope_id, "")
            guard = ScopeGuard(text, title)
            violations_map[scope_id] = guard.validate(result)
        return violations_map


def create_scope_guard(
    scope_text: str,
    scope_title: str,
    config: Optional[ScopeGuardConfig] = None,
    known_other_scope_titles: Optional[Set[str]] = None,
) -> ScopeGuard:
    """ScopeGuard 팩토리 함수."""
    return ScopeGuard(
        scope_text=scope_text,
        scope_title=scope_title,
        known_other_scope_titles=known_other_scope_titles,
    )
