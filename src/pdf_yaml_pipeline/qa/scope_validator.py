"""스코프 참조 차단 검증기.

LLM 응답이 제공된 스코프 내에서만 답변했는지 검증합니다.

검증 레벨:
  - Level A: 입력 차단 (LLM에게 한 스코프만 제공)
  - Level B: 출력 검증 (응답에서 외부 참조 탐지)
  - Level C: 교차특약 라우팅 (필요 시 승인된 경로로만)

차단 대상:
  - 다른 특약명 언급
  - "다른 특약을 확인해야" 같은 외부 참조 표현
  - 스코프 텍스트에 없는 조항 인용
  - 일반 상식/추론 기반 답변
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.pipeline.qa.qa_prompt import QAResult

# ============================================================
# 검증 결과
# ============================================================


@dataclass
class ValidationResult:
    """검증 결과."""

    is_valid: bool
    violations: List[str] = field(default_factory=list)
    severity: str = "none"  # none | warning | error
    suggested_action: str = ""  # retry | reject | accept_with_warning

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "severity": self.severity,
            "suggested_action": self.suggested_action,
        }


# ============================================================
# 검증기
# ============================================================


class ScopeValidator:
    """스코프 참조 차단 검증기."""

    # 외부 참조 패턴
    EXTERNAL_REF_PATTERNS = [
        re.compile(r"다른\s*(특약|약관|담보)"),
        re.compile(r"(별도의|추가)\s*(특약|약관|담보)"),
        re.compile(r"(주계약|보통약관).*확인"),
        re.compile(r"일반적으로|통상적으로|보통|대체로"),
        re.compile(r"보험\s*상식|일반\s*원칙"),
        re.compile(r"(다른|별도)\s*조항.*참조"),
        re.compile(r"(해당|이)\s*문서\s*(외|밖)"),
    ]

    # 추론 표현 패턴 (근거 없이 추측)
    SPECULATION_PATTERNS = [
        re.compile(r"(아마도|아마|추측|예상)"),
        re.compile(r"(~일\s*것|~로\s*보임|~인\s*듯)"),
        re.compile(r"(명확하지\s*않지만|확실하지\s*않지만).*단정"),
    ]

    def __init__(
        self,
        scope_text: str,
        scope_title: str,
        known_scope_titles: Optional[Set[str]] = None,
    ):
        """
        Args:
            scope_text: 현재 스코프 텍스트
            scope_title: 현재 스코프 제목
            known_scope_titles: 알려진 모든 스코프 제목 (다른 특약명 탐지용)
        """
        self.scope_text = scope_text
        self.scope_title = scope_title
        self.known_scope_titles = known_scope_titles or set()

        # 스코프 내 조항 번호 추출
        self.scope_articles = self._extract_articles(scope_text)

    def _extract_articles(self, text: str) -> Set[str]:
        """텍스트에서 조항 번호 추출."""
        pattern = re.compile(r"제\s*(\d+)\s*조")
        matches = pattern.findall(text)
        return {f"제{n}조" for n in matches}

    def validate(self, result: QAResult) -> ValidationResult:
        """QA 결과 검증.

        Args:
            result: QA 결과

        Returns:
            ValidationResult
        """
        violations = []

        # 1. 다른 특약명 언급 검사
        other_scopes = self._check_other_scope_mentions(result.answer)
        if other_scopes:
            violations.append(f"다른 특약 언급: {', '.join(other_scopes)}")

        # 2. 외부 참조 표현 검사
        external_refs = self._check_external_references(result.answer)
        if external_refs:
            violations.append(f"외부 참조 표현: {', '.join(external_refs)}")

        # 3. 스코프 외 조항 인용 검사
        invalid_articles = self._check_article_validity(result.evidence)
        if invalid_articles:
            violations.append(f"스코프 외 조항 인용: {', '.join(invalid_articles)}")

        # 4. 근거 없는 추측 검사
        speculations = self._check_speculation(result.answer)
        if speculations:
            violations.append(f"근거 없는 추측 표현: {', '.join(speculations)}")

        # 5. 면책 확인 누락 검사 (보장 판단 시)
        if result.decision == "보장" and not result.exclusions_checked:
            violations.append("면책 조항 확인 누락")

        # 6. 근거 인용 누락 검사
        if result.decision in ("보장", "비보장") and not result.evidence:
            violations.append("근거 인용 누락")

        # 결과 결정
        if not violations:
            return ValidationResult(
                is_valid=True,
                severity="none",
                suggested_action="accept",
            )

        # 심각도 결정
        critical = any("다른 특약 언급" in v or "스코프 외 조항" in v for v in violations)

        if critical:
            return ValidationResult(
                is_valid=False,
                violations=violations,
                severity="error",
                suggested_action="retry",
            )
        else:
            return ValidationResult(
                is_valid=False,
                violations=violations,
                severity="warning",
                suggested_action="accept_with_warning",
            )

    def _check_other_scope_mentions(self, text: str) -> List[str]:
        """다른 특약명 언급 검사."""
        found = []
        for title in self.known_scope_titles:
            if title == self.scope_title:
                continue
            # 정규식 이스케이프
            pattern = re.escape(title)
            if re.search(pattern, text):
                found.append(title)
        return found

    def _check_external_references(self, text: str) -> List[str]:
        """외부 참조 표현 검사."""
        found = []
        for pattern in self.EXTERNAL_REF_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                found.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
        return list(set(found))

    def _check_article_validity(self, evidence: list) -> List[str]:
        """스코프 외 조항 인용 검사."""
        invalid = []
        for e in evidence:
            article = e.article if hasattr(e, "article") else e.get("article", "")
            # 조항 번호 정규화
            match = re.search(r"제\s*(\d+)\s*조", article)
            if match:
                normalized = f"제{match.group(1)}조"
                if normalized not in self.scope_articles:
                    invalid.append(article)
        return invalid

    def _check_speculation(self, text: str) -> List[str]:
        """추측 표현 검사."""
        found = []
        for pattern in self.SPECULATION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                found.extend(matches)
        return list(set(found))


# ============================================================
# 교차특약 라우터
# ============================================================


class CrossScopeRouter:
    """교차특약 질문 라우터.

    질문이 여러 스코프에 걸쳐 있을 때 필요한 스코프를 판정합니다.
    모델이 임의로 섞는 게 아니라 파이프라인 단계로 처리.
    """

    # 교차특약 신호 패턴
    CROSS_SCOPE_PATTERNS = [
        re.compile(r"(주계약|보통약관).*와.*특약"),
        re.compile(r"(두|여러)\s*(특약|담보)"),
        re.compile(r"(전체|종합|통합).*보장"),
        re.compile(r"(모든|각)\s*(특약|담보)"),
    ]

    def __init__(self, all_scope_titles: List[str]):
        """
        Args:
            all_scope_titles: 문서 내 모든 스코프 제목
        """
        self.all_scope_titles = all_scope_titles

    def detect_needed_scopes(self, question: str) -> List[str]:
        """질문에 필요한 스코프 목록 판정.

        Args:
            question: 사용자 질문

        Returns:
            필요한 스코프 제목 목록 (비어있으면 단일 스코프 QA)
        """
        needed = []

        # 1. 명시적 특약명 언급 확인
        for title in self.all_scope_titles:
            if title in question:
                needed.append(title)

        # 2. 교차특약 신호 확인
        for pattern in self.CROSS_SCOPE_PATTERNS:
            if pattern.search(question):
                # 명시적 특약이 없으면 모든 특약 필요
                if not needed:
                    return self.all_scope_titles.copy()
                break

        return needed

    def is_cross_scope_question(self, question: str) -> bool:
        """교차특약 질문인지 판정."""
        needed = self.detect_needed_scopes(question)
        return len(needed) > 1 or any(p.search(question) for p in self.CROSS_SCOPE_PATTERNS)


# ============================================================
# 유틸리티
# ============================================================


def validate_qa_result(
    result: QAResult,
    scope_text: str,
    scope_title: str,
    known_scope_titles: Optional[Set[str]] = None,
) -> ValidationResult:
    """QA 결과 검증 (편의 함수)."""
    validator = ScopeValidator(
        scope_text=scope_text,
        scope_title=scope_title,
        known_scope_titles=known_scope_titles,
    )
    return validator.validate(result)
