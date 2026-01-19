"""특약 QA 결과 병합 엔진.

병합 규칙(권장):
  - 비보장(면책) > 조건부 > 보장 > 근거부족 (보수적)
  - exclusions_checked=False인 '보장'은 자동으로 '조건부'로 강등
  - 근거 인용(evidence) 없는 답변은 신뢰도 하향 및 경고

출력 형태: 조건부 출력 (단정 금지)
  - 항상 근거+조건을 함께 표시
  - 법적 안전성 높음
  - 사용자가 최종 판단
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


# ============================================================
# 결정 타입
# ============================================================


class Decision(str, Enum):
    """판단 결과."""

    COVERED = "보장"
    NOT_COVERED = "비보장"
    CONDITIONAL = "조건부"
    INSUFFICIENT = "근거부족"

    @classmethod
    def from_str(cls, s: str) -> "Decision":
        mapping = {
            "보장": cls.COVERED,
            "비보장": cls.NOT_COVERED,
            "조건부": cls.CONDITIONAL,
            "근거부족": cls.INSUFFICIENT,
        }
        return mapping.get(s, cls.INSUFFICIENT)


# ============================================================
# 데이터 클래스
# ============================================================


@dataclass
class ScopeQAResult:
    """스코프별 QA 결과."""

    scope_id: str
    scope_title: str
    decision: Decision
    answer: str
    evidence: List[Dict[str, str]] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    exclusions_checked: bool = False
    exclusions_found: List[str] = field(default_factory=list)
    confidence: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "scope_id": self.scope_id,
            "scope_title": self.scope_title,
            "decision": self.decision.value if isinstance(self.decision, Decision) else self.decision,
            "answer": self.answer,
            "evidence": self.evidence,
            "conditions": self.conditions,
            "exclusions_checked": self.exclusions_checked,
            "exclusions_found": self.exclusions_found,
            "confidence": self.confidence,
            "meta": self.meta,
        }


@dataclass
class Conflict:
    """충돌 정보."""

    type: str  # decision_conflict | exclusion_conflict | evidence_conflict
    details: str
    scopes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "details": self.details,
            "scopes": self.scopes,
        }


@dataclass
class MergedResult:
    """병합된 최종 결과."""

    doc_id: str
    question: str
    final_decision: Decision
    final_answer: str
    scope_results: List[ScopeQAResult]
    conflicts: List[Conflict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    all_conditions: List[str] = field(default_factory=list)
    all_exclusions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "question": self.question,
            "final_decision": self.final_decision.value,
            "final_answer": self.final_answer,
            "scope_results": [r.to_dict() for r in self.scope_results],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "warnings": self.warnings,
            "all_conditions": self.all_conditions,
            "all_exclusions": self.all_exclusions,
        }


# ============================================================
# 병합 엔진
# ============================================================


class MergeEngine:
    """특약별 QA 결과 병합 엔진."""

    # 결정 우선순위 (높을수록 우선)
    PRIORITY = {
        Decision.NOT_COVERED: 3,  # 비보장 최우선
        Decision.CONDITIONAL: 2,
        Decision.COVERED: 1,
        Decision.INSUFFICIENT: 0,
    }

    def merge(
        self,
        doc_id: str,
        question: str,
        scope_results: List[ScopeQAResult],
    ) -> MergedResult:
        """스코프별 결과를 병합.

        Args:
            doc_id: 문서 ID
            question: 원본 질문
            scope_results: 스코프별 QA 결과

        Returns:
            MergedResult
        """
        warnings: List[str] = []
        conflicts: List[Conflict] = []

        if not scope_results:
            return MergedResult(
                doc_id=doc_id,
                question=question,
                final_decision=Decision.INSUFFICIENT,
                final_answer="문서 근거 부족입니다. 관련 특약/담보를 확인해야 합니다.",
                scope_results=[],
                warnings=["No scope results provided"],
            )

        # 1. 정규화 (근거 누락 시 경고, 면책 미확인 시 강등)
        normalized = [self._normalize(r, warnings) for r in scope_results]

        # 2. 우선순위 정렬
        normalized.sort(
            key=lambda r: (self.PRIORITY.get(r.decision, 0), r.confidence),
            reverse=True,
        )

        # 3. 충돌 탐지
        conflicts.extend(self._detect_conflicts(normalized))

        # 4. 최종 결정
        top = normalized[0]
        final_decision = top.decision

        # 5. 조건/면책 통합
        all_conditions = []
        all_exclusions = []
        for r in normalized:
            all_conditions.extend(r.conditions)
            all_exclusions.extend(r.exclusions_found)

        # 6. 최종 답변 구성
        final_answer = self._compose_final_answer(question, normalized, final_decision, all_conditions, all_exclusions)

        return MergedResult(
            doc_id=doc_id,
            question=question,
            final_decision=final_decision,
            final_answer=final_answer,
            scope_results=normalized,
            conflicts=conflicts,
            warnings=warnings,
            all_conditions=list(set(all_conditions)),
            all_exclusions=list(set(all_exclusions)),
        )

    def _normalize(self, r: ScopeQAResult, warnings: List[str]) -> ScopeQAResult:
        """결과 정규화."""
        # evidence 없으면 경고 + confidence 하향
        if not r.evidence:
            warnings.append(f"Missing evidence: {r.scope_id}")
            r.confidence = min(r.confidence, 0.55)

        # 면책 확인 안 했는데 보장 단정 → 조건부로 강등
        if r.decision == Decision.COVERED and not r.exclusions_checked:
            warnings.append(f"Covered without exclusions_checked → downgrade: {r.scope_id}")
            r.decision = Decision.CONDITIONAL
            r.conditions.append("미지급/면책 조항 확인 필요")
            r.confidence = min(r.confidence, 0.7)

        # confidence 범위 클램프
        r.confidence = max(0.0, min(1.0, r.confidence))

        return r

    def _detect_conflicts(self, results: List[ScopeQAResult]) -> List[Conflict]:
        """충돌 탐지."""
        conflicts = []

        decisions = {r.decision for r in results}

        # 보장 vs 비보장 충돌
        if Decision.COVERED in decisions and Decision.NOT_COVERED in decisions:
            conflicts.append(
                Conflict(
                    type="decision_conflict",
                    details="특약별 판단이 '보장'과 '비보장'으로 충돌합니다.",
                    scopes=[
                        {
                            "scope_id": r.scope_id,
                            "decision": r.decision.value,
                            "confidence": r.confidence,
                        }
                        for r in results
                    ],
                )
            )

        # 면책 발견된 경우
        exclusion_scopes = [r for r in results if r.exclusions_found]
        if exclusion_scopes:
            conflicts.append(
                Conflict(
                    type="exclusion_found",
                    details="일부 특약에서 면책/미지급 사유가 발견되었습니다.",
                    scopes=[
                        {
                            "scope_id": r.scope_id,
                            "exclusions": r.exclusions_found,
                        }
                        for r in exclusion_scopes
                    ],
                )
            )

        return conflicts

    def _compose_final_answer(
        self,
        question: str,
        results: List[ScopeQAResult],
        final_decision: Decision,
        all_conditions: List[str],
        all_exclusions: List[str],
    ) -> str:
        """최종 답변 구성 (조건부 출력)."""
        lines: List[str] = []

        # 헤더
        lines.append(f"## 질문\n{question}\n")
        lines.append(f"## 판단\n**{final_decision.value}**\n")

        # 주요 근거 (최고 우선순위 스코프)
        main = results[0]
        lines.append(f"## 주요 근거\n**{main.scope_title}** (신뢰도: {main.confidence:.0%})\n")
        lines.append(main.answer.strip())

        if main.evidence:
            lines.append("\n### 인용 근거")
            for ev in main.evidence[:3]:  # 최대 3개
                article = ev.get("article", "")
                quote = ev.get("quote", "")
                lines.append(f'- {article}: "{quote}"')

        # 조건 목록
        if all_conditions:
            lines.append("\n## 충족 필요 조건")
            for cond in set(all_conditions):
                lines.append(f"- {cond}")

        # 면책/미지급 사유
        if all_exclusions:
            lines.append("\n## 주의: 면책/미지급 사유")
            for exc in set(all_exclusions):
                lines.append(f"- {exc}")

        # 다른 스코프 요약
        if len(results) > 1:
            lines.append("\n## 기타 특약/담보 판단")
            for r in results[1:]:
                lines.append(f"- **{r.scope_title}**: {r.decision.value} " f"(신뢰도: {r.confidence:.0%})")

        # 면책사항
        lines.append("\n---")
        lines.append(
            "*본 판단은 제공된 약관 텍스트만을 근거로 하며, " "최종 보험금 지급 여부는 보험사의 심사를 따릅니다.*"
        )

        return "\n".join(lines)


# ============================================================
# 유틸리티
# ============================================================


def merge_qa_results(
    doc_id: str,
    question: str,
    scope_results: List[ScopeQAResult],
) -> MergedResult:
    """QA 결과 병합 (편의 함수)."""
    engine = MergeEngine()
    return engine.merge(doc_id, question, scope_results)
