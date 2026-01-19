"""QA 프롬프트 템플릿 (스코프 고정).

특약/담보 단위로 스코프를 고정하여 LLM이 해당 범위 내에서만 답변하도록 합니다.

핵심 원칙:
  - 스코프 밖 참조 = 실패
  - 근거 인용 필수
  - 면책 조항 확인 필수
  - 조건부 출력 (단정 금지)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# 프롬프트 템플릿
# ============================================================

SYSTEM_PROMPT = """역할: 너는 보험 약관 해석을 돕는 QA 엔진이다.

규칙(반드시 준수):
1) 아래 <SCOPE>에 제공된 특약/담보 텍스트만 근거로 답한다.
2) <SCOPE> 밖의 지식(일반 보험 상식, 다른 특약, 유사 약관, 추론)을 사용하지 않는다.
3) 답변에 반드시 근거 조항(제N조) 또는 원문 인용(짧게)을 포함한다.
4) 근거가 없으면 반드시 "문서 근거 부족"이라고 답하고, 필요한 근거가 무엇인지 말한다.
5) 보장/면책 판단은 반드시 "지급사유"와 "미지급사유(면책)"를 둘 다 확인했을 때만 단정한다.
6) 최종 판단을 단정하지 말고, 근거와 조건을 함께 제시한다.
7) 확신도(confidence)를 0~1로 출력한다.

출력 형식(JSON만, 다른 텍스트 없이):
{
  "scope_title": "특약/담보명",
  "question": "원본 질문",
  "answer": "상세 답변 (근거 포함)",
  "decision": "보장|비보장|조건부|근거부족",
  "conditions": ["충족해야 할 조건 목록"],
  "evidence": [
    {"article": "제N조", "quote": "원문 인용 (50자 이내)"}
  ],
  "exclusions_checked": true,
  "exclusions_found": ["발견된 면책/미지급 사유"],
  "confidence": 0.0,
  "warnings": ["경고 사항"]
}"""

USER_PROMPT_TEMPLATE = """<SCOPE_TITLE>
{scope_title}
</SCOPE_TITLE>

<SCOPE>
{scope_text}
</SCOPE>

<QUESTION>
{question}
</QUESTION>

위 <SCOPE> 내용만을 근거로 질문에 답하세요. JSON 형식으로만 출력하세요."""


# ============================================================
# 데이터 클래스
# ============================================================


@dataclass
class Evidence:
    """근거 인용."""

    article: str  # "제1조", "제 1 조"
    quote: str  # 원문 인용 (50자 이내)


@dataclass
class QAResult:
    """QA 결과."""

    scope_id: str
    scope_title: str
    question: str
    answer: str
    decision: str  # 보장|비보장|조건부|근거부족
    conditions: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    exclusions_checked: bool = False
    exclusions_found: List[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "scope_id": self.scope_id,
            "scope_title": self.scope_title,
            "question": self.question,
            "answer": self.answer,
            "decision": self.decision,
            "conditions": self.conditions,
            "evidence": [{"article": e.article, "quote": e.quote} for e in self.evidence],
            "exclusions_checked": self.exclusions_checked,
            "exclusions_found": self.exclusions_found,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict, scope_id: str = "") -> "QAResult":
        return cls(
            scope_id=scope_id or data.get("scope_id", ""),
            scope_title=data.get("scope_title", ""),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            decision=data.get("decision", "근거부족"),
            conditions=data.get("conditions", []),
            evidence=[
                Evidence(article=e.get("article", ""), quote=e.get("quote", "")) for e in data.get("evidence", [])
            ],
            exclusions_checked=data.get("exclusions_checked", False),
            exclusions_found=data.get("exclusions_found", []),
            confidence=data.get("confidence", 0.0),
            warnings=data.get("warnings", []),
        )


# ============================================================
# 프롬프트 생성기
# ============================================================


class QAPromptBuilder:
    """QA 프롬프트 생성기."""

    def __init__(self, max_scope_chars: int = 12000):
        """
        Args:
            max_scope_chars: 스코프 텍스트 최대 문자 수
        """
        self.max_scope_chars = max_scope_chars

    def build_messages(
        self,
        scope_title: str,
        scope_text: str,
        question: str,
    ) -> List[Dict[str, str]]:
        """OpenAI 호환 메시지 리스트 생성.

        Args:
            scope_title: 특약/담보명
            scope_text: 특약/담보 원문
            question: 사용자 질문

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        # 스코프 텍스트 길이 제한
        if len(scope_text) > self.max_scope_chars:
            scope_text = scope_text[: self.max_scope_chars]
            scope_text += "\n\n[... 텍스트가 잘렸습니다. 전체 내용은 원문을 확인하세요.]"

        user_content = USER_PROMPT_TEMPLATE.format(
            scope_title=scope_title,
            scope_text=scope_text,
            question=question,
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def parse_response(
        self,
        response_text: str,
        scope_id: str,
    ) -> QAResult:
        """LLM 응답을 QAResult로 파싱.

        Args:
            response_text: LLM 응답 텍스트 (JSON)
            scope_id: 스코프 ID

        Returns:
            QAResult
        """
        try:
            # JSON 추출 (```json ... ``` 형식 대응)
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```json"):
                        in_json = True
                        continue
                    if line.startswith("```"):
                        in_json = False
                        continue
                    if in_json:
                        json_lines.append(line)
                text = "\n".join(json_lines)

            data = json.loads(text)
            result = QAResult.from_dict(data, scope_id)
            result.raw_response = response_text
            return result

        except json.JSONDecodeError as e:
            return QAResult(
                scope_id=scope_id,
                scope_title="",
                question="",
                answer="",
                decision="근거부족",
                confidence=0.0,
                warnings=[f"JSON 파싱 실패: {e}"],
                raw_response=response_text,
            )


# ============================================================
# 요약 프롬프트
# ============================================================

SUMMARY_SYSTEM_PROMPT = """역할: 너는 보험 약관을 요약하는 엔진이다.

규칙(반드시 준수):
1) 아래 <SCOPE>에 제공된 특약/담보 텍스트만 근거로 요약한다.
2) <SCOPE> 밖의 지식을 사용하지 않는다.
3) 핵심 보장, 조건, 면책, 한도를 구분하여 정리한다.

출력 형식(JSON만):
{
  "scope_title": "특약/담보명",
  "what_it_covers": ["보장 내용 목록"],
  "key_conditions": ["가입/지급 조건 목록"],
  "key_exclusions": ["면책/미지급 사유 목록"],
  "limits": ["보장 한도/제한 사항"],
  "important_articles": ["중요 조항 번호"]
}"""

SUMMARY_USER_TEMPLATE = """<SCOPE_TITLE>
{scope_title}
</SCOPE_TITLE>

<SCOPE>
{scope_text}
</SCOPE>

위 <SCOPE>를 요약하세요. JSON 형식으로만 출력하세요."""


@dataclass
class ScopeSummary:
    """스코프 요약."""

    scope_id: str
    scope_title: str
    what_it_covers: List[str] = field(default_factory=list)
    key_conditions: List[str] = field(default_factory=list)
    key_exclusions: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    important_articles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "scope_id": self.scope_id,
            "scope_title": self.scope_title,
            "what_it_covers": self.what_it_covers,
            "key_conditions": self.key_conditions,
            "key_exclusions": self.key_exclusions,
            "limits": self.limits,
            "important_articles": self.important_articles,
        }


def build_summary_messages(
    scope_title: str,
    scope_text: str,
    max_chars: int = 12000,
) -> List[Dict[str, str]]:
    """요약 프롬프트 메시지 생성."""
    if len(scope_text) > max_chars:
        scope_text = scope_text[:max_chars]
        scope_text += "\n\n[... 텍스트가 잘렸습니다.]"

    user_content = SUMMARY_USER_TEMPLATE.format(
        scope_title=scope_title,
        scope_text=scope_text,
    )

    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
