# SPDX-License-Identifier: MIT
"""
Teacher → Student Distillation 데이터 생성.

핵심 원칙:
- 약관(canonical) 기준 답변만 증류
- Teacher의 최종 답변 + 근거 구조만 전달
- Student는 추론 흉내 X / 판단 결과만 학습
- Chain-of-Thought 제거
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TeacherOutput:
    """Teacher 80B 출력 (학습용)."""

    instruction: str
    context: Dict[str, str]  # canonical, summary, operational
    teacher_answer: Dict[str, Any]  # final, sources

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StudentInput:
    """Student MoE 30B 학습 입력."""

    instruction: str
    context: str  # RAG로 조합된 단일 컨텍스트
    output: str  # 최종 답변만

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_teacher_output(
    question: str,
    canonical_ctx: str,
    summary_ctx: Optional[str],
    operational_ctx: Optional[str],
    final_answer: str,
    used_sources: List[str],
) -> TeacherOutput:
    """Teacher 출력 데이터 생성."""
    context = {
        "canonical": canonical_ctx,
        "summary": summary_ctx or "",
        "operational": operational_ctx or "",
    }

    teacher_answer = {
        "final": final_answer,
        "sources": used_sources,
    }

    return TeacherOutput(
        instruction=question,
        context=context,
        teacher_answer=teacher_answer,
    )


def teacher_to_student(teacher: TeacherOutput) -> StudentInput:
    """
    Teacher 출력 → Student 입력 변환.

    - Chain-of-Thought 제거
    - 컨텍스트 단일화
    - 최종 답변만 유지
    """
    # 사용된 소스만 컨텍스트에 포함
    ctx_parts = []
    sources = teacher.teacher_answer.get("sources", ["canonical"])

    if "canonical" in sources and teacher.context.get("canonical"):
        ctx_parts.append(f"[약관]\n{teacher.context['canonical']}")

    if "summary" in sources and teacher.context.get("summary"):
        ctx_parts.append(f"[상품요약서]\n{teacher.context['summary']}")

    if "operational" in sources and teacher.context.get("operational"):
        ctx_parts.append(f"[사업방법서]\n{teacher.context['operational']}")

    context = "\n\n".join(ctx_parts)
    final = teacher.teacher_answer.get("final", "")

    return StudentInput(
        instruction=teacher.instruction,
        context=context,
        output=final,
    )


def generate_distillation_dataset(
    teacher_outputs: List[TeacherOutput],
    output_path: Path,
    format: str = "jsonl",
) -> Dict[str, int]:
    """
    Distillation 데이터셋 생성.

    Args:
        teacher_outputs: Teacher 출력 리스트
        output_path: 출력 파일 경로
        format: jsonl 또는 json

    Returns:
        통계 딕셔너리
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    student_inputs = [teacher_to_student(t) for t in teacher_outputs]

    if format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for s in student_inputs:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in student_inputs],
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "teacher_count": len(teacher_outputs),
        "student_count": len(student_inputs),
        "output_path": str(output_path),
    }


# MoE Expert 분류 (Router용)
MOE_EXPERT_MAPPING = {
    "definition": 0,  # 약관 정의/보장/면책
    "explanation": 1,  # 쉬운 설명 (요약서 스타일)
    "operation": 2,  # 절차/운영 (사업방법서)
    "multimodal": 3,  # 표/이미지 (VL)
    "router": 4,  # 질문 분류 (Thinking Router)
}


def assign_moe_expert(query_type: str) -> int:
    """질문 유형에 따른 MoE Expert 할당."""
    return MOE_EXPERT_MAPPING.get(query_type, 0)
