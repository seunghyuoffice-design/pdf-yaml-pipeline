"""
4단계 파이프라인: Generate → Critique → Filter → Pack
Dyarchy 로컬 LLM 통합을 위한 최적화된 아키텍처
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

# 스키마 임포트
try:
    from ..schemas.document import DocumentMetadata
    from ..schemas.training import ContextItem
except ImportError:
    # 로컬 테스트용 기본 클래스
    class ContextItem(BaseModel):
        content: str
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class DocumentMetadata(BaseModel):
        source: str
        page: int
        coordinates: Optional[Dict[str, Any]] = None


class TaskType(str, Enum):
    """작업 유형"""

    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    TRANSLATE = "translate"
    QA = "qa"
    REASONING = "reasoning"


class DomainType(str, Enum):
    """도메인 유형"""

    INSURANCE = "insurance"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    GENERAL = "general"


class KoreanConstraint(str, Enum):
    """한국어 제약 조건"""

    PII_FILTER = "pii_filter"  # 개인정보 필터링
    FORMAL_SPEECH = "formal_speech"  # 격식체 사용
    HONORIFICS = "honorifics"  # 존댓말 적절성
    TERMINOLOGY = "terminology"  # 전문용어 통일
    NO_SLUR = "no_slur"  # 비속어 금지
    LENGTH_LIMIT = "length_limit"  # 길이 제한


@dataclass
class PipelineConfig:
    """파이프라인 설정"""

    max_tokens_input: int = 8000
    max_tokens_output: int = 2048
    temperature: float = 0.7
    max_retries: int = 2
    batch_size: int = 50
    enable_korean_checks: bool = True
    korean_constraints: List[KoreanConstraint] = None

    def __post_init__(self):
        if self.korean_constraints is None:
            self.korean_constraints = [
                KoreanConstraint.PII_FILTER,
                KoreanConstraint.FORMAL_SPEECH,
                KoreanConstraint.HONORIFICS,
                KoreanConstraint.TERMINOLOGY,
                KoreanConstraint.NO_SLUR,
                KoreanConstraint.LENGTH_LIMIT,
            ]


# Backward-compat alias to distinguish from conversion pipeline config.
TrainingPipelineConfig = PipelineConfig


class GeneratedExample(BaseModel):
    """생성된 예제"""

    id: str = Field(..., description="고유 ID")
    task_type: TaskType = Field(..., description="작업 유형")
    domain_type: DomainType = Field(..., description="도메인 유형")

    # 입력
    instruction: str = Field(..., description="지시문")
    input_context: List[ContextItem] = Field(..., description="입력 컨텍스트")

    # 출력
    output: str = Field(..., description="생성된 응답")
    thinking: Optional[str] = Field(None, description="생각 과정")

    # 메타데이터
    model_name: str = Field(..., description="사용된 모델")
    temperature: float = Field(..., description="사용된 온도")
    tokens_used: int = Field(..., description="사용된 토큰 수")
    generation_time: float = Field(..., description="생성 시간")

    # 품질 평가
    quality_score: Optional[float] = Field(None, description="품질 점수")
    passed_korean_checks: bool = Field(True, description="한국어 검사 통과 여부")
    constraint_violations: List[str] = Field(default_factory=list, description="위반된 제약조건")

    # 생성 정보
    created_at: str = Field(..., description="생성 시간")
    batch_id: str = Field(..., description="배치 ID")
    hash: str = Field(..., description="해시값")


class CritiqueResult(BaseModel):
    """비평 결과"""

    example_id: str = Field(..., description="예제 ID")

    # 품질 평가 (1-10 점수)
    coherence: float = Field(..., description="일관성 (1-10)")
    accuracy: float = Field(..., description="정확성 (1-10)")
    completeness: float = Field(..., description="완전성 (1-10)")
    korean_quality: float = Field(..., description="한국어 품질 (1-10)")

    # 평가 요인
    reasoning_quality: Optional[float] = Field(None, description="추론 품질")
    domain_knowledge: Optional[float] = Field(None, description="도메인 지식")
    clarity: float = Field(..., description="명확성 (1-10)")

    # 통과 여부
    overall_score: float = Field(..., description="종합 점수 (1-10)")
    passed: bool = Field(..., description="통과 여부 (7점 이상)")
    reasoning: str = Field(..., description="평가 이유")

    # 개선 제안
    suggestions: List[str] = Field(default_factory=list, description="개선 제안")

    # 평가 정보
    critic_model: str = Field(..., description="비평 모델")
    evaluation_time: float = Field(..., description="평가 시간")
    created_at: str = Field(..., description="평가 시간")


class FilterCriteria(BaseModel):
    """필터링 기준"""

    min_overall_score: float = Field(7.0, description="최소 종합 점수")
    min_korean_quality: float = Field(6.0, description="최소 한국어 품질 점수")
    max_constraint_violations: int = Field(2, description="최대 제약조건 위반 수")
    must_pass_korean_checks: bool = Field(True, description="한국어 검사 통과 필수")
    allow_duplicate_content: bool = Field(False, description="중복 내용 허용 여부")


class KoreanQualityChecker:
    """한국어 품질 검사기"""

    def __init__(self):
        # PII 패턴 (간단화된 예시)
        self.pii_patterns = [
            r"\d{2,4}[-]\d{2,4}[-]\d{2,4}",  # 생년월일
            r"\d{3}-\d{2}-\d{4}",  # 주민번호
            r"\d{2,3}-\d{3,4}-\d{4}",  # 사업자등록번호
            r"01[016]-\d{3,4}-\d{7}",  # 휴대폰
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # 이메일
        ]

        # 비속어 목록 (간단화된 예시)
        self.slur_words = [
            "씨발",
            "개새끼",
            "미친",
            "병신",
            "존나게",
            "놈",
            "좆",
            "미친놈",
            "쌉년",
            "한심",
        ]

        # 격식체 표현
        self.formal_markers = ["~입니다", "~니다", "~해야 합니다", "~하십시오"]

        # 존댓말 검증을 위한 기본 존칭
        self.honorifics = ["님", "씨", "선생님", "교수님", "박사님"]

    def check_pii(self, text: str) -> bool:
        """개인정보 포함 여부 검사"""
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                return True
        return False

    def check_slur(self, text: str) -> bool:
        """비속어 포함 여부 검사"""
        return any(slur in text for slur in self.slur_words)

    def check_formality(self, text: str) -> Tuple[bool, str]:
        """격식체 사용 여부 검사"""
        has_formal = any(marker in text for marker in self.formal_markers)
        if has_formal:
            return True, "적절한 격식체 사용"
        else:
            return False, "격식체 사용 필요"

    def check_honorifics(self, text: str) -> Tuple[bool, str]:
        """존댓말 적절성 검사"""
        # 간단화된 검사 - 실제로는 더 복잡한 문맥 분석 필요
        context_words = text.split()
        has_honorific = any(honorific in context_words for honorific in self.honorifics)

        # 비즈니스 컨텍스트에서는 존댓말이 적절할 수 있음
        if has_honorific:
            return True, "존댓말 적절히 사용됨"

        # 비즈니스 컨텍스트가 아닌데 존댓말이 없으면 부적절할 수 있음
        if "질문" in text or "문의" in text:
            return False, "질문/문의 맥락에서 존댓말 사용 필요"

        return True, "존댓말 적절함"

    def check_length(self, text: str, max_length: int = 5000) -> Tuple[bool, str]:
        """길이 제한 검사"""
        if len(text) > max_length:
            return False, f"텍스트 길이 초과 ({len(text)} > {max_length})"
        return True, "적절한 길이"

    def check_all_constraints(
        self, text: str, constraints: List[KoreanConstraint], config: PipelineConfig
    ) -> Dict[str, Any]:
        """모든 제약조건 검사"""
        results = {"passed": True, "violations": [], "suggestions": []}

        for constraint in constraints:
            if constraint == KoreanConstraint.PII_FILTER:
                has_pii = self.check_pii(text)
                if has_pii:
                    results["passed"] = False
                    results["violations"].append("PII 정보 포함")
                    results["suggestions"].append("개인정보를 마스킹하거나 제거하세요")

            elif constraint == KoreanConstraint.NO_SLUR:
                has_slur = self.check_slur(text)
                if has_slur:
                    results["passed"] = False
                    results["violations"].append("부적절한 표현 포함")
                    results["suggestions"].append("비속어를 제거하고 적절한 표현으로 수정하세요")

            elif constraint == KoreanConstraint.FORMAL_SPEECH:
                is_formal, message = self.check_formality(text)
                if not is_formal:
                    results["passed"] = False
                    results["violations"].append("비격식체")
                    results["suggestions"].append(message)

            elif constraint == KoreanConstraint.HONORIFICS:
                is_appropriate, message = self.check_honorifics(text)
                if not is_appropriate:
                    results["violations"].append("존댓말 부적절")
                    results["suggestions"].append(message)

            elif constraint == KoreanConstraint.LENGTH_LIMIT:
                is_appropriate, message = self.check_length(text, config.max_tokens_output * 4)
                if not is_appropriate:
                    results["violations"].append("길이 제한 초과")
                    results["suggestions"].append(message)

        return results


class FourStagePipeline:
    """4단계 파이프라인: Generate → Critique → Filter → Pack"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.korean_checker = KoreanQualityChecker()

        # 저장 디렉토리
        self.output_dir = Path("./generated_data")
        self.output_dir.mkdir(exist_ok=True)

        # 진행 상태 추적
        self.current_batch = []
        self.processed_examples = []

        # 중복 검사용 해시 세트
        self._seen_hashes: set = set()

    async def generate_examples(
        self,
        instructions: List[str],
        contexts: List[List[ContextItem]],
        task_type: TaskType,
        domain_type: DomainType,
    ) -> List[GeneratedExample]:
        """1단계: 예제 생성"""
        print(f"📝 1단계: 예제 생성 시작 ({len(instructions)}개)")

        examples = []
        batch_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

        # Dyarchy LLM 호출 (가상 - 실제로는 Dyarchy API 호출)
        for i, (instruction, context) in enumerate(zip(instructions, contexts)):
            start_time = time.time()

            # 실제 구현에서는 여기에 Dyarchy LLM 호출
            # 현재는 모의 생성
            thinking = self._generate_thinking(instruction, context)
            output = self._generate_output(instruction, context, thinking)

            example = GeneratedExample(
                id=f"gen_{batch_id}_{i:03d}",
                task_type=task_type,
                domain_type=domain_type,
                instruction=instruction,
                input_context=context,
                output=output,
                thinking=thinking,
                model_name="local-llm",
                temperature=self.config.temperature,
                tokens_used=len(output.split()),
                generation_time=time.time() - start_time,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                batch_id=batch_id,
                hash=hashlib.md5(output.encode()).hexdigest(),
            )

            # 한국어 검사
            korean_results = self.korean_checker.check_all_constraints(
                output + (thinking or ""), self.config.korean_constraints, self.config
            )

            example.passed_korean_checks = korean_results["passed"]
            example.constraint_violations = korean_results["violations"]

            examples.append(example)

        print(f"✅ 생성 완료: {len(examples)}개 예제")
        return examples

    def _generate_thinking(self, instruction: str, context: List[ContextItem]) -> str:
        """생각 과정 생성 (모의)"""
        context_text = "\n".join([item.content for item in context])

        thinking = f"""생각 과정:
1. 사용자 요청 분석: "{instruction}"
2. 관련 컨텍스트 확인: {len(context)}개 항목 제공됨
3. 핵심 정보 추출: {context_text[:200]}...
4. 논리적 사고: 보험/도메인 지식 적용
5. 구조화된 답변 작성"""

        return thinking

    def _generate_output(self, instruction: str, context: List[ContextItem], thinking: str) -> str:
        """최종 출력 생성 (모의)"""
        context_text = "\n".join([item.content for item in context])

        # 실제 구현에서는 Dyarchy LLM 호출
        output = f"""사용자 요청에 대한 답변입니다.

요청: {instruction}

관련 정보:
{context_text}

분석 결과:
{thinking}

최종 답변:
보험 도메인 전문가로서 정확하고 상세하게 답변해 드립니다.
"""

        return output

    async def critique_examples(self, examples: List[GeneratedExample]) -> List[CritiqueResult]:
        """2단계: 품질 비평"""
        print(f"🔍 2단계: 품질 비평 시작 ({len(examples)}개)")

        critiques = []

        for example in examples:
            time.time()

            # Dyarchy 비평 모델 호출 (가상)
            critique = self._generate_critique(example)

            critiques.append(critique)

        print(f"✅ 비평 완료: {len(critiques)}개 평가")
        return critiques

    def _generate_critique(self, example: GeneratedExample) -> CritiqueResult:
        """품질 비평 생성 (모의)"""
        # 간단화된 평가 로직 (실제로는 더 복잡한 평가 모델 사용)

        # 일관성 평가
        coherence_score = self._evaluate_coherence(example.output)

        # 정확성 평가
        accuracy_score = self._evaluate_accuracy(example.output, example.input_context)

        # 완전성 평가
        completeness_score = self._evaluate_completeness(example.output, example.instruction)

        # 한국어 품질 평가
        korean_quality_score = self._evaluate_korean_quality(example.output)

        # 명확성 평가
        clarity_score = self._evaluate_clarity(example.output)

        # 종합 점수
        overall_score = (
            coherence_score + accuracy_score + completeness_score + korean_quality_score + clarity_score
        ) / 5

        passed = overall_score >= 7.0

        # 평가 이유
        reasoning = self._generate_reasoning(overall_score, passed)

        return CritiqueResult(
            example_id=example.id,
            coherence=coherence_score,
            accuracy=accuracy_score,
            completeness=completeness_score,
            korean_quality=korean_quality_score,
            clarity=clarity_score,
            overall_score=overall_score,
            passed=passed,
            reasoning=reasoning,
            critic_model="critic-model",
            evaluation_time=0.1,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _evaluate_coherence(self, text: str) -> float:
        """일관성 평가 (1-10)"""
        # 간단화된 평가
        if len(text) < 50:
            return 3.0

        sentences = text.split(".")
        if len(sentences) < 2:
            return 5.0

        # 문장 간 연결성
        coherence_indicators = ["그러나", "따라서", "그리고", "또한", "이 때문에"]
        coherence_count = sum(
            1 for sentence in sentences if any(indicator in sentence for indicator in coherence_indicators)
        )

        score = min(10.0, 3.0 + coherence_count * 0.5)
        return score

    def _evaluate_accuracy(self, output: str, context: List[ContextItem]) -> float:
        """정확성 평가 (1-10)"""
        # 간단화된 평가 - 실제로는 도메인 지식 기반 평가 필요
        context_text = " ".join([item.content for item in context])

        # 컨텍스트 관련성 검사
        common_words = set(context_text.split()) & set(output.split())
        coverage = len(common_words) / max(len(set(output.split())), 1)

        score = 4.0 + coverage * 6.0
        return min(10.0, score)

    def _evaluate_completeness(self, output: str, instruction: str) -> float:
        """완전성 평가 (1-10)"""
        # 간단화된 평가
        if len(output) < 100:
            return 3.0

        # 질문에 대한 답변 completeness
        question_words = set(instruction.split())
        answer_words = set(output.split())

        overlap = len(question_words & answer_words)
        completeness = overlap / max(len(question_words), 1)

        score = 3.0 + completeness * 7.0
        return min(10.0, score)

    def _evaluate_korean_quality(self, text: str) -> float:
        """한국어 품질 평가 (1-10)"""
        # 문법적 완성성, 자연스러움 평가
        # 간단화된 평가

        # 기본적인 한국어 구조 확인
        korean_patterns = [
            "은/는",
            "이/가",
            "을/를",
            "의",
            "에",
            "에서",
            "으로",
            "까지",
        ]

        pattern_count = sum(1 for pattern in korean_patterns if pattern.replace("/", "") in text)

        # 문장 부호사용
        punctuation_score = min(2.0, text.count(".") + text.count("!") + text.count("?"))

        score = 4.0 + (pattern_count / len(korean_patterns)) * 2.0 + punctuation_score
        return min(10.0, score)

    def _evaluate_clarity(self, text: str) -> float:
        """명확성 평가 (1-10)"""
        # 간단화된 평가
        if len(text) < 20:
            return 3.0

        # 평균 문장 길이
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return 3.0

        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)

        # 너무 길거나 너무 짧은 문장 패널티
        if avg_sentence_length > 200:
            clarity_penalty = 2.0
        elif avg_sentence_length < 10:
            clarity_penalty = 2.0
        else:
            clarity_penalty = 0.0

        score = 8.0 - clarity_penalty
        return max(1.0, score)

    def _generate_reasoning(self, overall_score: float, passed: bool) -> str:
        """평가 이유 생성"""
        if passed:
            if overall_score >= 9.0:
                return "매우 우수한 품질을 보입니다."
            elif overall_score >= 8.0:
                return "우수한 품질을 보입니다."
            else:
                return "양호한 품질을 보입니다."
        else:
            if overall_score >= 6.0:
                return "일부 개선이 필요합니다."
            elif overall_score >= 4.0:
                return "상당한 개선이 필요합니다."
            else:
                return "전면적인 재작성이 필요합니다."

    async def filter_examples(
        self,
        examples: List[GeneratedExample],
        critiques: List[CritiqueResult],
        criteria: FilterCriteria,
    ) -> Tuple[List[GeneratedExample], List[GeneratedExample]]:
        """3단계: 예제 필터링"""
        print(f"🔎 3단계: 예제 필터링 시작 ({len(examples)}개)")

        passed_examples = []
        failed_examples = []

        for example, critique in zip(examples, critiques):
            # 비평 결과와 예제 매핑
            example.quality_score = critique.overall_score
            example.passed_korean_checks = example.passed_korean_checks

            # 필터링 조건 확인
            passes_filter = self._check_filter_criteria(example, critique, criteria)

            if passes_filter:
                passed_examples.append(example)
            else:
                failed_examples.append(example)

        print(f"✅ 필터링 완료: 통과 {len(passed_examples)}개, 제외 {len(failed_examples)}개")
        return passed_examples, failed_examples

    def _check_filter_criteria(
        self,
        example: GeneratedExample,
        critique: CritiqueResult,
        criteria: FilterCriteria,
    ) -> bool:
        """필터링 조건 확인"""

        # 종합 점수 확인
        if critique.overall_score < criteria.min_overall_score:
            return False

        # 한국어 품질 확인
        if critique.korean_quality < criteria.min_korean_quality:
            return False

        # 한국어 검사 통과 여부 확인
        if criteria.must_pass_korean_checks and not example.passed_korean_checks:
            return False

        # 제약조건 위반 횟수 확인
        if len(example.constraint_violations) > criteria.max_constraint_violations:
            return False

        # 중복 내용 확인 (해시 기반)
        if not criteria.allow_duplicate_content:
            content_hash = hashlib.md5((example.instruction + example.output).encode()).hexdigest()
            if content_hash in self._seen_hashes:
                return False
            self._seen_hashes.add(content_hash)

        return True

    async def pack_examples(self, examples: List[GeneratedExample], format_type: str = "yaml") -> str:
        """4단계: 예제 패킹"""
        print(f"📦 4단계: 예제 패킹 시작 ({len(examples)}개)")

        if format_type.lower() == "yaml":
            return self._pack_yaml(examples)
        elif format_type.lower() == "jsonl":
            return self._pack_jsonl(examples)
        else:
            raise ValueError(f"지원하지 않는 포맷: {format_type}")

    def _pack_yaml(self, examples: List[GeneratedExample]) -> str:
        """YAML 포맷으로 패킹"""
        yaml_data = []

        for example in examples:
            yaml_example = {
                "instruction": example.instruction,
                "input": [ctx.dict() for ctx in example.input_context],
                "output": example.output,
                "metadata": {
                    "task_type": example.task_type,
                    "domain_type": example.domain_type,
                    "model": example.model_name,
                    "quality_score": example.quality_score,
                    "tokens_used": example.tokens_used,
                    "generation_time": example.generation_time,
                    "passed_korean_checks": example.passed_korean_checks,
                    "constraint_violations": example.constraint_violations,
                    "created_at": example.created_at,
                    "batch_id": example.batch_id,
                    "hash": example.hash,
                },
            }

            if example.thinking:
                yaml_example["metadata"]["thinking"] = example.thinking

            yaml_data.append(yaml_example)

        return yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True)

    def _pack_jsonl(self, examples: List[GeneratedExample]) -> str:
        """JSONL 포맷으로 패킹"""
        jsonl_lines = []

        for example in examples:
            jsonl_example = {
                "instruction": example.instruction,
                "input": [ctx.dict() for ctx in example.input_context],
                "output": example.output,
                "metadata": {
                    "task_type": example.task_type,
                    "domain_type": example.domain_type,
                    "model": example.model_name,
                    "quality_score": example.quality_score,
                    "tokens_used": example.tokens_used,
                    "generation_time": example.generation_time,
                    "passed_korean_checks": example.passed_korean_checks,
                    "constraint_violations": example.constraint_violations,
                    "created_at": example.created_at,
                    "batch_id": example.batch_id,
                    "hash": example.hash,
                },
            }

            if example.thinking:
                jsonl_example["metadata"]["thinking"] = example.thinking

            jsonl_lines.append(json.dumps(jsonl_example, ensure_ascii=False))

        return "\n".join(jsonl_lines)

    async def run_full_pipeline(
        self,
        instructions: List[str],
        contexts: List[List[ContextItem]],
        task_type: TaskType,
        domain_type: DomainType,
        filter_criteria: Optional[FilterCriteria] = None,
        output_format: str = "yaml",
    ) -> str:
        """전체 파이프라인 실행"""
        print("🚀 4단계 파이프라인 실행 시작")
        start_time = time.time()

        # 기본 필터링 기준
        if filter_criteria is None:
            filter_criteria = FilterCriteria()

        try:
            # 1단계: 생성
            examples = await self.generate_examples(instructions, contexts, task_type, domain_type)

            # 2단계: 비평
            critiques = await self.critique_examples(examples)

            # 비평 결과를 예제에 매핑
            for example, critique in zip(examples, critiques):
                example.quality_score = critique.overall_score

            # 3단계: 필터링
            passed_examples, failed_examples = await self.filter_examples(examples, critiques, filter_criteria)

            # 4단계: 패킹
            packed_data = await self.pack_examples(passed_examples, output_format)

            # 결과 저장
            output_file = self.output_dir / f"generated_{task_type}_{domain_type}_{int(time.time())}.{output_format}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(packed_data)

            # 통계 정보
            total_time = time.time() - start_time
            stats = {
                "total_instructions": len(instructions),
                "generated_examples": len(examples),
                "critiqued_examples": len(critiques),
                "passed_examples": len(passed_examples),
                "failed_examples": len(failed_examples),
                "pass_rate": len(passed_examples) / len(examples) * 100,
                "total_time": total_time,
                "output_file": str(output_file),
                "filter_criteria": filter_criteria.dict(),
            }

            # 통계 저장
            stats_file = self.output_dir / f"stats_{int(time.time())}.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            print(f"✅ 파이프라인 완료: {len(passed_examples)}개 예제 생성됨")
            print(f"   통과율: {stats['pass_rate']:.1f}%")
            print(f"   출력 파일: {output_file}")
            print(f"   소요 시간: {total_time:.2f}초")

            return packed_data

        except Exception as e:
            print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
            raise


# 사용 예제
async def main():
    """메인 실행 함수"""

    # 파이프라인 설정
    config = PipelineConfig(
        max_tokens_input=8000,
        max_tokens_output=2048,
        temperature=0.7,
        enable_korean_checks=True,
    )

    # 파이프라인 생성
    pipeline = FourStagePipeline(config)

    # 샘플 데이터
    instructions = [
        "자동차 보험의 보상 범위에 대해 설명해주세요.",
        "보험금 청구 절차를 단계별로 안내해주세요.",
        "실손보험과 타실보험의 차이점을 설명해주세요.",
    ]

    # 샘플 컨텍스트 (실제로는 Dyarchy에서 생성)
    contexts = [
        [
            ContextItem(
                content="자동차 보험 약관 제1조 보상 범위에 관한 규정",
                metadata={"source": "insurance_policy", "page": 1},
            ),
            ContextItem(
                content="보상 범위 예시 및 제외 사항",
                metadata={"source": "insurance_policy", "page": 2},
            ),
        ]
        for _ in instructions
    ]

    # 필터링 기준
    filter_criteria = FilterCriteria(
        min_overall_score=7.0,
        min_korean_quality=6.0,
        max_constraint_violations=2,
        must_pass_korean_checks=True,
    )

    # 파이프라인 실행
    try:
        result = await pipeline.run_full_pipeline(
            instructions=instructions,
            contexts=contexts,
            task_type=TaskType.QA,
            domain_type=DomainType.INSURANCE,
            filter_criteria=filter_criteria,
            output_format="yaml",
        )

        print("\n📋 생성된 데이터 미리보기:")
        print(result[:500] + "..." if len(result) > 500 else result)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
