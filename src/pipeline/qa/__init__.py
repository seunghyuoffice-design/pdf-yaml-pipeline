"""QA Pipeline for Insurance Policy Documents.

특약/담보 단위 스코프 고정 QA 시스템.

주요 컴포넌트:
  - QAPromptBuilder: 스코프 고정 프롬프트 생성
  - ScopeValidator: 참조 차단 검증
  - ScopeGuard: 입력 차단 + 출력 검증 통합
  - MergeEngine: 특약별 결과 병합
  - QAPipeline: 전체 흐름 통합

사용 예시:
    from src.pipeline.qa import QAPipeline, qa_from_file

    # 파일에서 QA
    result = qa_from_file(
        file_path=Path("policy.md"),
        question="상해입원일당은 언제 지급되나요?",
        llm_call=my_llm_function,
    )

    # 파이프라인 직접 사용
    pipeline = QAPipeline()
    result = pipeline.process_document(
        doc_id="policy_001",
        doc_text=text,
        question="...",
        llm_call=my_llm_function,
    )
"""

from src.pipeline.qa.qa_prompt import (
    QAPromptBuilder,
    QAResult,
    Evidence,
    ScopeSummary,
    build_summary_messages,
)
from src.pipeline.qa.scope_validator import (
    ScopeValidator,
    ValidationResult,
    CrossScopeRouter,
    validate_qa_result,
)
from src.pipeline.qa.scope_guard import (
    ScopeGuard,
    ScopeGuardConfig,
    BatchScopeGuard,
    BatchQAConfig,
    create_scope_guard,
)
from src.pipeline.qa.merge_engine import (
    Decision,
    ScopeQAResult,
    Conflict,
    MergedResult,
    MergeEngine,
    merge_qa_results,
)
from src.pipeline.qa.qa_pipeline import (
    QAPipeline,
    QAPipelineConfig,
    single_scope_qa,
    qa_from_file,
)
from src.pipeline.qa.config import (
    QAOperationConfig,
    OutputPolicy,
    SchemaPolicy,
    CrossScopePolicy,
    ValidationPolicy,
    MergePolicy,
    DEFAULT_CONFIG,
    validate_output_text,
    validate_decision,
    validate_response_schema,
    print_operation_rules,
)

__all__ = [
    # Prompt
    "QAPromptBuilder",
    "QAResult",
    "Evidence",
    "ScopeSummary",
    "build_summary_messages",
    # Validator
    "ScopeValidator",
    "ValidationResult",
    "CrossScopeRouter",
    "validate_qa_result",
    # Guard
    "ScopeGuard",
    "ScopeGuardConfig",
    "BatchScopeGuard",
    "BatchQAConfig",
    "create_scope_guard",
    # Merge
    "Decision",
    "ScopeQAResult",
    "Conflict",
    "MergedResult",
    "MergeEngine",
    "merge_qa_results",
    # Pipeline
    "QAPipeline",
    "QAPipelineConfig",
    "single_scope_qa",
    "qa_from_file",
    # Config (Operation Policy)
    "QAOperationConfig",
    "OutputPolicy",
    "SchemaPolicy",
    "CrossScopePolicy",
    "ValidationPolicy",
    "MergePolicy",
    "DEFAULT_CONFIG",
    "validate_output_text",
    "validate_decision",
    "validate_response_schema",
    "print_operation_rules",
]
