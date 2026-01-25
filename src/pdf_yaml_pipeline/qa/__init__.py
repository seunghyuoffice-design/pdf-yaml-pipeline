"""QA Pipeline for Insurance Policy Documents.

특약/담보 단위 스코프 고정 QA 시스템.

주요 컴포넌트:
  - QAPromptBuilder: 스코프 고정 프롬프트 생성
  - ScopeValidator: 참조 차단 검증
  - ScopeGuard: 입력 차단 + 출력 검증 통합
  - MergeEngine: 특약별 결과 병합
  - QAPipeline: 전체 흐름 통합

사용 예시:
    from pdf_yaml_pipeline.qa import QAPipeline, qa_from_file

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

from pdf_yaml_pipeline.qa.config import (
    DEFAULT_CONFIG,
    CrossScopePolicy,
    MergePolicy,
    OutputPolicy,
    QAOperationConfig,
    SchemaPolicy,
    ValidationPolicy,
    print_operation_rules,
    validate_decision,
    validate_output_text,
    validate_response_schema,
)
from pdf_yaml_pipeline.qa.merge_engine import (
    Conflict,
    Decision,
    MergedResult,
    MergeEngine,
    ScopeQAResult,
    merge_qa_results,
)
from pdf_yaml_pipeline.qa.qa_pipeline import (
    QAPipeline,
    QAPipelineConfig,
    qa_from_file,
    single_scope_qa,
)
from pdf_yaml_pipeline.qa.qa_prompt import (
    Evidence,
    QAPromptBuilder,
    QAResult,
    ScopeSummary,
    build_summary_messages,
)
from pdf_yaml_pipeline.qa.scope_guard import (
    BatchQAConfig,
    BatchScopeGuard,
    ScopeGuard,
    ScopeGuardConfig,
    create_scope_guard,
)
from pdf_yaml_pipeline.qa.scope_validator import (
    CrossScopeRouter,
    ScopeValidator,
    ScopeValidationResult,
    ValidationResult,
    validate_qa_result,
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
    "ScopeValidationResult",
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
