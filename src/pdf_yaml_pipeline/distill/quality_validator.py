# SPDX-License-Identifier: MIT
"""Distillation 데이터 품질 검증기."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from pdf_yaml_pipeline.distill.quality_rules import (
    Finding,
    validate_canonical_quote,
    validate_cot_leak,
    validate_cross_document,
    validate_grounding_minimal,
    validate_length,
    validate_meta,
    validate_role_policy,
    validate_schema,
    validate_tone_risk,
)


@dataclass
class ValidationResult:
    ok: bool
    findings: List[Finding]

    def has_warnings(self) -> bool:
        return any(f.level == "warn" for f in self.findings)

    def fail_codes(self) -> List[str]:
        return [f.code for f in self.findings if f.level == "fail"]

    def warn_codes(self) -> List[str]:
        return [f.code for f in self.findings if f.level == "warn"]


# Backward-compat alias for distillation validation output.
DistillValidationResult = ValidationResult


def validate_record(rec: Dict[str, Any]) -> ValidationResult:
    """
    레코드 전체 검증.

    Returns:
        ValidationResult with ok=True if no fail-level findings
    """
    findings: List[Finding] = []

    # 순서대로 검증
    findings += validate_schema(rec)
    findings += validate_meta(rec)
    findings += validate_cot_leak(rec)
    findings += validate_length(rec)
    findings += validate_role_policy(rec)
    findings += validate_grounding_minimal(rec)
    findings += validate_canonical_quote(rec)
    findings += validate_tone_risk(rec)
    findings += validate_cross_document(rec)

    ok = all(f.level != "fail" for f in findings)
    return ValidationResult(ok=ok, findings=findings)
