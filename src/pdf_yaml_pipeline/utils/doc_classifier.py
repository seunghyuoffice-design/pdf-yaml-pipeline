"""Document classification helpers."""

from __future__ import annotations

import re

_SOURCE_RULES = [
    (re.compile(r"(약관|terms)", re.I), "terms_and_conditions"),
    (re.compile(r"(사업방법서|business)", re.I), "business_method"),
    (re.compile(r"(요약서|summary)", re.I), "product_summary"),
    (re.compile(r"(금감원|fss|보도자료|press)", re.I), "fss_release"),
    (re.compile(r"(판례|precedent|판결)", re.I), "court_precedent"),
]

_CONTENT_RULES = [
    (re.compile(r"(보험.*약관|약관.*보험)", re.I), "terms_and_conditions"),
    (re.compile(r"(보험료.*산출|산출.*보험료)", re.I), "business_method"),
    (re.compile(r"(청구.*절차|절차.*청구)", re.I), "claims_process"),
]


def classify_document_label(source_path: str, content: str | None = None) -> str:
    """Classify document into a label string used by converters."""
    source = source_path or ""
    content = content or ""

    for pattern, label in _SOURCE_RULES:
        if pattern.search(source):
            return label

    for pattern, label in _CONTENT_RULES:
        if pattern.search(content):
            return label

    return "general"


__all__ = ["classify_document_label"]
