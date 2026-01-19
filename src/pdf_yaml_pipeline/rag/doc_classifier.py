# SPDX-License-Identifier: MIT
from enum import Enum
from pathlib import Path


class DocumentRole(str, Enum):
    CANONICAL = "canonical"
    PARAPHRASE = "paraphrase"
    OPERATIONAL = "operational"


_RULES = {
    DocumentRole.CANONICAL: ["약관", "보통약관", "특별약관"],
    DocumentRole.PARAPHRASE: ["상품요약서", "요약서"],
    DocumentRole.OPERATIONAL: ["사업방법서"],
}


def classify_document_role(source_path: str) -> DocumentRole:
    name = Path(source_path).name
    for role, keys in _RULES.items():
        if any(k in name for k in keys):
            return role
    return DocumentRole.CANONICAL
