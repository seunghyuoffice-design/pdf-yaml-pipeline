# SPDX-License-Identifier: MIT
"""RAG Retriever with clause-aware peek generation.

FAISS 기반 벡터 검색 및 난이도 분류용 rag_peek 생성.
"""

from __future__ import annotations

import builtins
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_TOP_K = 3  # 보험 약관 RAG 최적값 (정확도 우선, 토큰 절약)


# =============================================================================
# RAG Retriever
# =============================================================================

_ALLOWED_PICKLE_BUILTINS = {
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "NoneType",
}


class _SafeUnpickler(pickle.Unpickler):
    """Pickle을 기본 자료형으로 제한."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in _ALLOWED_PICKLE_BUILTINS:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"Disallowed pickle type: {module}.{name}")


def _load_legacy_metadata(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Legacy metadata not found: {path}")
    with path.open("rb") as f:
        data = _SafeUnpickler(f).load()
    if not isinstance(data, list):
        raise ValueError("index_metadata.pkl must be a list")
    return data


class RagRetriever:
    """FAISS 기반 RAG 검색기."""

    def __init__(
        self,
        index_dir: Path,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """Initialize retriever.

        Args:
            index_dir: FAISS 인덱스 디렉토리 경로
            model: SentenceTransformer 모델명
        """
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.metas = self._load_metadata(index_dir)
        self.model = SentenceTransformer(model)

    def _load_metadata(self, index_dir: Path) -> List[Dict[str, Any]]:
        json_path = index_dir / "index_metadata.json"
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("index_metadata.json must be a list")
            return data

        allow_pickle = os.getenv("ALLOW_PICKLE_METADATA", "false").lower() == "true"
        if not allow_pickle:
            raise RuntimeError(
                "index_metadata.json not found and pickle loading is disabled. "
                "Rebuild the index or set ALLOW_PICKLE_METADATA=true to allow legacy pickle."
            )

        return _load_legacy_metadata(index_dir / "index_metadata.pkl")

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """쿼리로 유사 청크 검색.

        Args:
            query: 검색 쿼리
            k: 반환할 상위 결과 수 (기본: 3)

        Returns:
            검색 결과 리스트. 각 항목:
            {
                "score": float,
                "meta": {...}
            }
        """
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(qv, k)

        out: List[Dict[str, Any]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            m = self.metas[i]
            out.append(
                {
                    "score": float(s),
                    "meta": m,
                }
            )
        return out


# =============================================================================
# RAG Peek Builder (for Difficulty Classifier)
# =============================================================================


def build_rag_peek(
    search_results: List[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """검색 결과에서 난이도 분류용 peek 생성.

    난이도 분류기는 조항명(clause_title)만 본다.
    텍스트 본문은 포함하지 않음.

    Args:
        search_results: RagRetriever.search() 결과
        top_k: 상위 N개만 사용

    Returns:
        난이도 분류기 입력용 peek dict:
        {
            "top_chunks": [
                {"role": str, "clause_title": str, "chunk_id": int},
                ...
            ]
        }
    """
    chunks = []
    for result in search_results[:top_k]:
        meta = result.get("meta", {})
        chunks.append(
            {
                "role": meta.get("role"),
                "clause_title": meta.get("clause_title"),
                "chunk_id": meta.get("idx"),
            }
        )

    return {"top_chunks": chunks}


def extract_clause_titles(search_results: List[Dict[str, Any]]) -> List[str]:
    """검색 결과에서 조항명만 추출.

    Args:
        search_results: RagRetriever.search() 결과

    Returns:
        조항명 리스트 (중복 제거, 순서 유지)
    """
    seen = set()
    titles = []
    for result in search_results:
        meta = result.get("meta", {})
        title = meta.get("clause_title")
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


# =============================================================================
# Risk Assessment (Difficulty Scoring)
# =============================================================================


def assess_clause_risk(clause_titles: List[str]) -> Dict[str, Any]:
    """조항명 기반 리스크 평가.

    난이도 분류기에서 사용하는 조항 충돌/면책 탐지.

    Args:
        clause_titles: 조항명 리스트

    Returns:
        {
            "score": int,
            "flags": List[str],
            "tier": "L0" | "L1" | "L2"
        }
    """
    flags: List[str] = []
    score = 0

    joined = " ".join(clause_titles)

    # 면책/지급거절 조항 탐지
    if "지급하지" in joined or "면책" in joined or "제외" in joined:
        flags.append("exclusion_clause_present")
        score += 5

    # 지급사유 + 면책 충돌 쌍 탐지
    has_payment = "지급사유" in joined or "지급" in joined
    has_exclusion = "지급하지" in joined or "면책" in joined
    if has_payment and has_exclusion:
        flags.append("conflict_pair_present")
        score += 5

    # 별표/부표 (복잡한 표 구조)
    if "별표" in joined or "부표" in joined:
        flags.append("appendix_table_present")
        score += 2

    # 부칙 (예외 규정)
    if "부칙" in joined:
        flags.append("addendum_present")
        score += 1

    # 난이도 등급 결정
    if score >= 9:
        tier = "L2"  # 235B Teacher 필수
    elif score >= 5:
        tier = "L1"  # 80B Teacher 또는 조건부 235B
    else:
        tier = "L0"  # Student 단독 처리 가능

    return {
        "score": score,
        "flags": flags,
        "tier": tier,
    }


__all__ = [
    "RagRetriever",
    "build_rag_peek",
    "extract_clause_titles",
    "assess_clause_risk",
    "DEFAULT_TOP_K",
]
