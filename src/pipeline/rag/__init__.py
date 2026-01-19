# SPDX-License-Identifier: MIT
"""RAG (Retrieval-Augmented Generation) module for YAML pipeline."""

from src.pipeline.rag.doc_classifier import classify_document_role, DocumentRole
from src.pipeline.rag.chunk_builder import build_rag_chunks, table_rows_to_sentences
from src.pipeline.rag.faiss_indexer import FaissIndexer
from src.pipeline.rag.retriever import RagRetriever
from src.pipeline.rag.query_classifier import classify_query, QueryType
from src.pipeline.rag.context_assembler import assemble_context
from src.pipeline.rag.prompt_templates import (
    SYSTEM_PROMPT,
    build_prompt,
    format_answer_with_sources,
)

__all__ = [
    # 문서 분류
    "classify_document_role",
    "DocumentRole",
    # 청킹
    "build_rag_chunks",
    "table_rows_to_sentences",
    # 인덱싱
    "FaissIndexer",
    # 검색
    "RagRetriever",
    # 질문 분류
    "classify_query",
    "QueryType",
    # 컨텍스트 조합
    "assemble_context",
    # 프롬프트
    "SYSTEM_PROMPT",
    "build_prompt",
    "format_answer_with_sources",
]
