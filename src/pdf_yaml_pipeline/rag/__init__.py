# SPDX-License-Identifier: MIT
"""RAG (Retrieval-Augmented Generation) module for YAML pipeline.

Keep this package importable without optional heavy dependencies (faiss,
sentence-transformers, Pillow). Submodules like chunk_builder are used by parsers
and should not fail just because indexing/search extras are unavailable.
"""

from pdf_yaml_pipeline.rag.chunk_builder import build_rag_chunks, table_rows_to_sentences
from pdf_yaml_pipeline.rag.context_assembler import assemble_context
from pdf_yaml_pipeline.rag.doc_classifier import DocumentRole, classify_document_role
from pdf_yaml_pipeline.rag.prompt_templates import SYSTEM_PROMPT, build_prompt, format_answer_with_sources
from pdf_yaml_pipeline.rag.query_classifier import QueryType, classify_query

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


def __getattr__(name: str):
    if name == "FaissIndexer":
        from pdf_yaml_pipeline.rag.faiss_indexer import FaissIndexer

        return FaissIndexer
    if name == "RagRetriever":
        from pdf_yaml_pipeline.rag.retriever import RagRetriever

        return RagRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
