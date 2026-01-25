"""특약/담보 청크 파서 (간결 버전).

약관 문서에서 특약/담보/별표 단위로 청크를 분할합니다.

병렬화 원칙:
  - 파일 간 병렬화: ✅ N 워커 분담
  - 페이지 단위 병렬화: ❌ 금지 (문맥 손실)
  - 파일 내부 청크: 특약/담보 단위 (의미적 완결 단위)

확인된 실제 패턴:
  - 특약 헤더: X-X. ○○ 특별약관, ○○특약
  - 조항: 제1조(...), 제 1 조 (...), ## 제1조 (...)
  - 항: ① ② ③
  - 별표: 【별표1】, 【별표2】
  - 별표/부칙: 문서 공통 (특약 종속 아님)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================
# 정규식 정의
# ============================================================

RE_SPECIAL_CLAUSE = re.compile(r"^\s*(\d+\-\d+\.\s*)?.{2,50}(특약|특별약관)\s*$")

RE_COVERAGE = re.compile(r"^\s*.{2,40}담보\s*$")

RE_ARTICLE = re.compile(r"^\s*(##\s*)?(제\s*\d+\s*조)\s*(\(.+?\))?")

RE_APPENDIX = re.compile(r"^\s*【\s*별표\s*\d+\s*】")

RE_ADDENDUM = re.compile(r"^\s*부\s*칙\s*$")

RE_NOISE = re.compile(r"^\s*(-\s*\d+\s*-|페이지\s*\d+|<!--\s*image\s*-->)\s*$")


# ============================================================
# 데이터 클래스
# ============================================================


@dataclass
class ArticleMeta:
    """조항 메타데이터."""

    article: str  # "제1조", "제 1 조"
    title: str  # "(목적)", "(보장내용)"
    line: int  # 청크 내 상대 라인


@dataclass
class ClauseChunk:
    """특약/담보/별표 청크."""

    chunk_type: str  # special_clause | coverage | appendix | addendum
    title: str  # 특약명 / 담보명 / 별표명
    articles: List[ArticleMeta] = field(default_factory=list)
    text: str = ""
    start_line: int = 0
    end_line: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def article_count(self) -> int:
        return len(self.articles)

    def to_dict(self) -> Dict:
        return {
            "chunk_type": self.chunk_type,
            "title": self.title,
            "articles": [{"article": a.article, "title": a.title, "line": a.line} for a in self.articles],
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "char_count": self.char_count,
            "article_count": self.article_count,
        }


# ============================================================
# 파서 구현
# ============================================================


class SpecialClauseParser:
    """특약/담보/별표 청크 파서."""

    def __init__(self, max_chars: int = 12000):
        """
        Args:
            max_chars: 청크 최대 문자 수 (초과 시 경고)
        """
        self.max_chars = max_chars

    def parse(self, text: str) -> List[ClauseChunk]:
        """텍스트를 청크로 분할.

        Args:
            text: 입력 텍스트 (Markdown)

        Returns:
            ClauseChunk 리스트
        """
        lines = text.splitlines()
        chunks: List[ClauseChunk] = []

        # 헤더 탐지
        special_headers = self._find_special_headers(lines)
        coverage_headers = self._find_coverage_headers(lines)
        appendices = self._find_appendices(lines)
        addenda = self._find_addenda(lines)

        # 모든 헤더 통합 (라인순 정렬)
        all_headers = []
        all_headers.extend([(i, t, "special_clause") for i, t in special_headers])
        all_headers.extend([(i, t, "coverage") for i, t in coverage_headers])
        all_headers.extend([(i, t, "appendix") for i, t in appendices])
        all_headers.extend([(i, t, "addendum") for i, t in addenda])
        all_headers.sort(key=lambda x: x[0])

        # 청크 생성
        for idx, (start, title, chunk_type) in enumerate(all_headers):
            # 끝 라인 결정 (다음 헤더 전까지)
            if idx + 1 < len(all_headers):
                end = all_headers[idx + 1][0] - 1
            else:
                end = len(lines) - 1

            block_lines = lines[start : end + 1]
            block_text = "\n".join(block_lines).strip()

            # 조항 추출 (별표/부칙 제외)
            articles = []
            if chunk_type in ("special_clause", "coverage"):
                articles = self._extract_articles(block_lines)

            chunks.append(
                ClauseChunk(
                    chunk_type=chunk_type,
                    title=title,
                    articles=articles,
                    text=block_text,
                    start_line=start,
                    end_line=end,
                )
            )

        # 헤더가 없으면 전체를 단일 청크로
        if not chunks:
            chunks.append(
                ClauseChunk(
                    chunk_type="document",
                    title="전체문서",
                    articles=self._extract_articles(lines),
                    text=text.strip(),
                    start_line=0,
                    end_line=len(lines) - 1,
                )
            )

        return chunks

    def parse_to_dicts(self, text: str) -> List[Dict]:
        """텍스트를 청크 딕셔너리 리스트로 변환."""
        return [c.to_dict() for c in self.parse(text)]

    # --------------------------------------------------
    # 헤더 탐지
    # --------------------------------------------------

    def _find_special_headers(self, lines: List[str]) -> List[tuple]:
        """특약/특별약관 헤더 탐지."""
        headers = []
        for i, line in enumerate(lines):
            s = line.strip()
            # Markdown 헤더 제거
            if s.startswith("##"):
                s = s.lstrip("#").strip()
            if RE_SPECIAL_CLAUSE.match(s) and not RE_NOISE.match(s):
                headers.append((i, s))
        return headers

    def _find_coverage_headers(self, lines: List[str]) -> List[tuple]:
        """담보 헤더 탐지."""
        headers = []
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("##"):
                s = s.lstrip("#").strip()
            if RE_COVERAGE.match(s) and not RE_NOISE.match(s):
                # "담보" 단독은 제외 (테이블 헤더일 수 있음)
                if len(s) > 3:
                    headers.append((i, s))
        return headers

    def _find_appendices(self, lines: List[str]) -> List[tuple]:
        """별표 헤더 탐지."""
        result = []
        for i, line in enumerate(lines):
            s = line.strip()
            if RE_APPENDIX.match(s):
                result.append((i, s))
        return result

    def _find_addenda(self, lines: List[str]) -> List[tuple]:
        """부칙 헤더 탐지."""
        result = []
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("##"):
                s = s.lstrip("#").strip()
            if RE_ADDENDUM.match(s):
                result.append((i, s))
        return result

    # --------------------------------------------------
    # 조항 추출
    # --------------------------------------------------

    def _extract_articles(self, lines: List[str]) -> List[ArticleMeta]:
        """청크 내 조항 추출."""
        articles = []
        for idx, line in enumerate(lines):
            s = line.strip()
            m = RE_ARTICLE.match(s)
            if m:
                articles.append(
                    ArticleMeta(
                        article=m.group(2),
                        title=m.group(3) or "",
                        line=idx,
                    )
                )
        return articles


# ============================================================
# 유틸리티 함수
# ============================================================


def parse_document(text: str, max_chars: int = 12000) -> List[Dict]:
    """문서를 청크로 파싱 (편의 함수).

    Args:
        text: 입력 텍스트
        max_chars: 청크 최대 문자 수

    Returns:
        청크 딕셔너리 리스트
    """
    parser = SpecialClauseParser(max_chars=max_chars)
    return parser.parse_to_dicts(text)


def validate_chunks(chunks: List[ClauseChunk], max_chars: int = 12000) -> List[str]:
    """청크 검증.

    Returns:
        경고 메시지 리스트
    """
    warnings = []

    for i, chunk in enumerate(chunks):
        # 문자 수 초과
        if chunk.char_count > max_chars:
            warnings.append(f"Chunk {i} '{chunk.title[:20]}' exceeds max_chars: " f"{chunk.char_count} > {max_chars}")

        # 특약인데 조항 없음
        if chunk.chunk_type == "special_clause" and chunk.article_count == 0:
            if chunk.char_count > 500:
                warnings.append(f"Chunk {i} '{chunk.title[:20]}' is special_clause " f"but has no articles")

        # 제목 너무 짧음
        if len(chunk.title) < 3:
            warnings.append(f"Chunk {i} has short title: '{chunk.title}'")

    return warnings
