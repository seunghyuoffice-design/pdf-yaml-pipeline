"""특약/담보 청크 파서.

약관 문서에서 특약/담보 단위로 청크를 분할합니다.

병렬화 원칙:
  - 파일 간 병렬화: ✅ N 워커 분담
  - 페이지 단위 병렬화: ❌ 금지 (문맥 손실)
  - 파일 내부 청크: 특약/담보 단위 (의미적 완결 단위)

청크 계층:
  1. 문서 (document)
  2. 섹션 (보통약관 / 특별약관 / 별표)
  3. 특약/담보 블록
  4. 조항 (제N조) - 선택적 2차 분할

헤더 패턴 (AIG 약관 기준):
  - 특약: "X-X. ...특별약관", "○○특약", "○○에 관한 특약"
  - 담보: "○○담보"
  - 조항: "제N조(...)", "제 N 조 (...)"
  - 별표: "【별표N】", "[별표N]"
  - 부칙: "부칙"
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ChunkType(Enum):
    """청크 유형."""

    DOCUMENT = "document"
    SECTION = "section"
    SPECIAL_CLAUSE = "special_clause"  # 특약
    COVERAGE = "coverage"  # 담보
    ARTICLE = "article"  # 조항
    APPENDIX = "appendix"  # 별표
    ADDENDUM = "addendum"  # 부칙
    OTHER = "other"


@dataclass
class Article:
    """조항 정보."""

    number: str  # "제1조", "제 1 조"
    title: Optional[str]  # "(목적)", "(보장내용)"
    start_line: int
    end_line: int
    start_char: int
    end_char: int


@dataclass
class Chunk:
    """청크 단위."""

    chunk_id: str
    chunk_type: ChunkType
    title: str
    scope_path: List[str]  # ["특별약관", "상해입원일당특약"]
    start_line: int
    end_line: int
    text: str
    articles: List[Article] = field(default_factory=list)
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        """토큰 수 추정 (한글 기준 대략 1.5자/토큰)."""
        return len(self.text) // 2


@dataclass
class ParseResult:
    """파싱 결과."""

    doc_id: str
    chunks: List[Chunk]
    metadata: Dict
    warnings: List[str] = field(default_factory=list)


class ChunkParser:
    """특약/담보 청크 파서."""

    # 헤더 패턴 (우선순위 순)
    PATTERNS = {
        # 특별약관 패턴
        "special_numbered": re.compile(r"^(\d+-\d+)\.\s*(.{2,50}(?:특별약관|특약))\s*$"),
        "special_keyword": re.compile(r"^(.{2,40}(?:특약|에\s*관한\s*특약))\s*$"),
        # 담보 패턴
        "coverage": re.compile(r"^(.{2,40}담보)\s*$"),
        # 별표 패턴
        "appendix": re.compile(r"^[【\[](별표\s*\d+)[】\]]\s*(.*)$"),
        "appendix_alt": re.compile(r"^(별표\s*\d+)\s*[\.:\s]\s*(.*)$"),
        # 부칙 패턴
        "addendum": re.compile(r"^(부\s*칙)\s*$"),
        # 조항 패턴
        "article": re.compile(r"^(제\s*\d+\s*조)\s*(\([^)]+\))?\s*$"),
        "article_inline": re.compile(r"(제\s*\d+\s*조)\s*(\([^)]+\))?"),
        # 장/절 패턴
        "chapter": re.compile(r"^(제\s*\d+\s*(?:편|장|절))\s*(.*)$"),
    }

    # 노이즈 패턴 (헤더 후보에서 제외)
    NOISE_PATTERNS = [
        re.compile(r"^\s*-\s*\d+\s*-\s*$"),  # 페이지 번호
        re.compile(r"^\s*페이지\s*\d+\s*$"),
        re.compile(r"^<!--\s*image\s*-->$"),  # 이미지 마커
        re.compile(r"^\s*\d+\s*$"),  # 숫자만
        re.compile(r"^\s*[|─━═]+\s*$"),  # 테이블 구분선
    ]

    # 토큰 임계치 (2차 분할 기준)
    TOKEN_THRESHOLD = 6000

    def __init__(
        self,
        token_threshold: int = 6000,
        min_header_score: int = 3,
    ):
        self.token_threshold = token_threshold
        self.min_header_score = min_header_score

    def parse(self, text: str, doc_id: Optional[str] = None) -> ParseResult:
        """텍스트를 청크로 분할.

        Args:
            text: 입력 텍스트 (Markdown)
            doc_id: 문서 ID (없으면 해시 생성)

        Returns:
            ParseResult
        """
        if not doc_id:
            doc_id = hashlib.md5(text[:1000].encode()).hexdigest()[:12]

        lines = text.split("\n")
        warnings = []

        # 1단계: 헤더 후보 탐지
        headers = self._detect_headers(lines)

        if not headers:
            # 헤더 없으면 전체를 단일 청크로
            warnings.append("No headers detected, treating as single chunk")
            chunk = Chunk(
                chunk_id=f"{doc_id}:doc",
                chunk_type=ChunkType.DOCUMENT,
                title="document",
                scope_path=["document"],
                start_line=0,
                end_line=len(lines) - 1,
                text=text,
            )
            return ParseResult(
                doc_id=doc_id,
                chunks=[chunk],
                metadata={"total_lines": len(lines)},
                warnings=warnings,
            )

        # 2단계: 헤더 기준으로 블록 분할
        chunks = self._split_by_headers(lines, headers, doc_id)

        # 3단계: 긴 청크는 조항 단위로 2차 분할
        final_chunks = []
        for chunk in chunks:
            if chunk.token_estimate > self.token_threshold:
                sub_chunks = self._split_by_articles(chunk, doc_id)
                if len(sub_chunks) > 1:
                    final_chunks.extend(sub_chunks)
                    continue
            final_chunks.append(chunk)

        # 4단계: 검증
        for chunk in final_chunks:
            self._validate_chunk(chunk)

        return ParseResult(
            doc_id=doc_id,
            chunks=final_chunks,
            metadata={
                "total_lines": len(lines),
                "total_chunks": len(final_chunks),
                "headers_detected": len(headers),
            },
            warnings=warnings,
        )

    def parse_file(self, file_path: Path) -> ParseResult:
        """파일에서 파싱."""
        doc_id = file_path.stem
        text = file_path.read_text(encoding="utf-8")
        return self.parse(text, doc_id)

    def _is_noise(self, line: str) -> bool:
        """노이즈 라인 여부."""
        s = line.strip()
        if not s:
            return True
        return any(p.search(s) for p in self.NOISE_PATTERNS)

    def _header_score(self, line: str) -> int:
        """헤더 후보 점수 계산."""
        s = line.strip()

        if not s or self._is_noise(s):
            return -10

        score = 0

        # 긍정 신호
        if s.endswith("특약") or "에 관한 특약" in s:
            score += 4
        if s.endswith("특별약관"):
            score += 4
        if s.endswith("담보"):
            score += 3
        if s.startswith("별표") or "【별표" in s:
            score += 3
        if s == "부칙" or s == "부 칙":
            score += 3
        if re.match(r"^\d+-\d+\.", s):  # 번호 매기기 형식
            score += 2
        if 4 <= len(s) <= 40:
            score += 1
        if s.startswith("##"):  # Markdown 헤더
            score += 1

        # 부정 신호
        special_char_ratio = len(re.findall(r"[【】\[\]()（）]", s)) / max(len(s), 1)
        if special_char_ratio > 0.3:
            score -= 2
        if len(s) < 3:
            score -= 2
        if len(s) > 50:
            score -= 1

        return score

    def _detect_headers(self, lines: List[str]) -> List[Tuple[int, str, ChunkType, int]]:
        """헤더 후보 탐지.

        Returns:
            [(line_index, title, chunk_type, score), ...]
        """
        headers = []

        for i, line in enumerate(lines):
            s = line.strip()

            # Markdown 헤더 제거
            if s.startswith("##"):
                s = s.lstrip("#").strip()

            score = self._header_score(s)
            if score < self.min_header_score:
                continue

            # 패턴 매칭으로 타입 결정
            chunk_type = ChunkType.OTHER
            title = s

            # 특별약관 (번호 형식)
            m = self.PATTERNS["special_numbered"].match(s)
            if m:
                chunk_type = ChunkType.SPECIAL_CLAUSE
                title = m.group(2)
                score += 2

            # 특별약관 (키워드)
            elif self.PATTERNS["special_keyword"].match(s):
                chunk_type = ChunkType.SPECIAL_CLAUSE

            # 담보
            elif self.PATTERNS["coverage"].match(s):
                chunk_type = ChunkType.COVERAGE

            # 별표
            elif self.PATTERNS["appendix"].match(s):
                m = self.PATTERNS["appendix"].match(s)
                chunk_type = ChunkType.APPENDIX
                title = f"{m.group(1)} {m.group(2)}".strip()

            elif self.PATTERNS["appendix_alt"].match(s):
                m = self.PATTERNS["appendix_alt"].match(s)
                chunk_type = ChunkType.APPENDIX
                title = f"{m.group(1)} {m.group(2)}".strip()

            # 부칙
            elif self.PATTERNS["addendum"].match(s):
                chunk_type = ChunkType.ADDENDUM

            # 유효한 타입만 추가
            if chunk_type != ChunkType.OTHER:
                headers.append((i, title, chunk_type, score))

        # 점수순 정렬 후 라인순 정렬
        headers.sort(key=lambda x: (-x[3], x[0]))

        # 중복 제거 (같은 라인에 여러 매칭 시 최고 점수만)
        seen_lines = set()
        unique_headers = []
        for h in headers:
            if h[0] not in seen_lines:
                seen_lines.add(h[0])
                unique_headers.append(h)

        # 라인 순서로 재정렬
        unique_headers.sort(key=lambda x: x[0])

        return unique_headers

    def _split_by_headers(
        self,
        lines: List[str],
        headers: List[Tuple[int, str, ChunkType, int]],
        doc_id: str,
    ) -> List[Chunk]:
        """헤더 기준으로 청크 분할."""
        chunks = []

        for idx, (start_line, title, chunk_type, score) in enumerate(headers):
            # 끝 라인 결정
            if idx + 1 < len(headers):
                end_line = headers[idx + 1][0] - 1
            else:
                end_line = len(lines) - 1

            # 텍스트 추출
            text = "\n".join(lines[start_line : end_line + 1]).strip()

            # 청크 ID 생성
            chunk_id = f"{doc_id}:{chunk_type.value}:{title[:20]}"

            # 스코프 경로
            scope_path = [chunk_type.value, title]

            chunk = Chunk(
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                title=title,
                scope_path=scope_path,
                start_line=start_line,
                end_line=end_line,
                text=text,
                confidence=min(1.0, score / 10),
            )

            # 조항 인덱싱
            chunk.articles = self._index_articles(text, start_line)

            chunks.append(chunk)

        return chunks

    def _index_articles(self, text: str, base_line: int) -> List[Article]:
        """청크 내 조항 인덱싱."""
        articles = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            m = self.PATTERNS["article"].match(line.strip())
            if m:
                number = m.group(1)
                title = m.group(2) if m.group(2) else None

                # 끝 위치는 다음 조항 전까지
                end_line = i
                for j in range(i + 1, len(lines)):
                    if self.PATTERNS["article"].match(lines[j].strip()):
                        end_line = j - 1
                        break
                    end_line = j

                articles.append(
                    Article(
                        number=number,
                        title=title,
                        start_line=base_line + i,
                        end_line=base_line + end_line,
                        start_char=sum(len(lines[k]) + 1 for k in range(i)),
                        end_char=sum(len(lines[k]) + 1 for k in range(end_line + 1)),
                    )
                )

        return articles

    def _split_by_articles(self, chunk: Chunk, doc_id: str) -> List[Chunk]:
        """긴 청크를 조항 단위로 2차 분할."""
        if not chunk.articles:
            return [chunk]

        # 조항 그룹핑 (토큰 임계치 기준)
        groups = []
        current_group = []
        current_tokens = 0

        lines = chunk.text.split("\n")

        for article in chunk.articles:
            # 조항 텍스트
            rel_start = article.start_line - chunk.start_line
            rel_end = article.end_line - chunk.start_line
            article_text = "\n".join(lines[rel_start : rel_end + 1])
            article_tokens = len(article_text) // 2

            if current_tokens + article_tokens > self.token_threshold and current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0

            current_group.append(article)
            current_tokens += article_tokens

        if current_group:
            groups.append(current_group)

        # 그룹이 1개면 분할 불필요
        if len(groups) <= 1:
            return [chunk]

        # 서브 청크 생성
        sub_chunks = []
        for g_idx, group in enumerate(groups):
            first_article = group[0]
            last_article = group[-1]

            rel_start = first_article.start_line - chunk.start_line
            rel_end = last_article.end_line - chunk.start_line
            text = "\n".join(lines[rel_start : rel_end + 1])

            range_str = f"{first_article.number}~{last_article.number}"
            sub_chunk = Chunk(
                chunk_id=f"{chunk.chunk_id}:{g_idx}",
                chunk_type=ChunkType.ARTICLE,
                title=f"{chunk.title} ({range_str})",
                scope_path=chunk.scope_path + [range_str],
                start_line=first_article.start_line,
                end_line=last_article.end_line,
                text=text,
                articles=group,
                confidence=chunk.confidence,
            )
            sub_chunks.append(sub_chunk)

        return sub_chunks

    def _validate_chunk(self, chunk: Chunk):
        """청크 검증."""
        # 경고: 특약인데 조항이 없음
        if chunk.chunk_type == ChunkType.SPECIAL_CLAUSE:
            if not chunk.articles and chunk.token_estimate > 1000:
                chunk.warnings.append("Special clause without articles detected")

        # 경고: 제목이 너무 짧음
        if len(chunk.title) < 3:
            chunk.warnings.append("Title too short, possible noise")

        # 경고: 텍스트가 너무 짧음
        if chunk.token_estimate < 50:
            chunk.warnings.append("Chunk too short, may be incomplete")


def chunk_to_dict(chunk: Chunk) -> Dict:
    """청크를 딕셔너리로 변환."""
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type.value,
        "title": chunk.title,
        "scope_path": chunk.scope_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "articles": [
            {
                "number": a.number,
                "title": a.title,
                "start_line": a.start_line,
                "end_line": a.end_line,
            }
            for a in chunk.articles
        ],
        "token_estimate": chunk.token_estimate,
        "confidence": chunk.confidence,
        "warnings": chunk.warnings,
    }


def parse_result_to_dict(result: ParseResult) -> Dict:
    """ParseResult를 딕셔너리로 변환."""
    return {
        "doc_id": result.doc_id,
        "chunks": [chunk_to_dict(c) for c in result.chunks],
        "metadata": result.metadata,
        "warnings": result.warnings,
    }
