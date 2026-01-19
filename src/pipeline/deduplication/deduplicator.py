"""Deduplication system for training materials.

훈련 데이터의 중복을 탐지하고 제거하는 시스템.
Exact match, semantic similarity, structural similarity 등 다양한 중복 탐지 알고리즘 지원.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class DeduplicationConfig:
    """중복 제거 설정."""

    def __init__(
        self,
        min_length: int = 50,
        similarity_threshold: float = 0.85,
        enable_semantic: bool = False,
        enable_structural: bool = True,
        batch_size: int = 1000,
    ):
        self.min_length = min_length
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        self.enable_structural = enable_structural
        self.batch_size = batch_size


class ExactDeduplicator:
    """정확히 일치하는 내용 중복 제거."""

    @staticmethod
    def generate_content_hash(content: str) -> str:
        """내용 해시 생성."""
        # 공백 정규화 후 해시 생성
        normalized = re.sub(r"\s+", " ", content.strip())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def deduplicate(
        self, examples: List[Dict[str, Any]], config: DeduplicationConfig
    ) -> Tuple[List[Dict[str, Any]], int]:
        """정확히 일치하는 내용 중복 제거.

        Args:
            examples: 예제 목록
            config: 중복 제거 설정

        Returns:
            Tuple[중복 제거된 예제 목록, 제거된 수]
        """
        seen_hashes: Set[str] = set()
        deduplicated = []
        removed_count = 0

        for example in examples:
            content = self._extract_content(example)

            if len(content) < config.min_length:
                deduplicated.append(example)
                continue

            content_hash = self.generate_content_hash(content)

            if content_hash in seen_hashes:
                removed_count += 1
                continue

            seen_hashes.add(content_hash)
            deduplicated.append(example)

        logger.info(f"Exact deduplication: removed {removed_count} duplicates")
        return deduplicated, removed_count

    def _extract_content(self, example: Dict[str, Any]) -> str:
        """예제에서 중요 내용 추출."""
        # 여러 필드에서 내용 추출
        content_parts = []

        # output이 우선
        if example.get("output"):
            content_parts.append(example["output"])

        # input 추가
        if example.get("input"):
            content_parts.append(example["input"])

        # instruction 추가
        if example.get("instruction"):
            content_parts.append(example["instruction"])

        return " ".join(content_parts)


class StructuralDeduplicator:
    """구조적 유사성 기반 중복 제거."""

    @staticmethod
    def extract_structure(example: Dict[str, Any]) -> Dict[str, Any]:
        """예제의 구조 정보 추출."""
        structure = {
            "has_input": bool(example.get("input")),
            "has_instruction": bool(example.get("instruction")),
            "output_length": len(example.get("output", "")),
            "input_length": len(example.get("input", "")),
            "instruction_length": len(example.get("instruction", "")),
            "source_type": example.get("source", ""),
        }

        # 키워드 추출
        output = example.get("output", "")
        structure["keywords"] = set(re.findall(r"\b\w+\b", output.lower())[:10])

        return structure

    def calculate_similarity(self, struct1: Dict, struct2: Dict) -> float:
        """구조적 유사도 계산."""
        similarity = 0.0
        weight_sum = 0.0

        # 기본 구조 유사도
        for key in ["has_input", "has_instruction", "source_type"]:
            weight = 0.2
            if struct1[key] == struct2[key]:
                similarity += weight
            weight_sum += weight

        # 길이 유사도
        length_keys = ["output_length", "input_length", "instruction_length"]
        for key in length_keys:
            weight = 0.1
            len1, len2 = struct1[key], struct2[key]
            if len1 == 0 and len2 == 0:
                similarity += weight
            elif len1 > 0 and len2 > 0:
                length_similarity = 1 - abs(len1 - len2) / max(len1, len2)
                similarity += weight * length_similarity
            weight_sum += weight

        # 키워드 유사도
        weight = 0.3
        keywords1, keywords2 = struct1["keywords"], struct2["keywords"]
        if keywords1 and keywords2:
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            jaccard = intersection / union if union > 0 else 0
            similarity += weight * jaccard
        weight_sum += weight

        return similarity / weight_sum if weight_sum > 0 else 0.0

    def deduplicate(
        self, examples: List[Dict[str, Any]], config: DeduplicationConfig
    ) -> Tuple[List[Dict[str, Any]], int]:
        """구조적 유사성 기반 중복 제거.

        Args:
            examples: 예제 목록
            config: 중복 제거 설정

        Returns:
            Tuple[중복 제거된 예제 목록, 제거된 수]
        """
        if not config.enable_structural:
            return examples, 0

        structures = [self.extract_structure(ex) for ex in examples]
        deduplicated = []
        kept_structures: List[Dict[str, Any]] = []
        removed_count = 0

        for i, example in enumerate(examples):
            is_duplicate = False

            for kept_struct in kept_structures:
                similarity = self.calculate_similarity(structures[i], kept_struct)

                if similarity >= config.similarity_threshold:
                    removed_count += 1
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(example)
                kept_structures.append(structures[i])

        logger.info(f"Structural deduplication: removed {removed_count} duplicates")
        return deduplicated, removed_count


class SemanticDeduplicator:
    """의미적 유사성 기반 중복 제거 (간단한 TF-IDF 기반)."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.document_freq: Dict[str, int] = defaultdict(int)

    def build_vocabulary(self, examples: List[Dict[str, Any]]):
        """어휘 집합 구축."""
        doc_count = 0

        for example in examples:
            content = self._extract_content(example)
            if not content:
                continue

            # 토큰화
            tokens = self._tokenize(content)
            unique_tokens = set(tokens)

            # 문서 빈도 증가
            for token in unique_tokens:
                self.document_freq[token] += 1
                self.vocab[token] = self.vocab.get(token, 0) + 1

            doc_count += 1

        # IDF 계산
        self.idf = {}
        for token in self.vocab:
            self.idf[token] = len(examples) / (self.document_freq[token] + 1)

    def _extract_content(self, example: Dict[str, Any]) -> str:
        """예제에서 내용 추출."""
        content_parts = []

        if example.get("output"):
            content_parts.append(example["output"])
        if example.get("input"):
            content_parts.append(example["input"])

        return " ".join(content_parts)

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화."""
        # 간단한 토큰화 (실제로는 더 정교한 방식 사용)
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def calculate_tfidf(self, content: str) -> Dict[str, float]:
        """TF-IDF 벡터 계산."""
        tokens = self._tokenize(content)
        token_counts = defaultdict(int)

        for token in tokens:
            token_counts[token] += 1

        tfidf = {}
        total_tokens = len(tokens)

        for token, count in token_counts.items():
            tf = count / total_tokens
            idf = self.idf.get(token, 1.0)
            tfidf[token] = tf * idf

        return tfidf

    def cosine_similarity(self, tfidf1: Dict[str, float], tfidf2: Dict[str, float]) -> float:
        """코사인 유사도 계산."""
        # 공통 토큰
        common_tokens = set(tfidf1.keys()).intersection(set(tfidf2.keys()))

        if not common_tokens:
            return 0.0

        # 내적 계산
        dot_product = sum(tfidf1[token] * tfidf2[token] for token in common_tokens)

        # 노름 계산
        norm1 = sum(value**2 for value in tfidf1.values()) ** 0.5
        norm2 = sum(value**2 for value in tfidf2.values()) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def deduplicate(
        self, examples: List[Dict[str, Any]], config: DeduplicationConfig
    ) -> Tuple[List[Dict[str, Any]], int]:
        """의미적 유사성 기반 중복 제거.

        Args:
            examples: 예제 목록
            config: 중복 제거 설정

        Returns:
            Tuple[중복 제거된 예제 목록, 제거된 수]
        """
        if not config.enable_semantic:
            return examples, 0

        # 어휘 집합 구축
        self.build_vocabulary(examples)

        # TF-IDF 벡터 계산
        tfidf_vectors = []
        for example in examples:
            content = self._extract_content(example)
            if content:
                tfidf_vectors.append(self.calculate_tfidf(content))
            else:
                tfidf_vectors.append({})

        deduplicated = []
        removed_count = 0

        for i, example in enumerate(examples):
            is_duplicate = False

            for j, kept_example in enumerate(deduplicated):
                similarity = self.cosine_similarity(tfidf_vectors[i], tfidf_vectors[examples.index(kept_example)])

                if similarity >= config.similarity_threshold:
                    removed_count += 1
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(example)

        logger.info(f"Semantic deduplication: removed {removed_count} duplicates")
        return deduplicated, removed_count


class TrainingDeduplicator:
    """훈련 데이터 중복 제거 시스템."""

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.exact_dedup = ExactDeduplicator()
        self.structural_dedup = StructuralDeduplicator()
        self.semantic_dedup = SemanticDeduplicator()

    def deduplicate(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """다단계 중복 제거 수행.

        Args:
            examples: 중복 제거할 예제 목록

        Returns:
            Tuple[최종 예제 목록, 단계별 제거 통계]
        """
        logger.info(f"Starting deduplication for {len(examples)} examples")

        stats = {
            "initial_count": len(examples),
            "exact_removed": 0,
            "structural_removed": 0,
            "semantic_removed": 0,
            "final_count": 0,
        }

        # 1단계: 정확히 일치하는 중복 제거
        deduplicated, exact_removed = self.exact_dedup.deduplicate(examples, self.config)
        stats["exact_removed"] = exact_removed

        # 2단계: 구조적 유사성 기반 중복 제거
        deduplicated, structural_removed = self.structural_dedup.deduplicate(deduplicated, self.config)
        stats["structural_removed"] = structural_removed

        # 3단계: 의미적 유사성 기반 중복 제거 (선택적)
        if self.config.enable_semantic:
            deduplicated, semantic_removed = self.semantic_dedup.deduplicate(deduplicated, self.config)
            stats["semantic_removed"] = semantic_removed

        stats["final_count"] = len(deduplicated)
        stats["total_removed"] = stats["exact_removed"] + stats["structural_removed"] + stats["semantic_removed"]

        logger.info(
            f"Deduplication completed: {stats['initial_count']} -> {stats['final_count']} "
            f"(removed {stats['total_removed']} total)"
        )

        return deduplicated, stats

    def deduplicate_file(self, input_path: Path, output_path: Path) -> Dict[str, int]:
        """파일 중복 제거.

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로

        Returns:
            Dict[str, int]: 중복 제거 통계
        """
        # 파일 읽기
        if input_path.suffix == ".yaml":
            from src.pipeline.converters.yaml_converter import YAMLReader

            examples = YAMLReader.read(input_path)
        elif input_path.suffix == ".jsonl":
            from src.pipeline.converters.jsonl_converter import JSONLReader

            examples = JSONLReader.read(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        # 중복 제거
        deduplicated, stats = self.deduplicate(examples)

        # 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".yaml":
            import yaml

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    deduplicated,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2,
                )
        elif output_path.suffix == ".jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for example in deduplicated:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Deduplicated data saved to {output_path}")
        return stats

    def analyze_duplicates(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """중복 패턴 분석.

        Args:
            examples: 분석할 예제 목록

        Returns:
            Dict[str, Any]: 중복 분석 결과
        """
        analysis = {
            "total_examples": len(examples),
            "source_distribution": defaultdict(int),
            "average_length": 0,
            "length_distribution": defaultdict(int),
            "duplicate_clusters": [],
        }

        # 소스 분포
        for example in examples:
            source = example.get("source", "unknown")
            analysis["source_distribution"][source] += 1

        # 길이 분포
        lengths = []
        for example in examples:
            output_length = len(example.get("output", ""))
            lengths.append(output_length)

            if output_length < 100:
                analysis["length_distribution"]["short"] += 1
            elif output_length < 500:
                analysis["length_distribution"]["medium"] += 1
            else:
                analysis["length_distribution"]["long"] += 1

        analysis["average_length"] = sum(lengths) / len(lengths) if lengths else 0

        # 중복 클러스터 분석 (정확히 일치)
        exact_dedup = ExactDeduplicator()
        seen_hashes = defaultdict(list)

        for i, example in enumerate(examples):
            content = exact_dedup._extract_content(example)
            if len(content) >= self.config.min_length:
                content_hash = exact_dedup.generate_content_hash(content)
                seen_hashes[content_hash].append((i, example))

        # 중복 클러스터 수집
        for content_hash, cluster in seen_hashes.items():
            if len(cluster) > 1:
                analysis["duplicate_clusters"].append(
                    {
                        "hash": content_hash[:8],
                        "count": len(cluster),
                        "sources": [ex[1].get("source", "unknown") for ex in cluster],
                        "indices": [ex[0] for ex in cluster],
                    }
                )

        return analysis


__all__ = [
    "DeduplicationConfig",
    "ExactDeduplicator",
    "StructuralDeduplicator",
    "SemanticDeduplicator",
    "TrainingDeduplicator",
]
