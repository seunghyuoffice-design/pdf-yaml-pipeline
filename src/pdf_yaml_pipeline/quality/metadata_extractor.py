# SPDX-License-Identifier: MIT
"""메타데이터 추출기

YAML 메타데이터 필드에서 채널 정보를 추출한다.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MetadataResult:
    """메타데이터 추출 결과"""

    product_name: Optional[str]
    version: Optional[str]
    channel_hint: Optional[str]  # 메타데이터에서 직접 발견된 채널
    quality_score: float  # 메타데이터 완전성 점수 (0.0 - 1.0)
    raw_metadata: Dict[str, Any]


class MetadataExtractor:
    """메타데이터 추출기

    YAML 메타데이터 필드에서 채널 힌트와 품질 정보를 추출한다.

    Academy 근거: Dublin Core Metadata Initiative (DCMI, 1995)
    """

    REQUIRED_FIELDS = ["product_name", "version"]
    OPTIONAL_FIELDS = ["channel", "source_file", "created_at", "company"]

    # 버전 패턴 (YYYYMM 또는 YYYY.MM 등)
    VERSION_PATTERN = re.compile(r"(\d{4})[-.]?(\d{2})")

    # 채널 힌트 필드명들
    CHANNEL_FIELDS = ["channel", "sales_channel", "distribution_channel", "채널"]

    def extract(self, yaml_data: Dict[str, Any]) -> MetadataResult:
        """메타데이터 추출

        Args:
            yaml_data: YAML 파일 내용

        Returns:
            메타데이터 추출 결과
        """
        metadata = self._get_metadata_section(yaml_data)

        product_name = self._extract_product_name(yaml_data, metadata)
        version = self._extract_version(yaml_data, metadata)
        channel_hint = self._extract_channel_hint(metadata)
        quality_score = self._calculate_quality(product_name, version, metadata)

        return MetadataResult(
            product_name=product_name,
            version=version,
            channel_hint=channel_hint,
            quality_score=quality_score,
            raw_metadata=metadata,
        )

    def _get_metadata_section(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 섹션 추출

        Args:
            yaml_data: YAML 파일 내용

        Returns:
            메타데이터 딕셔너리
        """
        # metadata 섹션
        if "metadata" in yaml_data and isinstance(yaml_data["metadata"], dict):
            return yaml_data["metadata"]

        # meta 섹션
        if "meta" in yaml_data and isinstance(yaml_data["meta"], dict):
            return yaml_data["meta"]

        # document.metadata 섹션
        if "document" in yaml_data and isinstance(yaml_data["document"], dict):
            doc = yaml_data["document"]
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                return doc["metadata"]

        # 루트 레벨에서 직접 추출
        return {k: v for k, v in yaml_data.items() if k in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS}

    def _extract_product_name(self, yaml_data: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """상품명 추출

        Args:
            yaml_data: YAML 파일 내용
            metadata: 메타데이터 섹션

        Returns:
            상품명 또는 None
        """
        # 메타데이터에서 추출
        for field in ["product_name", "productName", "상품명", "title", "name"]:
            if field in metadata:
                return self._normalize_product_name(str(metadata[field]))

        # document.title에서 추출
        if "document" in yaml_data and isinstance(yaml_data["document"], dict):
            if "title" in yaml_data["document"]:
                return self._normalize_product_name(str(yaml_data["document"]["title"]))

        return None

    def _normalize_product_name(self, name: str) -> str:
        """상품명 정규화

        Args:
            name: 원본 상품명

        Returns:
            정규화된 상품명
        """
        # 공백 정규화
        name = re.sub(r"\s+", " ", name.strip())
        # 괄호 내용 보존 (채널 정보 포함 가능)
        return name

    def _extract_version(self, yaml_data: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """버전 추출 (YYYYMM 형식)

        Args:
            yaml_data: YAML 파일 내용
            metadata: 메타데이터 섹션

        Returns:
            버전 문자열 또는 None
        """
        # 메타데이터에서 추출
        for field in ["version", "버전", "effective_date", "date"]:
            if field in metadata:
                match = self.VERSION_PATTERN.search(str(metadata[field]))
                if match:
                    return f"{match.group(1)}{match.group(2)}"

        # 파일명에서 추출 시도
        if "source_file" in metadata:
            match = self.VERSION_PATTERN.search(str(metadata["source_file"]))
            if match:
                return f"{match.group(1)}{match.group(2)}"

        return None

    def _extract_channel_hint(self, metadata: Dict[str, Any]) -> Optional[str]:
        """채널 힌트 추출

        Args:
            metadata: 메타데이터 섹션

        Returns:
            채널 힌트 또는 None
        """
        for field in self.CHANNEL_FIELDS:
            if field in metadata and metadata[field]:
                return str(metadata[field])
        return None

    def _calculate_quality(
        self,
        product_name: Optional[str],
        version: Optional[str],
        metadata: Dict[str, Any],
    ) -> float:
        """메타데이터 품질 점수 계산

        Args:
            product_name: 상품명
            version: 버전
            metadata: 메타데이터 섹션

        Returns:
            품질 점수 (0.0 - 1.0)
        """
        score = 0.0

        # 필수 필드
        if product_name:
            score += 0.4
        if version:
            score += 0.3

        # 선택 필드
        optional_count = sum(1 for field in self.OPTIONAL_FIELDS if field in metadata and metadata[field])
        score += min(optional_count * 0.1, 0.3)

        return min(score, 1.0)
