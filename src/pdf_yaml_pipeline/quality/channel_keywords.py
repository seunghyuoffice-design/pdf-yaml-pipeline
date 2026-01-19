# SPDX-License-Identifier: MIT
"""채널 키워드 사전 관리

Dyarchy 채널 분류 시스템의 키워드 확장기.
동의어/유의어 사전 관리 및 우선순위 규칙 제공.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List


@dataclass
class ChannelKeywords:
    """채널 키워드 사전

    채널별 키워드, 동의어, 부정 키워드, 우선순위를 관리한다.
    Academy 근거: Query Expansion (Efthimiadis, 1996)
    """

    CHANNELS: Dict[str, Dict] = field(
        default_factory=lambda: {
            "제휴": {
                "primary": ["제휴", "제휴사"],
                "synonyms": ["파트너", "협력사", "제휴상품", "제휴채널", "제휴보험"],
                "negative": ["제휴은행", "제휴카드", "제휴할인"],
                "priority": 1,
            },
            "TM": {
                "primary": ["TM", "텔레마케팅"],
                "synonyms": ["전화판매", "전화영업", "아웃바운드", "인바운드"],
                "negative": ["ATM", "TMS"],
                "priority": 2,
            },
            "홈쇼핑": {
                "primary": ["홈쇼핑"],
                "synonyms": ["TV홈쇼핑", "케이블홈쇼핑", "홈쇼핑채널", "홈쇼핑몰"],
                "negative": [],
                "priority": 3,
            },
            "콜센터": {
                "primary": ["콜센터"],
                "synonyms": ["고객센터", "상담센터", "고객상담", "상담원"],
                "negative": ["콜센터번호"],
                "priority": 4,
            },
            "방송": {
                "primary": ["방송"],
                "synonyms": ["TV", "텔레비전", "방송채널", "라디오"],
                "negative": ["방송통신", "방송법"],
                "priority": 5,
            },
        }
    )

    def get_all_keywords(self, channel: str) -> Set[str]:
        """채널의 모든 키워드 반환 (primary + synonyms)

        Args:
            channel: 채널명

        Returns:
            키워드 집합
        """
        if channel not in self.CHANNELS:
            return set()
        ch = self.CHANNELS[channel]
        return set(ch["primary"]) | set(ch["synonyms"])

    def get_negative_keywords(self, channel: str) -> Set[str]:
        """부정 키워드 반환

        Args:
            channel: 채널명

        Returns:
            부정 키워드 집합
        """
        if channel not in self.CHANNELS:
            return set()
        return set(self.CHANNELS[channel]["negative"])

    def get_priority(self, channel: str) -> int:
        """채널 우선순위 반환 (낮을수록 높은 우선순위)

        Args:
            channel: 채널명

        Returns:
            우선순위 (1-5, 1이 최고)
        """
        if channel not in self.CHANNELS:
            return 99
        return self.CHANNELS[channel]["priority"]

    def get_all_channels(self) -> List[str]:
        """모든 채널명 반환 (우선순위 순)

        Returns:
            채널명 리스트
        """
        return sorted(self.CHANNELS.keys(), key=lambda c: self.get_priority(c))

    def get_primary_keywords(self, channel: str) -> Set[str]:
        """주요 키워드만 반환

        Args:
            channel: 채널명

        Returns:
            주요 키워드 집합
        """
        if channel not in self.CHANNELS:
            return set()
        return set(self.CHANNELS[channel]["primary"])
