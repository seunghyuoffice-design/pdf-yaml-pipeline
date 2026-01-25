"""Quality gate module.

품질 게이트 판정 및 등급 산출.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import GateResult, QualityGrade, ThresholdConfig


class QualityGate:
    """품질 게이트 판정.

    테이블 품질 메트릭을 평가하고 등급을 산출한다.
    테이블 크기에 따라 동적 임계값을 적용할 수 있다.
    """

    # 테이블 크기 분류 기준
    SIZE_SMALL = 10
    SIZE_MEDIUM = 50

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """초기화.

        Args:
            config: 임계값 설정 (None이면 기본값 사용)
        """
        self.config = config or ThresholdConfig()

    def evaluate(
        self,
        table: dict[str, Any],
        use_dynamic_threshold: bool = True,
    ) -> GateResult:
        """테이블 품질 평가.

        Args:
            table: 테이블 딕셔너리 (cells 리스트 포함)
            use_dynamic_threshold: 동적 임계값 사용 여부

        Returns:
            GateResult: 평가 결과
        """
        cells = table.get("cells", []) or []

        # 빈 테이블 처리
        if not cells:
            return GateResult(
                passed=False,
                grade=QualityGrade.F,
                action="reprocess",
                metrics={
                    "fill_rate": 0.0,
                    "avg_reliability": 0.0,
                    "low_conf_ratio": 1.0,
                    "header_ok": False,
                    "cell_count": 0,
                },
                reasons=["empty_table"],
            )

        # 동적 임계값 적용
        config = self.config
        if use_dynamic_threshold:
            config = self.get_dynamic_config(len(cells))

        # 메트릭 계산
        metrics = self._calculate_metrics(cells, config)

        # 판정
        reasons = []
        if metrics["fill_rate"] < config.min_fill_rate:
            reasons.append(f"low_fill_rate ({metrics['fill_rate']:.2f} < {config.min_fill_rate})")
        if metrics["avg_reliability"] < config.min_avg_reliability:
            reasons.append(f"low_avg_reliability ({metrics['avg_reliability']:.2f} < {config.min_avg_reliability})")
        if metrics["low_conf_ratio"] > config.max_low_conf_ratio:
            reasons.append(f"high_low_conf_ratio ({metrics['low_conf_ratio']:.2f} > {config.max_low_conf_ratio})")
        if config.header_required and not metrics["header_ok"]:
            reasons.append("header_validation_failed")

        passed = len(reasons) == 0
        grade = self._calculate_grade(metrics)
        action = self._determine_action(passed, grade)

        return GateResult(
            passed=passed,
            grade=grade,
            action=action,
            metrics=metrics,
            reasons=reasons,
        )

    def get_dynamic_config(self, cell_count: int) -> ThresholdConfig:
        """테이블 크기에 따른 동적 임계값.

        소형 테이블: 엄격 (단일 오류 영향 큼)
        대형 테이블: 완화 (일부 오류 허용)

        Args:
            cell_count: 셀 개수

        Returns:
            ThresholdConfig: 동적 임계값 설정
        """
        if cell_count < self.SIZE_SMALL:
            # 소형 테이블: 엄격
            return ThresholdConfig(
                min_fill_rate=0.70,
                min_avg_reliability=0.75,
                max_low_conf_ratio=0.20,
                header_required=True,
                low_conf_threshold=0.60,
            )
        elif cell_count < self.SIZE_MEDIUM:
            # 중형 테이블: 기본
            return ThresholdConfig(
                min_fill_rate=0.60,
                min_avg_reliability=0.70,
                max_low_conf_ratio=0.25,
                header_required=True,
                low_conf_threshold=0.60,
            )
        else:
            # 대형 테이블: 완화
            return ThresholdConfig(
                min_fill_rate=0.55,
                min_avg_reliability=0.65,
                max_low_conf_ratio=0.30,
                header_required=True,
                low_conf_threshold=0.55,
            )

    def _calculate_metrics(
        self,
        cells: list[dict[str, Any]],
        config: ThresholdConfig,
    ) -> dict[str, Any]:
        """품질 메트릭 계산.

        Args:
            cells: 셀 리스트
            config: 임계값 설정

        Returns:
            메트릭 딕셔너리
        """
        filled = 0
        low_conf = 0
        reliabilities: list[float] = []

        for cell in cells:
            rel = float(cell.get("reliability") or 0.0)
            reliabilities.append(rel)

            if rel < config.low_conf_threshold:
                low_conf += 1

            has_content = bool((cell.get("text") or "").strip() or (cell.get("ocr_text") or "").strip())
            if has_content:
                filled += 1

        cell_count = len(cells)
        fill_rate = filled / max(1, cell_count)
        avg_reliability = sum(reliabilities) / max(1, len(reliabilities))
        low_conf_ratio = low_conf / max(1, cell_count)

        # 헤더 검증 (row 0)
        header_ok = self._validate_header(cells)

        return {
            "fill_rate": round(fill_rate, 4),
            "avg_reliability": round(avg_reliability, 4),
            "low_conf_ratio": round(low_conf_ratio, 4),
            "header_ok": header_ok,
            "cell_count": cell_count,
            "filled_count": filled,
            "low_conf_count": low_conf,
        }

    def _validate_header(self, cells: list[dict[str, Any]]) -> bool:
        """헤더 행 검증.

        Row 0의 셀들이 유효한 헤더인지 확인.

        Args:
            cells: 셀 리스트

        Returns:
            헤더 유효 여부
        """
        header_cells = [c for c in cells if int(c.get("row", -1)) == 0]

        if not header_cells:
            return False

        # 헤더 채움률
        header_filled = sum(
            1 for c in header_cells if (c.get("text") or "").strip() or (c.get("ocr_text") or "").strip()
        )
        header_fill_rate = header_filled / len(header_cells)

        # 헤더 평균 신뢰도
        header_reliabilities = [float(c.get("reliability") or 0.0) for c in header_cells]
        header_avg_rel = sum(header_reliabilities) / len(header_reliabilities)

        return header_fill_rate >= 0.5 and header_avg_rel >= 0.75

    def _calculate_grade(self, metrics: dict[str, Any]) -> QualityGrade:
        """품질 등급 계산.

        메트릭 조합으로 A-F 등급 산출.

        Args:
            metrics: 품질 메트릭

        Returns:
            QualityGrade: 품질 등급
        """
        fill = metrics["fill_rate"]
        rel = metrics["avg_reliability"]
        low = metrics["low_conf_ratio"]
        header = metrics["header_ok"]

        # 종합 점수 계산 (0-100)
        score = fill * 30 + rel * 40 + (1 - low) * 20 + (10 if header else 0)

        if score >= 95:
            return QualityGrade.A
        elif score >= 85:
            return QualityGrade.B
        elif score >= 70:
            return QualityGrade.C
        elif score >= 50:
            return QualityGrade.D
        else:
            return QualityGrade.F

    def _determine_action(self, passed: bool, grade: QualityGrade) -> str:
        """후속 액션 결정.

        Args:
            passed: 게이트 통과 여부
            grade: 품질 등급

        Returns:
            액션 문자열
        """
        if passed:
            return "index"

        if grade in (QualityGrade.D, QualityGrade.F):
            return "reprocess"

        return "manual_review"


__all__ = ["QualityGate", "ThresholdConfig"]
