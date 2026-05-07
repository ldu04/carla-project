"""
급정거(emergency brake) 감지: (A) TTC 변화율 슬라이딩 윈도우, (B) 앞차 bbox 면적 변화율.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


def sliding_window_dttc_dt(
    ttc_series: List[float],
    dt_per_step_s: float,
) -> float:
    """
    등간격 dt 가정 시 TTC 변화율 (dTTC/dt, 단위 s/s).
    슬라이딩 윈도우 샘플이 시간 순서대로일 때 (마지막 - 첫) / ((n-1)*dt).
    """
    if len(ttc_series) < 2 or dt_per_step_s <= 0:
        return 0.0
    return (ttc_series[-1] - ttc_series[0]) / (
        (len(ttc_series) - 1) * dt_per_step_s
    )


def bbox_area_relative_growth_rate(area_now: float, area_prev: float) -> float:
    """연속 프레임 면적 상대 증가율 ( (now-prev)/prev )."""
    if area_prev <= 1e-6:
        return 0.0
    return (area_now - area_prev) / area_prev


@dataclass
class EmergencyBrakeSignal:
    """한 번의 급정거 감지 결과."""

    methods: List[str]  # "A" and/or "B"
    ttc: float
    dttc_dt: Optional[float] = None
    bbox_area_rate: Optional[float] = None
    bbox_area: Optional[float] = None


class EmergencyBrakeDetector:
    """
    매 추론 스텝마다 feed 후 evaluate.
    A: 윈도 내 dTTC/dt <= threshold (음수 크면 급격히 위험 증가)
    B: bbox 면적 증가율 >= threshold
    """

    def __init__(
        self,
        window_frames: int,
        dttc_dt_threshold: float,
        bbox_area_rate_threshold: float,
        infer_dt_s: float,
        cooldown_s: float = 0.25,
    ) -> None:
        self._window_frames = max(2, window_frames)
        self._dttc_threshold = dttc_dt_threshold
        self._bbox_rate_threshold = bbox_area_rate_threshold
        self._infer_dt = infer_dt_s
        self._cooldown_s = cooldown_s
        self._ttc_buf: Deque[float] = deque(maxlen=self._window_frames)
        self._prev_area: Optional[float] = None
        self._last_emit_wall: Optional[float] = None

    def reset(self) -> None:
        self._ttc_buf.clear()
        self._prev_area = None
        self._last_emit_wall = None

    def update_and_evaluate(
        self,
        lead_ttc: Optional[float],
        lead_bbox_area: Optional[float],
        wall_time: float,
    ) -> Optional[EmergencyBrakeSignal]:
        methods: List[str] = []
        dttc_dt: Optional[float] = None
        bbox_rate: Optional[float] = None

        if lead_ttc is not None and lead_ttc < 900.0:
            self._ttc_buf.append(float(lead_ttc))

        if (
            len(self._ttc_buf) >= self._window_frames
            and self._infer_dt > 0
        ):
            series = list(self._ttc_buf)
            dttc_dt = sliding_window_dttc_dt(series, self._infer_dt)
            if dttc_dt <= self._dttc_threshold:
                methods.append("A")

        if lead_bbox_area is not None and self._prev_area is not None:
            bbox_rate = bbox_area_relative_growth_rate(
                lead_bbox_area, self._prev_area
            )
            if bbox_rate >= self._bbox_rate_threshold:
                methods.append("B")

        if lead_bbox_area is not None:
            self._prev_area = lead_bbox_area

        if not methods:
            return None

        if (
            self._last_emit_wall is not None
            and wall_time - self._last_emit_wall < self._cooldown_s
        ):
            return None

        ttc_val = (
            float(lead_ttc)
            if lead_ttc is not None
            else float(self._ttc_buf[-1] if self._ttc_buf else 999.0)
        )
        self._last_emit_wall = wall_time
        return EmergencyBrakeSignal(
            methods=methods,
            ttc=ttc_val,
            dttc_dt=dttc_dt,
            bbox_area_rate=bbox_rate,
            bbox_area=float(lead_bbox_area) if lead_bbox_area else None,
        )
