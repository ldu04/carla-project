"""급정거 감지기 — TTC 변화율·bbox 면적 변화율 (CARLA 없이)."""
from __future__ import annotations

import pytest

from brake_detector import (
    EmergencyBrakeDetector,
    bbox_area_relative_growth_rate,
    sliding_window_dttc_dt,
)


def test_sliding_window_dttc_dt_matches_linear_slope():
    """등간격 dt에서 선형 TTC 감소 시 dTTC/dt = (마지막-첫) / ((n-1)*dt)."""
    dt = 0.1
    # TTC가 매 스텝 0.2초씩 감소 → 5점이면 4스텝, 총 -0.8초
    series = [5.0, 4.8, 4.6, 4.4, 4.2]
    d = sliding_window_dttc_dt(series, dt)
    expected = (4.2 - 5.0) / (4 * dt)  # -2.0 s/s
    assert d == pytest.approx(expected)


def test_sliding_window_dttc_dt_short_series_returns_zero():
    assert sliding_window_dttc_dt([3.0], 0.05) == 0.0
    assert sliding_window_dttc_dt([], 0.05) == 0.0
    assert sliding_window_dttc_dt([1.0, 2.0], 0.0) == 0.0


def test_bbox_area_relative_growth_rate_threshold_crossing():
    assert bbox_area_relative_growth_rate(115.0, 100.0) == pytest.approx(0.15)
    assert bbox_area_relative_growth_rate(100.0, 100.0) == pytest.approx(0.0)
    assert bbox_area_relative_growth_rate(50.0, 0.0) == 0.0  # 이전 면적 근사 0


def test_detector_method_a_fires_when_dttc_dt_below_threshold():
    dt = 0.1
    win = 5
    # threshold -2.0: TTC가 빠르게 떨어져야 함 — 스텝당 -0.6 이상 (5점 윈도)
    # series 10,9.4,8.8,8.2,7.6 → 변화 -2.4 / (4*0.1) = -6.0
    det = EmergencyBrakeDetector(
        window_frames=win,
        dttc_dt_threshold=-2.0,
        bbox_area_rate_threshold=999.0,  # B 비활성
        infer_dt_s=dt,
        cooldown_s=0.0,
    )
    t = 0.0
    sig = None
    for ttc in [10.0, 9.4, 8.8, 8.2, 7.6]:
        t += 0.01
        sig = det.update_and_evaluate(ttc, None, t)
    assert sig is not None
    assert "A" in sig.methods


def test_detector_method_b_fires_when_bbox_growth_exceeds_threshold():
    dt = 0.05
    det = EmergencyBrakeDetector(
        window_frames=5,
        dttc_dt_threshold=-1e9,  # A 사실상 비활성 (버퍼 채워도 못 넘음)
        bbox_area_rate_threshold=0.15,
        infer_dt_s=dt,
        cooldown_s=0.0,
    )
    t = 0.0
    sig = None
    # TTC는 충분히 크게 유지 (A 비활성)
    for i, area in enumerate([100.0, 100.0, 130.0]):
        t += 0.01
        sig = det.update_and_evaluate(50.0, area, t)
    assert sig is not None
    assert "B" in sig.methods
    assert sig.bbox_area_rate == pytest.approx(0.3)


def test_detector_cooldown_suppresses_repeat():
    det = EmergencyBrakeDetector(
        window_frames=3,
        dttc_dt_threshold=-1.0,
        bbox_area_rate_threshold=999.0,
        infer_dt_s=0.1,
        cooldown_s=10.0,
    )
    series = [5.0, 4.5, 4.0]
    t = 0.0
    first = None
    for x in series:
        t += 0.01
        first = det.update_and_evaluate(x, None, t)
    assert first is not None
    second = det.update_and_evaluate(3.5, None, t + 0.01)
    assert second is None
