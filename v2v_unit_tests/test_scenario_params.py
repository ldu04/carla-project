"""시나리오 파라미터 조합·직렬화 (CARLA 서버 불필요)."""
from __future__ import annotations

from dataclasses import asdict

import pytest

from scenario_emergency_brake import ScenarioParams, ScenarioResult


@pytest.mark.parametrize(
    "speed,headway,trigger,v2v",
    [
        (30.0, 20.0, 35.0, True),
        (60.0, 35.0, 50.0, False),
        (45.0, 28.0, 40.0, True),
    ],
)
def test_scenario_params_roundtrip_dict(
    speed: float, headway: float, trigger: float, v2v: bool
) -> None:
    sp = ScenarioParams(
        initial_speed_kmh=speed,
        headway_m=headway,
        emergency_trigger_distance_m=trigger,
        v2v_enabled=v2v,
    )
    d = asdict(sp)
    assert d["initial_speed_kmh"] == speed
    assert d["headway_m"] == headway
    assert d["emergency_trigger_distance_m"] == trigger
    assert d["v2v_enabled"] is v2v


def test_scenario_result_default_and_collision_fields():
    r = ScenarioResult()
    assert r.collision is False
    assert r.delta_v_ms is None
    r.collision = True
    r.delta_v_ms = 4.5
    r.ttc_at_rear_brake_start = 1.2
    d = asdict(r)
    assert d["collision"] is True
    assert d["delta_v_ms"] == pytest.approx(4.5)
    assert d["ttc_at_rear_brake_start"] == pytest.approx(1.2)
