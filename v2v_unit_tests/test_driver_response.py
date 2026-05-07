"""driver_response: 반응 지연·브레이크 강도 범위 샘플 (CARLA 없이 검증)."""
from __future__ import annotations

import random

import pytest

from driver_response import (
    brake_pedal_for_tier,
    brake_pedal_sample,
    consume_due_schedules,
    reaction_delay_sample,
    schedule_brake_from_packet,
    timeline_fire_times,
    timeline_fire_times_fixed_delay,
)


def test_critical_brake_sample_inside_range():
    rng = random.Random(7)
    for _ in range(80):
        p = brake_pedal_sample("critical", 0.8, 1.0, 0.2, 0.5, rng)
        assert 0.8 <= p <= 1.0


def test_warning_brake_sample_inside_range():
    rng = random.Random(3)
    for _ in range(80):
        p = brake_pedal_sample("warning", 0.8, 1.0, 0.25, 0.55, rng)
        assert 0.25 <= p <= 0.55


def test_default_config_brake_critical_stronger_than_warning_band():
    """기본 설정: 긴급 브레이크 하한 > 주의 브레이크 상한."""
    import config

    assert config.BRAKE_CRITICAL_MIN > config.BRAKE_WARNING_MAX


def test_critical_reaction_sample_inside_range():
    rng = random.Random(42)
    for _ in range(50):
        d = reaction_delay_sample("critical", 0.2, 0.4, 0.6, 1.0, rng)
        assert 0.2 <= d <= 0.4


def test_warning_reaction_sample_inside_range():
    rng = random.Random(1)
    for _ in range(50):
        d = reaction_delay_sample("warning", 0.2, 0.4, 0.55, 0.95, rng)
        assert 0.55 <= d <= 0.95


def test_default_config_reaction_ranges_non_overlapping():
    import config

    assert config.REACTION_CRITICAL_MAX_S < config.REACTION_WARNING_MIN_S


def test_brake_pedal_scalar_tier_order():
    assert brake_pedal_for_tier("critical", 0.9, 0.4) > brake_pedal_for_tier(
        "warning", 0.9, 0.4
    )


def test_schedule_from_packet_uniform_delay_and_brake():
    """지연·브레이크 모두 점(min=max)이면 고정값."""
    pkt = {"meta": {"tier": "critical"}, "risk_type": "pedestrian", "ttc": 1.5}
    rng = random.Random(99)
    sch = schedule_brake_from_packet(
        pkt,
        100.0,
        0.0,
        0.35,
        0.35,
        0.75,
        0.75,
        0.88,
        0.88,
        0.40,
        0.40,
        rng,
    )
    assert sch is not None
    assert sch.sampled_reaction_s == pytest.approx(0.35)
    assert sch.sampled_brake_pedal == pytest.approx(0.88)
    assert sch.brake_pedal == sch.sampled_brake_pedal


def test_consume_due():
    from driver_response import ScheduledBrake

    pending = [
        ScheduledBrake(10.0, 0.5, "warning", 0.0, 0.1, 0.5),
        ScheduledBrake(10.5, 0.9, "critical", 0.0, 0.12, 0.9),
    ]
    p, tiers = consume_due_schedules(pending, 11.0)
    assert p == 0.9
    assert set(tiers) == {"warning", "critical"}
    assert len(pending) == 0


def test_timeline_non_overlap_critical_always_first_brake():
    rng = random.Random(123)
    lines = timeline_fire_times(
        [(0.0, "critical"), (0.0, "warning")],
        0.15,
        0.35,
        0.70,
        1.10,
        rng,
    )
    t_crit = next(x for x in lines if x[1] == "critical")[0]
    t_warn = next(x for x in lines if x[1] == "warning")[0]
    assert t_crit < t_warn


def test_fixed_delay_timeline_compat():
    lines = timeline_fire_times_fixed_delay(
        [(0.0, "critical"), (0.0, "warning")], 0.35, 0.75
    )
    assert lines[0][0] < lines[1][0]


def test_info_packet_no_schedule():
    pkt = {"meta": {"tier": "info"}, "ttc": 10.0}
    sch = schedule_brake_from_packet(
        pkt,
        0.0,
        0.0,
        0.2,
        0.4,
        0.5,
        1.0,
        0.7,
        1.0,
        0.3,
        0.6,
        random.Random(0),
    )
    assert sch is None


def test_brake_mean_critical_above_warning_monte_carlo():
    """같은 RNG 스트림 구조에서 구간이 분리되면 긴급 평균 페달이 더 큼."""
    rng_c = random.Random(1000)
    rng_w = random.Random(1000)
    crit = [
        brake_pedal_sample("critical", 0.75, 1.0, 0.2, 0.5, rng_c) for _ in range(500)
    ]
    warn = [
        brake_pedal_sample("warning", 0.75, 1.0, 0.2, 0.45, rng_w) for _ in range(500)
    ]
    assert sum(crit) / len(crit) > sum(warn) / len(warn)
