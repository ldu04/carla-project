"""TTC·종방향 거리·미니맵 투영 단위 테스트 (CARLA 없이 실행).

실행 (프로젝트 루트 carla_v2v 에서):
  python -m pytest v2v_unit_tests/test_geometry.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geometry import (
    compute_ttc,
    longitudinal_distance_m,
    project_world_to_minimap,
    risk_tier_from_ttc,
    unproject_pixel_ray_to_world,
)


def test_longitudinal_positive_ahead():
    ego = np.array([0.0, 0.0, 0.0])
    fwd = np.array([1.0, 0.0, 0.0])
    pt = np.array([10.0, 0.0, 0.0])
    assert longitudinal_distance_m(ego, fwd, pt) == pytest.approx(10.0)


def test_longitudinal_behind_negative():
    ego = np.array([0.0, 0.0, 0.0])
    fwd = np.array([1.0, 0.0, 0.0])
    pt = np.array([-5.0, 0.0, 0.0])
    assert longitudinal_distance_m(ego, fwd, pt) < 0


def test_ttc_basic():
    t = compute_ttc(50.0, 10.0, min_speed=1.0)
    assert t == pytest.approx(5.0)


def test_ttc_behind_returns_large():
    t = compute_ttc(-5.0, 10.0)
    assert t >= 900.0


def test_risk_tier():
    assert risk_tier_from_ttc(1.0, 2.0, 4.0) == "critical"
    assert risk_tier_from_ttc(3.0, 2.0, 4.0) == "warning"
    assert risk_tier_from_ttc(10.0, 2.0, 4.0) == "info"


def test_unproject_identity_forward():
    W = np.eye(4)
    u, v = 320.0, 180.0
    depth = 10.0
    p = unproject_pixel_ray_to_world(
        u, v, depth, 640, 360, 90.0, W
    )
    assert p[0] > 5.0


def test_minimap_rotation():
    ego_xy = (100.0, 100.0)
    ego_yaw = math.radians(0.0)
    hz = (110.0, 100.0)
    px, py = project_world_to_minimap(hz, ego_xy, ego_yaw, 2.0, (200, 200))
    assert px > 200


@pytest.mark.parametrize(
    "dist,speed,want_tier",
    [
        (50.0, 20.0, "warning"),
        (10.0, 20.0, "critical"),
    ],
)
def test_ttc_tiers_integration(dist, speed, want_tier):
    ttc = compute_ttc(dist, speed)
    tier = risk_tier_from_ttc(ttc, critical=2.0, warning=4.0)
    assert tier == want_tier
