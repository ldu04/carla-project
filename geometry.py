"""
2D 영상 좌표 ↔ CARLA 월드 좌표 근사, TTC·위험 등급.
데드 레커닝·핀홀 모델 기반 (단위 테스트 가능).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def depth_buffer_to_meters(depth_bgra: np.ndarray) -> np.ndarray:
    """
    CARLA sensor.camera.depth Raw 데이터 (BGRA)를 미터 단위 깊이 맵으로 변환.
    CARLA 0.9.x 공식 문서의 정규화 깊이 디코딩 방식.
    """
    if depth_bgra is None or depth_bgra.size == 0:
        return np.zeros((0,), dtype=np.float32)
    # shape (H, W, 4)
    b = depth_bgra[:, :, 0].astype(np.float64)
    g = depth_bgra[:, :, 1].astype(np.float64)
    r = depth_bgra[:, :, 2].astype(np.float64)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    depth_m = 1000.0 * normalized
    return depth_m.astype(np.float32)


def camera_intrinsic_px(
    image_width: int, image_height: int, fov_deg: float
) -> Tuple[float, float, float, float]:
    """fx, fy, cx, cy (픽셀 단위)."""
    fov_rad = math.radians(fov_deg)
    fx = image_width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx  # 정사각 픽셀 가정
    cx = image_width / 2.0
    cy = image_height / 2.0
    return fx, fy, cx, cy


def unproject_pixel_ray_to_world(
    u: float,
    v: float,
    depth_along_ray_m: float,
    image_width: int,
    image_height: int,
    fov_deg: float,
    world_from_camera_4x4: np.ndarray,
) -> np.ndarray:
    """
    CARLA depth를 광선 방향 거리로 두고, 픽셀을 월드 3D 좌표로 변환.
    OpenCV식 광선(un_project) → UE/CARLA 로컬(X 전방, Y 우, Z 상) → 월드.
    """
    fx, fy, cx, cy = camera_intrinsic_px(image_width, image_height, fov_deg)
    x_nd = (u - cx) / fx
    y_nd = (v - cy) / fy
    dir_cv = np.array([x_nd, y_nd, 1.0], dtype=np.float64)
    dir_cv /= np.linalg.norm(dir_cv)
    pt_cv = dir_cv * float(depth_along_ray_m)
    x_ue = pt_cv[2]
    y_ue = pt_cv[0]
    z_ue = -pt_cv[1]
    local = np.array([x_ue, y_ue, z_ue, 1.0], dtype=np.float64)
    world = world_from_camera_4x4 @ local
    return world[:3].astype(np.float64)


def camera_point_to_world(
    cam_transform_matrix: np.ndarray, point_cam: np.ndarray
) -> np.ndarray:
    """4x4 월드 변환 행렬 * 동차 좌표."""
    if cam_transform_matrix.shape != (4, 4):
        raise ValueError("cam_transform_matrix must be 4x4")
    hom = np.array(
        [point_cam[0], point_cam[1], point_cam[2], 1.0], dtype=np.float64
    )
    world = cam_transform_matrix @ hom
    return world[:3]


def longitudinal_distance_m(
    ego_location: np.ndarray,
    ego_forward_unit: np.ndarray,
    world_point: np.ndarray,
) -> float:
    """월드 점이 송신 차량 전방 축 기준으로 얼마나 앞에 있는지 (+-)."""
    rel = world_point - ego_location
    return float(np.dot(rel, ego_forward_unit))


def closing_speed_ms(
    ego_velocity_world: np.ndarray,
    ego_forward_unit: np.ndarray,
    object_velocity_world: Optional[np.ndarray] = None,
) -> float:
    """
    정면 충돌 위주 근사: 상대 접근 속도 (m/s).
    object_velocity_world 가 None 이면 정지물체로 간주.
    """
    v_e = float(np.dot(ego_velocity_world, ego_forward_unit))
    if object_velocity_world is None:
        return max(v_e, 0.0)
    v_o = float(np.dot(object_velocity_world, ego_forward_unit))
    return max(v_e - v_o, 0.0)


def compute_ttc(
    distance_longitudinal_m: float,
    closing_speed_ms_val: float,
    min_speed: float = 0.5,
    max_ttc: float = 999.0,
) -> float:
    """
    TTC = 거리 / 접근 속도.
    거리가 음수(뒤쪽)이면 위험 판정에서 제외하기 위해 큰 값 반환.
    """
    if distance_longitudinal_m <= 0:
        return max_ttc
    cs = max(closing_speed_ms_val, min_speed)
    return min(distance_longitudinal_m / cs, max_ttc)


def risk_tier_from_ttc(
    ttc: float, critical: float, warning: float
) -> str:
    if ttc <= critical:
        return "critical"
    if ttc <= warning:
        return "warning"
    return "info"


@dataclass
class EgoState:
    location: np.ndarray
    forward: np.ndarray
    velocity: np.ndarray


def project_world_to_minimap(
    hazard_xy: Tuple[float, float],
    ego_xy: Tuple[float, float],
    ego_yaw_rad: float,
    scale_m_per_px: float,
    center_px: Tuple[float, float],
) -> Tuple[int, int]:
    """
    월드 XY를 자차 기준 회전 후 미니맵 픽셀 좌표로 (투시/탑다운 강조용).
    """
    hx, hy = hazard_xy
    ex, ey = ego_xy
    dx = hx - ex
    dy = hy - ey
    c = math.cos(-ego_yaw_rad)
    s = math.sin(-ego_yaw_rad)
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    cx, cy = center_px
    px = int(cx + lx / scale_m_per_px)
    py = int(cy - ly / scale_m_per_px)
    return px, py
