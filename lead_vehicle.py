"""
YOLO 결과에서 전방 선행 차량 1대에 대한 TTC·바운딩박스 면적 추출.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from config import CAMERA_FOV_DEG, CAMERA_HEIGHT, CAMERA_WIDTH
from geometry import (
    closing_speed_ms,
    compute_ttc,
    depth_buffer_to_meters,
    longitudinal_distance_m,
    unproject_pixel_ray_to_world,
)

# yolo_risk 와 동일 COCO id
COCO_CAR = 2
COCO_BUS = 5
COCO_TRUCK = 7


def compute_lead_vehicle_metrics(
    yolo_result: Any,
    depth_bgra: np.ndarray,
    cam_matrix_4x4: np.ndarray,
    ego_loc: np.ndarray,
    ego_fwd: np.ndarray,
    ego_vel: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float, float, float]]]:
    """
    선행 차량(승용·트럭·버스) 중 종방향 거리가 가장 짧고 양수인 대상 1대의
    (TTC, bbox 면적, xyxy). 없으면 (None, None, None).
    """
    res = yolo_result
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return None, None, None

    depth_m = depth_buffer_to_meters(depth_bgra)
    xyxys = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)

    best_ttc = float("inf")
    best_area: Optional[float] = None
    best_xyxy: Optional[Tuple[float, float, float, float]] = None

    for xyxy, cid in zip(xyxys, cls_ids):
        if cid not in (COCO_CAR, COCO_BUS, COCO_TRUCK):
            continue
        x1, y1, x2, y2 = xyxy.tolist()
        area = max(0.0, (x2 - x1) * (y2 - y1))
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        # 깊이 샘플 (sender 와 동일 로직 인라인 최소화)
        ui = int(round(u))
        vi = int(round(v))
        h, w = depth_m.shape[:2]
        ui = max(0, min(w - 1, ui))
        vi = max(0, min(h - 1, vi))
        d = float(depth_m[vi, ui])
        if d <= 0.05 or np.isnan(d):
            patch = depth_m[
                max(0, vi - 2) : min(h, vi + 3),
                max(0, ui - 2) : min(w, ui + 3),
            ]
            if patch.size:
                d = float(np.median(patch))
        d = max(d, 0.1)

        loc = unproject_pixel_ray_to_world(
            u,
            v,
            d,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            CAMERA_FOV_DEG,
            cam_matrix_4x4,
        )
        dist_long = longitudinal_distance_m(ego_loc, ego_fwd, loc)
        if dist_long <= 0.5:
            continue
        cs = closing_speed_ms(ego_vel, ego_fwd, None)
        ttc = compute_ttc(dist_long, cs)
        if ttc < best_ttc:
            best_ttc = ttc
            best_area = area
            best_xyxy = (x1, y1, x2, y2)

    if best_xyxy is None or best_ttc >= 900.0:
        return None, None, None
    return float(best_ttc), float(best_area), best_xyxy
