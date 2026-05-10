"""
CARLA 포스터용: 차량 3대를 같은 차선(waypoint 기반)에 정적으로 배치하고
월드 고정 카메라로 1장 촬영 후 종료.

요구사항:
- Town04 직선 도로 waypoint 기반으로 A,B,C를 같은 차선에 배치
- 간격: --ab-m (A-B), --bc-m (B-C)
- A=빨강 트럭, B=파랑 트럭, C=초록 승용차
- 카메라: ABC 중심 후방 40m, 높이 20m, pitch -35, yaw=차량 진행 방향, FOV 90, 1920x1080
- 배치 후 world.tick() 5회 안정화, 다음 tick에서 이미지 저장
- Python 3.7, CARLA 0.9.x, 동기 모드
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Optional, Tuple

import numpy as np

from config import CARLA_HOST, CARLA_PORT


class LatestCarlaImage:
    """센서 콜백에서 최신 carla.Image 1장만 보관."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._img: Any = None
        self._frame: Optional[int] = None

    def set(self, img: Any) -> None:
        with self._lock:
            self._img = img
            self._frame = int(getattr(img, "frame", -1))

    def get(self) -> Tuple[Any, Optional[int]]:
        with self._lock:
            return self._img, self._frame


def _ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)


def _try_set_color(bp: Any, rgb: Tuple[int, int, int]) -> None:
    if bp.has_attribute("color"):
        bp.set_attribute("color", f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}")


def _intrinsic_matrix(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    k = np.identity(3)
    k[0, 0] = f
    k[1, 1] = f
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def _project_world_to_pixel(world_point: Any, camera_tf: Any, k: np.ndarray, w: int, h: int) -> bool:
    """
    CARLA 권장 예제를 따라 world->camera 변환 후 투영.
    return: in_frame 여부만.
    """
    p = np.array([world_point.x, world_point.y, world_point.z, 1.0], dtype=np.float64)
    world_2_cam = np.array(camera_tf.get_inverse_matrix(), dtype=np.float64)
    pc = world_2_cam @ p

    depth = float(pc[0])
    if depth <= 0.1:
        return False
    x = float(pc[1])
    y = float(-pc[2])
    z = depth

    u = (k[0, 0] * x / z) + k[0, 2]
    v = (k[1, 1] * y / z) + k[1, 2]
    return 0.0 <= u < float(w) and 0.0 <= v < float(h)


def _vehicle_bbox_corners_world(carla: Any, actor: Any) -> list:
    bb = actor.bounding_box
    bb_tf = carla.Transform(bb.location, bb.rotation)
    actor_tf = actor.get_transform()
    e = bb.extent
    corners_local = [
        carla.Location(x=sx * e.x, y=sy * e.y, z=sz * e.z)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    # Transform * Transform 미지원 빌드 호환: 두 단계 변환
    return [actor_tf.transform(bb_tf.transform(c)) for c in corners_local]


def _all_abc_in_frame(carla: Any, cam_tf: Any, w: int, h: int, fov: float, a: Any, b: Any, c: Any) -> bool:
    k = _intrinsic_matrix(w, h, fov)
    for act in (a, b, c):
        for p in _vehicle_bbox_corners_world(carla, act):
            if not _project_world_to_pixel(p, cam_tf, k, w, h):
                return False
    return True


def _count_abc_outside_frame(
    carla: Any, cam_tf: Any, w: int, h: int, fov: float, a: Any, b: Any, c: Any
) -> Tuple[int, int]:
    """ABC bbox 8코너 중 프레임 밖 포인트 개수/전체 개수."""
    k = _intrinsic_matrix(w, h, fov)
    total = 0
    outside = 0
    for act in (a, b, c):
        for p in _vehicle_bbox_corners_world(carla, act):
            total += 1
            if not _project_world_to_pixel(p, cam_tf, k, w, h):
                outside += 1
    return outside, total


def _wait_for_frame(buf: LatestCarlaImage, target_frame: int, timeout_s: float = 2.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        _img, fr = buf.get()
        if fr is not None and int(fr) == int(target_frame):
            return True
        time.sleep(0.005)
    return False


def _pick_vehicle_blueprints(bp_lib: Any) -> Tuple[Any, Any, Any]:
    """
    A/B 트럭은 가능한 서로 다른 blueprint id를 사용해 색상 덮어쓰기 위험을 줄인다.
    """
    truck_bps = []
    for f in ("vehicle.*truck*", "vehicle.carlamotors.*", "vehicle.*"):
        truck_bps = list(bp_lib.filter(f))
        if truck_bps:
            break
    sedan_bps = []
    for f in ("vehicle.tesla.model3", "vehicle.audi.*", "vehicle.*"):
        sedan_bps = list(bp_lib.filter(f))
        if sedan_bps:
            break
    if not truck_bps or not sedan_bps:
        raise RuntimeError("failed to pick vehicle blueprints")

    a_bp = truck_bps[0]
    b_bp = truck_bps[0]
    for bp in truck_bps[1:]:
        if getattr(bp, "id", None) != getattr(a_bp, "id", None):
            b_bp = bp
            break
    c_bp = sedan_bps[0]
    return a_bp, b_bp, c_bp


def _spawn_camera(world: Any, bp_lib: Any, w: int, h: int, fov: float) -> Tuple[Any, LatestCarlaImage]:
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(int(w)))
    cam_bp.set_attribute("image_size_y", str(int(h)))
    cam_bp.set_attribute("fov", str(float(fov)))
    cam_bp.set_attribute("sensor_tick", "0.0")
    buf = LatestCarlaImage()
    cam = world.spawn_actor(cam_bp, world.get_map().get_spawn_points()[0])  # 임시 위치
    cam.listen(buf.set)
    return cam, buf


def _compute_camera_tf(carla: Any, a: Any, b: Any, c: Any, z: float) -> Any:
    b_tf = b.get_transform()
    fwd = b_tf.get_forward_vector()
    la, lb, lc = a.get_location(), b.get_location(), c.get_location()
    center = carla.Location(
        x=(la.x + lb.x + lc.x) / 3.0,
        y=(la.y + lb.y + lc.y) / 3.0,
        z=(la.z + lb.z + lc.z) / 3.0,
    )
    cam_loc = carla.Location(x=center.x - fwd.x * 40.0, y=center.y - fwd.y * 40.0, z=float(z))
    yaw = float(b_tf.rotation.yaw)
    return carla.Transform(cam_loc, carla.Rotation(pitch=-35.0, yaw=yaw, roll=0.0))


def _find_straight_waypoint(world: Any, carla: Any, needed_forward_m: float) -> Any:
    """
    Town04에서 직선에 가까운 waypoint를 찾는다.
    - spawn_point들의 waypoint를 후보로 보고, yaw 변화가 작은 구간(직선)을 고른다.
    """
    m = world.get_map()
    spawns = m.get_spawn_points()
    if not spawns:
        raise RuntimeError("no spawn points in map")

    lane_type = carla.LaneType.Driving
    step = 2.0
    for i, sp in enumerate(spawns[:50]):
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=lane_type)
        if wp is None:
            continue
        ok = True
        yaw0 = float(wp.transform.rotation.yaw)
        cur = wp
        walked = 0.0
        while walked < needed_forward_m:
            nxts = cur.next(step)
            if not nxts:
                ok = False
                break
            cur = nxts[0]
            walked += step
            yaw = float(cur.transform.rotation.yaw)
            dyaw = abs(((yaw - yaw0 + 180.0) % 360.0) - 180.0)
            if dyaw > 5.0:
                ok = False
                break
        if ok:
            print(f"[wp_ok] spawn_points[{i}] road_id={wp.road_id} lane_id={wp.lane_id}", flush=True)
            return wp

    raise RuntimeError("failed to find a straight waypoint segment (try another map or widen search)")


def _spawn_abc_same_lane(world: Any, carla: Any, ab_m: float, bc_m: float) -> Tuple[Any, Any, Any]:
    bp_lib = world.get_blueprint_library()
    a_bp, b_bp, c_bp = _pick_vehicle_blueprints(bp_lib)
    _try_set_color(a_bp, (220, 60, 60))  # A red
    _try_set_color(b_bp, (70, 120, 220))  # B blue
    _try_set_color(c_bp, (60, 200, 120))  # C green

    needed = float(ab_m) + 5.0
    wp_b = _find_straight_waypoint(world, carla, needed_forward_m=needed)

    # B waypoint 기준으로 앞/뒤 waypoint를 같은 차선에서 따라간다.
    # next/previous는 같은 lane을 따라간다고 가정(Driving lane).
    def advance(wp: Any, meters: float) -> Any:
        step = 2.0
        cur = wp
        remain = abs(meters)
        while remain > 1e-6:
            d = min(step, remain)
            if meters >= 0:
                nxts = cur.next(d)
                if not nxts:
                    raise RuntimeError("waypoint.next() failed while advancing")
                cur = nxts[0]
            else:
                prevs = cur.previous(d)
                if not prevs:
                    raise RuntimeError("waypoint.previous() failed while advancing")
                cur = prevs[0]
            remain -= d
        return cur

    wp_a = advance(wp_b, float(ab_m))
    wp_c = advance(wp_b, -float(bc_m))

    # 약간 띄워서 스폰 (바닥에 박히는 것 완화)
    def lift(tf: Any, dz: float) -> Any:
        return carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + dz), tf.rotation)

    b_tf = lift(wp_b.transform, 0.8)
    a_tf = lift(wp_a.transform, 0.8)
    c_tf = lift(wp_c.transform, 0.8)

    b = world.try_spawn_actor(b_bp, b_tf)
    a = world.try_spawn_actor(a_bp, a_tf)
    c = world.try_spawn_actor(c_bp, c_tf)
    if not (a and b and c):
        for actor in (a, b, c):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        raise RuntimeError("spawn failed: could not place A,B,C on same lane")

    print("[spawn_ok] A,B,C spawned (same lane)", flush=True)
    return a, b, c


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ab-m", type=float, default=30.0)
    p.add_argument("--bc-m", type=float, default=20.0)
    p.add_argument("--output", required=True)
    p.add_argument("--host", default=CARLA_HOST)
    p.add_argument("--port", type=int, default=int(CARLA_PORT))
    p.add_argument("--map", default="Town04")
    p.add_argument("--fixed-dt", type=float, default=0.05)
    args = p.parse_args()

    import carla

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(60.0)
    client.load_world(str(args.map))
    time.sleep(3.0)
    world = client.get_world()

    cam = None
    a = b = c = None

    settings = world.get_settings()
    try:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_dt)
        world.apply_settings(settings)

        a, b, c = _spawn_abc_same_lane(world, carla, float(args.ab_m), float(args.bc_m))

        bp_lib = world.get_blueprint_library()
        cam, buf = _spawn_camera(world, bp_lib, 1920, 1080, 90.0)

        # 배치 후 안정화 tick 5회
        for _ in range(5):
            world.tick()

        # 안정화 후 위치 기준으로 카메라를 다시 계산/적용
        cam_tf = _compute_camera_tf(carla, a, b, c, z=20.0)
        cam.set_transform(cam_tf)

        # 촬영 tick (world.tick() 반환값이 snapshot)
        snap = world.tick()
        if not _wait_for_frame(buf, int(snap.frame), timeout_s=2.0):
            raise RuntimeError(f"camera frame not ready (snap.frame={int(snap.frame)})")

        outside, total = _count_abc_outside_frame(carla, cam_tf, 1920, 1080, 90.0, a, b, c)
        if outside > 0:
            print(
                f"[WARN] ABC bbox corners outside frame: {outside}/{total} "
                f"(ab={args.ab_m}m bc={args.bc_m}m z=20 fov=90)",
                flush=True,
            )

        img, _fr = buf.get()
        if img is None:
            raise RuntimeError("no image received from camera")

        _ensure_dir(str(args.output))
        img.save_to_disk(str(args.output), carla.ColorConverter.Raw)
        print(f"[ok] saved {args.output}", flush=True)

    finally:
        try:
            if cam is not None:
                cam.stop()
                cam.destroy()
        except Exception:
            pass
        for actor in (a, b, c):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        try:
            s2 = world.get_settings()
            s2.synchronous_mode = False
            world.apply_settings(s2)
        except Exception:
            pass


if __name__ == "__main__":
    main()

