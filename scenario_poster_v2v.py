"""
포스터용 3차량(A,B,C) 급정거 비교 시나리오.

요구사항 요약:
- 차량 3대: A(선행 트럭, 급정거), B(중간 트럭), C(후방 승용)
- A-B 30m, B-C 20m, 모두 60km/h
- A는 시뮬 시작 3.0s에 최대 제동(근사로 brake=1.0) 시작
- 시나리오1(기기 없음): B는 A 급정거 1.5s 후 반응, C는 B 급정거 1.5s 후 반응
- 시나리오2(기기 있음): A 급정거 신호 즉시 전달, B/C는 0.3s 후 반응
- 기록: tick마다 AB/BC 거리, B/C 속도, B/C 제동입력 + 충돌 여부/시각
- 캡처(월드 카메라, 캡처 직전 재배치 + ABC 프레임 검증):
  1) cruise_shot: t=2.0s
  2) brake_shot: AB 거리가 초기 대비 5m 감소한 첫 시점
  3) compare_shot: 시나리오1에서 기준(기본: BC < X m)을 만족하는 시각 T*를 찾고, 같은 T*에서 두 시나리오 모두 캡처
  4) result_shot:
     - 시나리오1: 충돌 직전 프레임(근사: BC<2m 첫 시점)
     - 시나리오2: C 완전 정지(속도<0.1m/s) 첫 시점

출력:
  <output_root>/scenario1/{cruise_shot,brake_shot,compare_shot,result_shot}.png
  <output_root>/scenario2/{...}.png
  <output_root>/scenario1/ticks.csv, scenario2/ticks.csv
  <output_root>/scenario1/summary.json, scenario2/summary.json

기본 실행은 확인용(부하 낮음): fixed_dt=0.08, 1280x720. 최종 포스터는 동일 dt로 --img-w 1920 --img-h 1080.
python scenario_poster_v2v.py --help 에 실행 순서 예시 참고.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from config import CARLA_HOST, CARLA_PORT


@dataclass
class ScenarioSpec:
    name: str
    v2v_enabled: bool
    b_reaction_s: float
    c_reaction_s: float
    c_reaction_relative_to: str  # "A" | "B"


@dataclass
class RunSummary:
    collision: bool
    collision_time_s: Optional[float]
    t_star_compare_s: Optional[float]
    bc_at_compare_m: Optional[float] = None
    compare_camera: Optional[Dict[str, Any]] = None


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
    os.makedirs(p, exist_ok=True)


def _speed_ms(actor: Any) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _distance_m(a: Any, b: Any) -> float:
    return a.get_location().distance(b.get_location())


def _cruise_control(target_ms: float, current_ms: float) -> Tuple[float, float]:
    err = target_ms - current_ms
    if err >= 0:
        throttle = min(0.75, max(0.0, 0.35 + err * 0.05))
        return throttle, 0.0
    brake = min(0.55, max(0.0, (-err) * 0.06))
    return 0.0, brake


def _forward_offset_tf(base: Any, forward_m: float, carla: Any) -> Any:
    f = base.get_forward_vector()
    loc = carla.Location(
        base.location.x + f.x * forward_m,
        base.location.y + f.y * forward_m,
        base.location.z + f.z * forward_m,
    )
    return carla.Transform(loc, base.rotation)


def _intrinsic_matrix(w: int, h: int, fov_deg: float) -> np.ndarray:
    # CARLA 카메라 모델(핀홀) 근사
    f = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    k = np.identity(3)
    k[0, 0] = f
    k[1, 1] = f
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def _project_world_to_pixel(
    world_point: Any,
    camera_tf: Any,
    k: np.ndarray,
    w: int,
    h: int,
) -> Tuple[bool, float, float]:
    """
    CARLA 권장 예제를 따라 world -> camera 변환 후 투영.
    return: (in_frame, u, v)
    """
    p = np.array([world_point.x, world_point.y, world_point.z, 1.0], dtype=np.float64)
    world_2_cam = np.array(camera_tf.get_inverse_matrix(), dtype=np.float64)
    pc = world_2_cam @ p  # camera 좌표계(카메라 기준)

    # CARLA 카메라 좌표계 변환: (x,y,z) -> (y, -z, x)
    # depth = x 축(전방)
    depth = float(pc[0])
    if depth <= 0.1:
        return False, -1.0, -1.0
    x = float(pc[1])
    y = float(-pc[2])
    z = depth

    u = (k[0, 0] * x / z) + k[0, 2]
    v = (k[1, 1] * y / z) + k[1, 2]
    return (0.0 <= u < float(w) and 0.0 <= v < float(h)), float(u), float(v)


def _vehicle_bbox_corners_world(carla: Any, actor: Any) -> list:
    """
    actor의 bounding box 8개 코너를 world 좌표로 반환.
    중심점만 검사하면 트럭이 잘릴 수 있어, 코너 기반으로 프레임 검증에 사용한다.
    """
    bb = actor.bounding_box
    # actor 기준 bbox transform을 world로 변환
    # bb.location은 actor 원점 기준 오프셋, bb.rotation은 actor 기준 회전.
    bb_tf = carla.Transform(bb.location, bb.rotation)
    actor_tf = actor.get_transform()
    e = bb.extent
    # 8 corners in bbox-local
    corners_local = [
        carla.Location(x= sx * e.x, y= sy * e.y, z= sz * e.z)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    # CARLA 0.9.13 Python: 일부 빌드에서 Transform * Transform 미지원 → 두 단계 변환으로 동일 결과
    return [actor_tf.transform(bb_tf.transform(c)) for c in corners_local]


def _vehicle_bbox_corners_at_predicted_location(
    carla: Any, actor: Any, loc_pred: Any
) -> list:
    """
    현재 자세·bounding_box 정의를 유지하고, 월드 원점만 loc_pred 기준으로 평행이동한 코너.
    카메라 `_compute_abc_camera_tf_from_points(..., la_p, lb_p, lc_p, ...)`와 같은 예측점을 쓴다.
    """
    loc_now = actor.get_location()
    dx = float(loc_pred.x - loc_now.x)
    dy = float(loc_pred.y - loc_now.y)
    dz = float(loc_pred.z - loc_now.z)
    return [
        carla.Location(x=p.x + dx, y=p.y + dy, z=p.z + dz)
        for p in _vehicle_bbox_corners_world(carla, actor)
    ]


def _all_vehicles_bbox_in_frame(
    carla: Any,
    camera_tf: Any,
    k: np.ndarray,
    w: int,
    h: int,
    actors: list,
) -> bool:
    """
    각 actor의 bbox 코너들이 모두 프레임 안에 들어오는지 확인.
    (너무 엄격해 실패가 잦다면 추후 완화 가능하지만, 포스터 목적상 안전하게 간다.)
    """
    for act in actors:
        for p in _vehicle_bbox_corners_world(carla, act):
            ok, _, _ = _project_world_to_pixel(p, camera_tf, k, w, h)
            if not ok:
                return False
    return True


def _all_bbox_points_in_frame(
    carla: Any, camera_tf: Any, k: np.ndarray, w: int, h: int, points: list
) -> bool:
    """미리 계산한 월드 좌표 점들이 모두 프레임 안인지 확인."""
    for p in points:
        ok, _, _ = _project_world_to_pixel(p, camera_tf, k, w, h)
        if not ok:
            return False
    return True


def _spawn_world_camera(
    world: Any,
    bp_lib: Any,
    w: int,
    h: int,
    fov: float,
    sensor_tick: float = 0.0,
) -> Tuple[Any, LatestCarlaImage]:
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(int(w)))
    cam_bp.set_attribute("image_size_y", str(int(h)))
    cam_bp.set_attribute("fov", str(float(fov)))
    cam_bp.set_attribute("sensor_tick", str(float(sensor_tick)))
    buf = LatestCarlaImage()
    cam = world.spawn_actor(cam_bp, world.get_map().get_spawn_points()[0])  # 임시 위치
    cam.listen(buf.set)
    return cam, buf


def _compute_abc_camera_tf_from_points(
    carla: Any, la: Any, lb: Any, lc: Any, b_tf: Any, z: float
) -> Any:
    """요구 스펙의 월드 카메라 Transform 계산(명시적 점 입력)."""
    yaw = float(b_tf.rotation.yaw)
    fwd = b_tf.get_forward_vector()
    center = carla.Location(
        x=(la.x + lb.x + lc.x) / 3.0,
        y=(la.y + lb.y + lc.y) / 3.0,
        z=(la.z + lb.z + lc.z) / 3.0,
    )
    cam_loc = carla.Location(
        x=center.x - fwd.x * 40.0,
        y=center.y - fwd.y * 40.0,
        z=float(z),
    )
    return carla.Transform(cam_loc, carla.Rotation(pitch=-35.0, yaw=yaw, roll=0.0))


def _predict_location(carla: Any, actor: Any, dt: float) -> Any:
    """현재 위치 + 속도*dt 로 다음 tick 위치를 근사."""
    loc = actor.get_location()
    vel = actor.get_velocity()
    return carla.Location(
        x=float(loc.x + vel.x * dt),
        y=float(loc.y + vel.y * dt),
        z=float(loc.z + vel.z * dt),
    )


def _ensure_camera_for_abc_and_maybe_respawn(
    world: Any,
    carla: Any,
    bp_lib: Any,
    cam: Any,
    buf: LatestCarlaImage,
    la: Any,
    lb: Any,
    lc: Any,
    b_tf: Any,
    bbox_points: list,
    base_z: float,
    base_fov: float,
    w: int,
    h: int,
    max_retries: int,
) -> Tuple[bool, Any, LatestCarlaImage, float, float]:
    """
    선택지1 구현:
    - z / fov 재시도 중 fov가 커지는 경우, 실제 센서를 재스폰하여 "검증 fov == 실제 캡처 fov"가 되게 한다.
    - 시간은 절대 진행시키지 않는다(world.tick 호출 금지).
    """
    last_cam = cam
    last_buf = buf

    ok = False
    final_z = base_z
    final_fov = base_fov

    for i in range(max_retries):
        final_z = base_z + 5.0 * i
        final_fov = base_fov + 10.0 * i
        cam_tf = _compute_abc_camera_tf_from_points(carla, la, lb, lc, b_tf, final_z)

        if i > 0:
            try:
                last_cam.stop()
                last_cam.destroy()
            except Exception:
                pass
            last_cam, last_buf = _spawn_world_camera(
                world, bp_lib, w, h, final_fov, sensor_tick=0.0
            )

        last_cam.set_transform(cam_tf)

        k = _intrinsic_matrix(w, h, final_fov)
        if _all_bbox_points_in_frame(carla, cam_tf, k, w, h, bbox_points):
            ok = True
            break

    return ok, last_cam, last_buf, final_z, final_fov


def _ensure_camera_fixed_xy_with_retry(
    world: Any,
    carla: Any,
    bp_lib: Any,
    cam: Any,
    buf: LatestCarlaImage,
    la: Any,
    lb: Any,
    lc: Any,
    base_x: float,
    base_y: float,
    base_z: float,
    base_pitch: float,
    base_yaw: float,
    base_roll: float,
    base_fov: float,
    w: int,
    h: int,
    max_retries: int,
    bbox_points: list,
) -> Tuple[bool, Any, LatestCarlaImage, float, float]:
    """
    compare_shot 구도 고정용:
    - x,y,yaw,pitch는 고정하고, z와 fov만 키워서 ABC가 프레임에 들어오게 만든다.
    - fov가 바뀌면 센서 재스폰(선택지1 일관).
    """
    last_cam = cam
    last_buf = buf

    ok = False
    final_z = base_z
    final_fov = base_fov

    for i in range(max_retries):
        final_z = base_z + 5.0 * i
        final_fov = base_fov + 10.0 * i

        if i > 0:
            try:
                last_cam.stop()
                last_cam.destroy()
            except Exception:
                pass
            last_cam, last_buf = _spawn_world_camera(
                world, bp_lib, w, h, final_fov, sensor_tick=0.0
            )

        cam_tf = carla.Transform(
            carla.Location(x=float(base_x), y=float(base_y), z=float(final_z)),
            carla.Rotation(
                pitch=float(base_pitch), yaw=float(base_yaw), roll=float(base_roll)
            ),
        )
        last_cam.set_transform(cam_tf)

        k = _intrinsic_matrix(w, h, final_fov)
        if _all_bbox_points_in_frame(carla, cam_tf, k, w, h, bbox_points):
            ok = True
            break

    return ok, last_cam, last_buf, final_z, final_fov


def _save_latest_png(img: Any, out_path: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    try:
        import carla

        img.save_to_disk(out_path, carla.ColorConverter.Raw)
    except Exception:
        img.save_to_disk(out_path)


def _capture_named_shot(buf: LatestCarlaImage, shot_path: str) -> None:
    """현재 buf의 프레임만 저장(순수 저장 함수)."""
    img, _fr = buf.get()
    if img is not None:
        _save_latest_png(img, shot_path)


def _pick_vehicle_blueprints(bp_lib: Any) -> Tuple[Any, Any, Any]:
    """
    포스터 가독성을 위해 A/B는 서로 다른 truck blueprint를 우선 선택한다.
    - CARLA Blueprint 객체는 set_attribute 시 내부 상태가 바뀌므로, 같은 객체를 재사용하면 색상이 덮일 수 있다.
    - 따라서 A/B는 가능하면 서로 다른 blueprint id를 사용한다(가장 안전).
    return: (a_truck_bp, b_truck_bp, sedan_bp)
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

    # A/B는 서로 다른 id 우선
    a_bp = truck_bps[0]
    b_bp = truck_bps[0]
    for bp in truck_bps[1:]:
        if getattr(bp, "id", None) != getattr(a_bp, "id", None):
            b_bp = bp
            break

    sedan_bp = sedan_bps[0]
    return a_bp, b_bp, sedan_bp


def _try_set_color(bp: Any, rgb: Tuple[int, int, int]) -> None:
    if bp.has_attribute("color"):
        bp.set_attribute("color", f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}")


def _spawn_abc(world: Any, carla: Any, ab_m: float, bc_m: float) -> Tuple[Any, Any, Any]:
    bp_lib = world.get_blueprint_library()
    a_truck_bp, b_truck_bp, sedan_bp = _pick_vehicle_blueprints(bp_lib)
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("no spawn points in map")

    # 포스터 가독성: A/B/C 색상 구분 (가능한 blueprint만)
    a_bp = a_truck_bp
    b_bp = b_truck_bp
    c_bp = sedan_bp
    _try_set_color(a_bp, (220, 60, 60))  # A: red
    _try_set_color(b_bp, (70, 120, 220))  # B: blue
    _try_set_color(c_bp, (60, 200, 120))  # C: green

    # 첫 스폰 포인트만 쓰면 장애물/경사/겹침으로 실패할 수 있음 → 여러 지점·Z 보정 재시도 (상한 10)
    max_bases = min(len(spawns), 10)
    z_lifts = (0.8, 1.2, 1.6)
    last_err = None
    for bi in range(max_bases):
        base = spawns[bi]
        for dz in z_lifts:
            b_tf = carla.Transform(
                carla.Location(base.location.x, base.location.y, base.location.z + dz),
                base.rotation,
            )
            a_tf = _forward_offset_tf(b_tf, ab_m, carla)
            c_tf = _forward_offset_tf(b_tf, -bc_m, carla)

            b = world.try_spawn_actor(b_bp, b_tf)
            a = world.try_spawn_actor(a_bp, a_tf)
            c = world.try_spawn_actor(c_bp, c_tf)
            if a and b and c:
                _m = world.get_map()
                _map_name = getattr(_m, "name", str(_m))
                print(
                    f"[spawn_ok] spawn_points[{bi}] z_lift={dz:.1f}m map={_map_name}",
                    flush=True,
                )
                return a, b, c

            for actor in (a, b, c):
                if actor is not None:
                    try:
                        actor.destroy()
                    except Exception:
                        pass
            last_err = (bi, dz)

    hint = (
        f"tried spawn_points[0..{max_bases - 1}] with z+{z_lifts} — "
        "맵을 바꾸거나 CARLA 재시작 후 재실행해 보세요."
    )
    if last_err:
        hint += f" last_try=(idx={last_err[0]}, dz={last_err[1]})."
    raise RuntimeError(f"spawn failed: could not place A,B,C ({hint})")


def _weather_from_name(carla: Any, name: str) -> Any:
    """WeatherParameters.<Name> lookup. 예: ClearNoon"""
    if not name:
        raise ValueError("weather name is empty")
    wp = getattr(carla.WeatherParameters, name, None)
    if wp is None:
        raise ValueError(f"unknown weather: {name}")
    return wp


def run_one_scenario(
    spec: ScenarioSpec,
    world: Any,
    output_dir: str,
    fixed_dt: float,
    speed_kmh: float,
    ab0_m: float,
    bc0_m: float,
    a_brake_time_s: float,
    max_ticks: int,
    image_w: int,
    image_h: int,
    cam_z: float,
    cam_fov: float,
    cam_retry: int,
    compare_t_star_s: Optional[float],
    compare_anchor: Literal["c_brake", "bc_lt"],
    compare_bc_lt_m: float,
    compare_lock_camera: bool,
    compare_camera_from_s1: Optional[Dict[str, Any]],
) -> RunSummary:
    import carla

    _ensure_dir(output_dir)

    settings = world.get_settings()
    cam = None
    a = b = c = None
    out: Optional[RunSummary] = None

    try:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(fixed_dt)
        world.apply_settings(settings)

        a, b, c = _spawn_abc(world, carla, ab0_m, bc0_m)

        # 월드 카메라(캡처 순간에만 transform 이동)
        bp_lib = world.get_blueprint_library()
        cam, buf = _spawn_world_camera(
            world, bp_lib, image_w, image_h, cam_fov, sensor_tick=0.0
        )
        print(
            f"[poster] {spec.name} rgb_cam {image_w}x{image_h} fov={cam_fov}",
            flush=True,
        )

        # 로그 파일
        ticks_csv = os.path.join(output_dir, "ticks.csv")
        summary_json = os.path.join(output_dir, "summary.json")

        speed_ms = float(speed_kmh) / 3.6
        ab_initial = _distance_m(a, b)
        bc_initial = _distance_m(b, c)

        # 반응 스케줄: "시뮬 시작 후" 기준이 되도록 t0를 잡고 상대시간으로 비교한다.
        # CARLA elapsed_seconds는 서버 누적 시간이어서 절대값이 커질 수 있음.
        t0_abs: Optional[float] = None
        t_a_brake_rel = float(a_brake_time_s)
        t_b_brake_rel = t_a_brake_rel + float(spec.b_reaction_s)
        if spec.c_reaction_relative_to.upper() == "B":
            t_c_brake_rel = t_b_brake_rel + float(spec.c_reaction_s)
        else:
            t_c_brake_rel = t_a_brake_rel + float(spec.c_reaction_s)
    
        # 캡처 플래그
        did_cruise = False
        did_brake = False
        did_compare = False
        did_result = False
    
        # compare_shot 기준(T*) 기록(시나리오1에서만 계산)
        t_star_found: Optional[float] = None
        bc_at_compare: Optional[float] = None
        compare_cam_spec: Optional[Dict[str, Any]] = None
    
        collision_time: Optional[float] = None
        collision = False
    
        with open(ticks_csv, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(
                f,
                fieldnames=[
                    "tick",
                    "sim_time_s",
                    "ab_m",
                    "bc_m",
                    "b_speed_ms",
                    "c_speed_ms",
                    "b_brake",
                    "c_brake",
                ],
            )
            wtr.writeheader()
    
            # 워밍업: 초기 속도 수렴 (최소 2초 권장)
            warmup_ticks = int(round(2.0 / max(fixed_dt, 1e-6)))
            _wu = max(10, warmup_ticks)
            print(
                f"[poster] {spec.name} warmup ticks={_wu} (~{_wu * fixed_dt:.2f}s sim) …",
                flush=True,
            )
            for _ in range(_wu):
                world.tick()
                for actor in (a, b, c):
                    cur = _speed_ms(actor)
                    th, br = _cruise_control(speed_ms, cur)
                    actor.apply_control(carla.VehicleControl(throttle=th, brake=br))

            # 워밍업이 끝난 뒤의 실제 간격을 "초기"로 삼는다. (브레이크샷 기준 일관성)
            ab_initial = float(_distance_m(a, b))
            bc_initial = float(_distance_m(b, c))
            print(
                f"[poster] {spec.name} warmup done ab={ab_initial:.2f}m bc={bc_initial:.2f}m "
                f"main_loop max_ticks={max_ticks}",
                flush=True,
            )

            # 캡처는 "조건을 만족한 tick 다음 tick"에 수행한다.
            # (카메라 transform이 다음 센서 프레임부터 적용되기 때문)
            pending_label: Optional[str] = None
            pending_path: Optional[str] = None
            last_used_fov: float = float(cam_fov)
            heartbeat_every = max(1, int(round(5.0 / max(fixed_dt, 1e-6))))

            for tick_idx in range(int(max_ticks)):
                # 캡처가 예약되어 있으면, tick 전에 카메라를 ABC 기준으로 배치해둔다.
                # (FOV 재시도에서 센서를 재스폰할 수 있으므로 cam/buf를 갱신한다.)
                if pending_path is not None:
                    # 이 tick(world.tick()) 이후 렌더될 상태를 근사해서 카메라를 배치한다.
                    # (카메라 배치와 캡처 프레임의 위치 오차를 줄이기 위한 1-tick 예측)
                    b_tf_now = b.get_transform()
                    la_p = _predict_location(carla, a, fixed_dt)
                    lb_p = _predict_location(carla, b, fixed_dt)
                    lc_p = _predict_location(carla, c, fixed_dt)
                    bbox_points = (
                        _vehicle_bbox_corners_at_predicted_location(carla, a, la_p)
                        + _vehicle_bbox_corners_at_predicted_location(carla, b, lb_p)
                        + _vehicle_bbox_corners_at_predicted_location(carla, c, lc_p)
                    )
    
                    # compare_shot은 두 시나리오의 "구도"를 맞추기 위해 scenario1의 카메라를 고정할 수 있다.
                    if (
                        pending_label == "compare"
                        and compare_lock_camera
                        and compare_camera_from_s1
                        and spec.name == "scenario2"
                    ):
                        base = compare_camera_from_s1
                        _ok, cam, buf, _z, _fov = _ensure_camera_fixed_xy_with_retry(
                            world,
                            carla,
                            bp_lib,
                            cam,
                            buf,
                            la_p,
                            lb_p,
                            lc_p,
                            base_x=float(base["x"]),
                            base_y=float(base["y"]),
                            base_z=float(base["z"]),
                            base_pitch=float(base["pitch"]),
                            base_yaw=float(base["yaw"]),
                            base_roll=float(base.get("roll", 0.0)),
                            base_fov=float(base["fov"]),
                            w=image_w,
                            h=image_h,
                            max_retries=cam_retry,
                            bbox_points=bbox_points,
                        )
                        # 고정 구도에서 C가 카메라 뒤로 가는 등으로 실패하면,
                        # 구도보다 "ABC 모두 프레임 내" 조건을 우선해서 일반 추적 카메라로 폴백.
                        if not _ok:
                            _ok, cam, buf, _z, _fov = _ensure_camera_for_abc_and_maybe_respawn(
                                world,
                                carla,
                                bp_lib,
                                cam,
                                buf,
                                la_p,
                                lb_p,
                                lc_p,
                                b_tf_now,
                                bbox_points,
                                cam_z,
                                cam_fov,
                                image_w,
                                image_h,
                                cam_retry,
                            )
                    else:
                        _ok, cam, buf, _z, _fov = _ensure_camera_for_abc_and_maybe_respawn(
                            world,
                            carla,
                            bp_lib,
                            cam,
                            buf,
                            la_p,
                            lb_p,
                            lc_p,
                            b_tf_now,
                            bbox_points,
                            cam_z,
                            cam_fov,
                            image_w,
                            image_h,
                            cam_retry,
                        )
    
                    _ = (_ok, _z, _fov)
                    # 실제로 사용된 fov를 기록해 compare_cam_spec에 저장 가능하게 한다.
                    last_used_fov = float(_fov)
                    if not _ok:
                        print(
                            f"[WARN] {spec.name}:{pending_label} 카메라 프레이밍 검증 실패 "
                            f"(ABC bbox가 프레임 밖). 잘린 이미지가 저장될 수 있습니다.",
                            flush=True,
                        )
    
                world.tick()
                snap = world.get_snapshot()
                sim_time_abs = float(snap.timestamp.elapsed_seconds)
                if t0_abs is None:
                    t0_abs = sim_time_abs
                sim_time = sim_time_abs - t0_abs
    
                # A 제동
                if sim_time >= t_a_brake_rel:
                    a.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                else:
                    cur_a = _speed_ms(a)
                    th_a, br_a = _cruise_control(speed_ms, cur_a)
                    a.apply_control(carla.VehicleControl(throttle=th_a, brake=br_a))
    
                # B 제동
                b_brake = 0.0
                if sim_time >= t_b_brake_rel:
                    b_brake = 1.0
                    b.apply_control(carla.VehicleControl(throttle=0.0, brake=b_brake))
                else:
                    cur_b = _speed_ms(b)
                    th_b, br_b = _cruise_control(speed_ms, cur_b)
                    b.apply_control(carla.VehicleControl(throttle=th_b, brake=br_b))
    
                # C 제동
                c_brake = 0.0
                if sim_time >= t_c_brake_rel:
                    c_brake = 1.0
                    c.apply_control(carla.VehicleControl(throttle=0.0, brake=c_brake))
                else:
                    cur_c = _speed_ms(c)
                    th_c, br_c = _cruise_control(speed_ms, cur_c)
                    c.apply_control(carla.VehicleControl(throttle=th_c, brake=br_c))
    
                ab_m = float(_distance_m(a, b))
                bc_m = float(_distance_m(b, c))
                b_speed = float(_speed_ms(b))
                c_speed = float(_speed_ms(c))

                wtr.writerow(
                    {
                        "tick": tick_idx,
                        "sim_time_s": f"{sim_time:.3f}",
                        "ab_m": f"{ab_m:.3f}",
                        "bc_m": f"{bc_m:.3f}",
                        "b_speed_ms": f"{b_speed:.3f}",
                        "c_speed_ms": f"{c_speed:.3f}",
                        "b_brake": f"{b_brake:.3f}",
                        "c_brake": f"{c_brake:.3f}",
                    }
                )

                if tick_idx > 0 and tick_idx % heartbeat_every == 0:
                    print(
                        f"[poster] {spec.name} tick={tick_idx} sim_t={sim_time:.2f}s "
                        f"ab={ab_m:.1f} bc={bc_m:.1f} "
                        f"shots cruise={did_cruise} brake={did_brake} "
                        f"compare={did_compare} result={did_result}",
                        flush=True,
                    )

                # 충돌 근사(포스터 목적): BC<2m
                if not collision and bc_m < 2.0:
                    collision = True
                    collision_time = sim_time
    
                # compare T* 찾기 (시나리오1에서만 계산)
                # - c_brake: C brake>0.05 첫 시점 (원안)
                # - bc_lt:   BC < X(m) 첫 시점 (포스터 임팩트용)
                if spec.name == "scenario1" and t_star_found is None:
                    if compare_anchor == "c_brake":
                        if c_brake > 0.05:
                            t_star_found = sim_time
                    else:
                        if bc_m < float(compare_bc_lt_m):
                            t_star_found = sim_time
    
                # --- 예약된 샷을 "현재 tick 프레임"으로 저장 (추가 tick 없음) ---
                if pending_path is not None:
                    _capture_named_shot(buf, pending_path)
                    if pending_label == "cruise":
                        did_cruise = True
                    elif pending_label == "brake":
                        did_brake = True
                    elif pending_label == "compare":
                        did_compare = True
                        bc_at_compare = bc_m
                        # scenario1에서 compare_shot 카메라 스펙 저장(나중에 scenario2에 고정 적용 가능)
                        if spec.name == "scenario1":
                            tf = cam.get_transform()
                            compare_cam_spec = {
                                "x": float(tf.location.x),
                                "y": float(tf.location.y),
                                "z": float(tf.location.z),
                                "pitch": float(tf.rotation.pitch),
                                "yaw": float(tf.rotation.yaw),
                                "roll": float(tf.rotation.roll),
                                # 주의: ensure_*에서 fov가 증가할 수 있으므로, 현재 fov를 기록한다.
                                # (sensor attribute를 읽는 API는 없으므로, 마지막으로 사용한 fov를 저장하는 방식이 이상적)
                                "fov": float(last_used_fov),
                            }
                    elif pending_label == "result":
                        did_result = True
                    pending_label = None
                    pending_path = None
    
                # --- 캡처 조건(예약): 조건을 만족한 tick "다음 tick"에 캡처하도록 예약 ---
                if pending_path is None:
                    # cruise_shot: 속도 수렴(±1km/h)한 첫 시점 (순항처럼 보이도록)
                    # A 급정거(3s) 전이어야 하므로 sim_time < t_a_brake_rel 조건도 포함
                    speed_tol = 1.0 / 3.6
                    if (
                        (not did_cruise)
                        and (sim_time < t_a_brake_rel)
                        and (abs(_speed_ms(a) - speed_ms) <= speed_tol)
                        and (abs(_speed_ms(b) - speed_ms) <= speed_tol)
                        and (abs(_speed_ms(c) - speed_ms) <= speed_tol)
                    ):
                        pending_label = "cruise"
                        pending_path = os.path.join(output_dir, "cruise_shot.png")
                    # fallback: 수렴이 안 되더라도 A 급정거 전에 반드시 1장 확보
                    elif (not did_cruise) and (sim_time < t_a_brake_rel) and (sim_time >= 2.5):
                        pending_label = "cruise"
                        pending_path = os.path.join(output_dir, "cruise_shot.png")
                    elif not did_brake and ab_m <= (ab_initial - 5.0):
                        pending_label = "brake"
                        pending_path = os.path.join(output_dir, "brake_shot.png")
                    elif (
                        (compare_t_star_s is not None)
                        and (not did_compare)
                        and sim_time >= compare_t_star_s
                    ):
                        pending_label = "compare"
                        pending_path = os.path.join(output_dir, "compare_shot.png")
                    elif not did_result:
                        if spec.name == "scenario1":
                            if bc_m < 2.0:
                                pending_label = "result"
                                pending_path = os.path.join(output_dir, "result_shot.png")
                            elif sim_time > t_a_brake_rel + 10.0:
                                # BC<2m 미도달(튕김/정지 등) 시 max_ticks까지 도는 것 방지
                                pending_label = "result"
                                pending_path = os.path.join(output_dir, "result_shot.png")
                        else:
                            # 완전 정지 수렴이 너무 느릴 수 있어, compare 이후 5초 경과 시 강제 캡처(근사)
                            if c_speed < 0.1 or (
                                did_compare
                                and (compare_t_star_s is not None)
                                and (sim_time > float(compare_t_star_s) + 5.0)
                            ):
                                pending_label = "result"
                                pending_path = os.path.join(output_dir, "result_shot.png")
    
                # 종료 조건: result_shot까지 찍었고 비교샷까지 찍었다면 종료 가능
                if did_cruise and did_brake and (compare_t_star_s is None or did_compare) and did_result:
                    break

        print(
            f"[poster] {spec.name} finished collision={collision} "
            f"t_star_compare_s="
            f"{t_star_found if spec.name == 'scenario1' else compare_t_star_s}",
            flush=True,
        )

        out = RunSummary(
            collision=bool(collision),
            collision_time_s=collision_time,
            t_star_compare_s=t_star_found if spec.name == "scenario1" else compare_t_star_s,
            bc_at_compare_m=bc_at_compare,
            compare_camera=compare_cam_spec,
        )
        with open(summary_json, "w", encoding="utf-8") as jf:
            json.dump(asdict(out), jf, ensure_ascii=False, indent=2)

    finally:
        # 정리: 센서/액터 제거 후 동기 모드 해제
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

        # 다음 시나리오/초기화 구간에서 deadlock을 피하기 위해 동기 모드를 명시적으로 해제
        try:
            s2 = world.get_settings()
            s2.synchronous_mode = False
            world.apply_settings(s2)
        except Exception:
            pass

    if out is None:
        raise RuntimeError(
            "scenario_poster_v2v: run_one_scenario aborted before writing summary.json"
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "실행 순서 예:\n"
            "  1차(구도·타이밍·WARN 확인, 부하 낮음): 기본값 = fixed-dt 0.08, 1280x720\n"
            "  2차(최종 포스터, 동일 물리): --fixed-dt 0.08 --img-w 1920 --img-h 1080"
        ),
    )
    p.add_argument("--output-root", default="/output", help="출력 루트 (컨테이너에서는 /output 권장)")
    p.add_argument("--map", dest="map_name", default="Town04", help="맵 이름 (예: Town04, Town06)")
    p.add_argument(
        "--weather",
        dest="weather_name",
        default="ClearNoon",
        help="날씨 프리셋 (예: ClearNoon, CloudyNoon, WetNoon)",
    )
    p.add_argument(
        "--fixed-dt",
        type=float,
        default=0.08,
        help="동기 시뮬 틱 간격(s). 기본 0.08은 부하·물리 균형. 최종 렌더는 동일 값 유지 권장.",
    )
    p.add_argument("--speed-kmh", type=float, default=60.0)
    p.add_argument("--ab-m", type=float, default=30.0)
    p.add_argument("--bc-m", type=float, default=20.0)
    p.add_argument("--a-brake-t", type=float, default=3.0)
    p.add_argument("--max-ticks", type=int, default=12000)
    p.add_argument(
        "--img-w",
        type=int,
        default=1280,
        help="RGB 너비(px). 확인용 기본 1280, 최종 포스터는 1920 권장.",
    )
    p.add_argument(
        "--img-h",
        type=int,
        default=720,
        help="RGB 높이(px). 확인용 기본 720, 최종 포스터는 1080 권장.",
    )
    p.add_argument("--cam-z", type=float, default=18.0)
    p.add_argument("--cam-fov", type=float, default=90.0)
    p.add_argument("--cam-retry", type=int, default=5)
    p.add_argument(
        "--compare-anchor",
        choices=["bc_lt", "c_brake"],
        default="bc_lt",
        help="compare_shot의 T* 기준. bc_lt=BC가 임계값보다 작아지는 시각(포스터 임팩트), c_brake=C brake>0.05 시각(원안)",
    )
    p.add_argument(
        "--compare-bc-lt-m",
        type=float,
        default=8.0,
        help="compare-anchor=bc_lt일 때 scenario1에서 BC < X(m) 최초 시각을 T*로 사용",
    )
    p.add_argument(
        "--compare-lock-camera",
        type=int,
        default=1,
        help="1이면 compare_shot에서 scenario2가 scenario1의 카메라 구도(x,y,yaw,pitch,fov)를 최대한 유지",
    )
    args = p.parse_args()

    # CARLA 연결 + 맵/날씨 1회 로드
    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)
    if args.map_name:
        client.load_world(str(args.map_name))
        # load_world는 비동기로 로드될 수 있어, 충분히 대기 후 world 참조 갱신
        # (wait_for_tick은 sync/async 상태에 따라 동작이 달라 불안정)
        time.sleep(3.0)
    world = client.get_world()
    if args.weather_name:
        world.set_weather(_weather_from_name(carla, str(args.weather_name)))

    scenario1 = ScenarioSpec(
        name="scenario1",
        v2v_enabled=False,
        b_reaction_s=1.5,
        c_reaction_s=1.5,
        c_reaction_relative_to="B",
    )
    scenario2 = ScenarioSpec(
        name="scenario2",
        v2v_enabled=True,
        b_reaction_s=0.3,
        c_reaction_s=0.3,
        c_reaction_relative_to="A",
    )

    out_root = str(args.output_root)
    out1_dir = os.path.join(out_root, "scenario1")
    out2_dir = os.path.join(out_root, "scenario2")
    _ensure_dir(out1_dir)
    _ensure_dir(out2_dir)

    # 1) scenario1 실행 → T* 확보
    s1 = run_one_scenario(
        scenario1,
        world,
        out1_dir,
        args.fixed_dt,
        args.speed_kmh,
        args.ab_m,
        args.bc_m,
        args.a_brake_t,
        args.max_ticks,
        args.img_w,
        args.img_h,
        args.cam_z,
        args.cam_fov,
        args.cam_retry,
        compare_t_star_s=None,
        compare_anchor=str(args.compare_anchor),
        compare_bc_lt_m=float(args.compare_bc_lt_m),
        compare_lock_camera=bool(int(args.compare_lock_camera)),
        compare_camera_from_s1=None,
    )
    print("[poster] scenario1 complete, physics settle …", flush=True)

    # 시나리오1 종료 직후 물리/스폰 정리 시간을 조금 준다.
    # async 모드에서는 tick이 의미 없을 수 있어 잠깐 sync로 돌렸다가 원복한다.
    try:
        tmp = world.get_settings()
        tmp.synchronous_mode = True
        tmp.fixed_delta_seconds = float(args.fixed_dt)
        world.apply_settings(tmp)
        for _ in range(5):
            world.tick()
    finally:
        try:
            tmp2 = world.get_settings()
            tmp2.synchronous_mode = False
            world.apply_settings(tmp2)
        except Exception:
            pass
    # 2) scenario2 실행 (scenario1에서 얻은 T*로 compare_shot 동기 캡처)
    _t_star = s1.t_star_compare_s
    if _t_star is None:
        print(
            "[warn] scenario1에서 compare T*를 찾지 못했습니다. "
            "compare_shot은 생성되지 않을 수 있습니다. "
            "X(--compare-bc-lt-m)를 키우거나 조건을 완화하세요.",
            flush=True,
        )
    print("[poster] starting scenario2 …", flush=True)
    s2 = run_one_scenario(
        scenario2,
        world,
        out2_dir,
        args.fixed_dt,
        args.speed_kmh,
        args.ab_m,
        args.bc_m,
        args.a_brake_t,
        args.max_ticks,
        args.img_w,
        args.img_h,
        args.cam_z,
        args.cam_fov,
        args.cam_retry,
        compare_t_star_s=_t_star,
        compare_anchor=str(args.compare_anchor),
        compare_bc_lt_m=float(args.compare_bc_lt_m),
        compare_lock_camera=bool(int(args.compare_lock_camera)),
        compare_camera_from_s1=s1.compare_camera,
    )

    # 상위 요약
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"scenario1": asdict(s1), "scenario2": asdict(s2)},
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()

