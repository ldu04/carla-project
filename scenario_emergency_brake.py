"""
CARLA 급정거 시나리오: 전방 장애물 차량 스폰 → 거리 조건에서 장애물 급제동,
동기 모드에서 대형차·승용 제어, 선택적 V2V(송신 추론 + 동일 프로세스 UDP 수신 제동).

실행 (CARLA 서버 가동 후):
  python scenario_emergency_brake.py --speed-kmh 45 --headway-m 28 --trigger-m 40 --v2v 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from brake_detector import EmergencyBrakeDetector
from config import (
    CAMERA_FOV_DEG,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CARLA_HOST,
    CARLA_PORT,
    EMERGENCY_BBOX_AREA_RATE_THRESHOLD,
    EMERGENCY_DETECTION_COOLDOWN_S,
    EMERGENCY_DTTC_DT_THRESHOLD,
    EMERGENCY_TTC_WINDOW_FRAMES,
    LOG_DIR,
    SCENARIO_HEADWAY_M,
    SCENARIO_INITIAL_SPEED_KMH,
    SCENARIO_OBSTACLE_AHEAD_M,
    SCENARIO_TRIGGER_DISTANCE_M,
    SCENARIO_V2V_ENABLED,
    UDP_PORT,
    YOLO_TARGET_HZ,
)
from receiver_brake_state import ReceiverBrakeController
from sender import (
    LatestPair,
    carla_depth_to_bgra,
    carla_image_to_bgr,
    carla_inference_tick,
    make_udp_socket,
)
from v2v_logger import log_scenario_result, setup_logger
from yolo_risk import YoloRiskPipeline

logger = setup_logger("scenario")


class LatestCarlaImage:
    """센서 콜백에서 최신 carla.Image 1장만 보관."""

    def __init__(self) -> None:
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


def _save_image(img: Any, out_path: str) -> None:
    """
    carla.Image를 디스크로 저장.
    기본은 PNG로 저장되며, offscreen에서도 동작한다.
    """
    # ColorConverter는 import carla 이후에만 접근 가능하므로 동적 참조
    try:
        import carla

        img.save_to_disk(out_path, carla.ColorConverter.Raw)
    except Exception:
        # 최후 폴백(변환 실패 시): Raw 그대로 저장 시도
        img.save_to_disk(out_path)


def _capture_event_photos(
    label: str,
    sim_time: float,
    out_dir: str,
    cams: Dict[str, LatestCarlaImage],
) -> None:
    _ensure_dir(out_dir)
    safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_"))
    t_ms = int(round(sim_time * 1000.0))
    for name, buf in cams.items():
        img, frame = buf.get()
        if img is None:
            continue
        fn = f"{t_ms:010d}_{safe_label}_{name}_frame{frame if frame is not None else -1}.png"
        _save_image(img, os.path.join(out_dir, fn))


@dataclass
class ScenarioParams:
    initial_speed_kmh: float = SCENARIO_INITIAL_SPEED_KMH
    headway_m: float = SCENARIO_HEADWAY_M
    emergency_trigger_distance_m: float = SCENARIO_TRIGGER_DISTANCE_M
    obstacle_ahead_m: float = SCENARIO_OBSTACLE_AHEAD_M
    v2v_enabled: bool = SCENARIO_V2V_ENABLED
    max_ticks: int = 8000
    fixed_delta_seconds: float = 0.05


@dataclass
class ScenarioResult:
    collision: bool = False
    delta_v_ms: Optional[float] = None
    ttc_at_rear_brake_start: Optional[float] = None
    perception_delay_s: Optional[float] = None
    first_emergency_detect_wall_s: Optional[float] = None
    first_rear_brake_wall_s: Optional[float] = None
    ticks_ran: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


def _forward_offset_tf(base: Any, forward_m: float, carla: Any) -> Any:
    """base 변환에서 전방으로 forward_m 만큼 이동한 Transform."""
    f = base.get_forward_vector()
    loc = carla.Location(
        base.location.x + f.x * forward_m,
        base.location.y + f.y * forward_m,
        base.location.z + f.z * forward_m,
    )
    return carla.Transform(loc, base.rotation)


def _set_actor_velocityAlong_heading(actor: Any, speed_ms: float, carla: Any) -> None:
    t = actor.get_transform()
    f = t.get_forward_vector()
    # CARLA Vector3D 정규화
    mag = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z) or 1.0
    actor.set_velocity(
        carla.Vector3D(
            f.x / mag * speed_ms, f.y / mag * speed_ms, f.z / mag * speed_ms
        )
    )


def _speed_ms(actor: Any) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _cruise_control(target_ms: float, current_ms: float) -> Tuple[float, float]:
    """
    매우 단순한 속도 유지 제어.
    - overshoot 시 약한 brake
    - undershoot 시 throttle
    """
    err = target_ms - current_ms
    if err >= 0:
        throttle = min(0.75, max(0.0, 0.35 + err * 0.05))
        return throttle, 0.0
    brake = min(0.45, max(0.0, (-err) * 0.06))
    return 0.0, brake


def _drain_udp(sock: socket.socket) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            out.append(json.loads(data.decode("utf-8")))
        except BlockingIOError:
            break
        except json.JSONDecodeError:
            continue
    return out


def run_scenario(params: ScenarioParams) -> ScenarioResult:
    import carla

    res = ScenarioResult(params=asdict(params))
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = params.fixed_delta_seconds
    world.apply_settings(settings)

    map_spawn = world.get_map().get_spawn_points()
    if not map_spawn:
        raise RuntimeError("no spawn points in current CARLA map")

    def _pick_blueprint(filters: List[str]) -> Any:
        for f in filters:
            bps = bp_lib.filter(f)
            if bps:
                return bps[0]
        # 최후 폴백
        return bp_lib.filter("vehicle.*")[0]

    def _try_spawn_with_jitter(bp: Any, tf: Any) -> Optional[Any]:
        # 약간의 높이 오프셋으로 지면/프레임 충돌을 완화
        base_tf = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.8),
            tf.rotation,
        )
        for i in range(20):
            jit = carla.Location(x=0.9 * (i % 4), y=0.9 * (i // 4), z=0.0)
            t2 = carla.Transform(base_tf.location + jit, base_tf.rotation)
            a = world.try_spawn_actor(bp, t2)
            if a is not None:
                return a
        return None

    def spawn_triplet() -> Tuple[Any, Any, Any, Any]:
        """
        (truck, obstacle, sedan, truck_tf) 를 스폰.
        - 트럭은 'carlamotors'가 없을 수 있어 truck 계열 필터를 우선으로 후보화
        - 스폰 포인트도 순회하며 충돌이 적은 지점을 찾는다
        """
        truck_bp = _pick_blueprint(
            [
                "vehicle.*truck*",
                "vehicle.carlamotors.*",
                "vehicle.*",
            ]
        )
        obstacle_bp = _pick_blueprint(["vehicle.tesla.model3", "vehicle.*"])
        sedan_bp = _pick_blueprint(["vehicle.audi.*", "vehicle.*"])

        # 시작점 후보는 여러 개를 시도 (맵/교통 상황 따라 특정 포인트가 막힐 수 있음)
        for idx in range(min(len(map_spawn), 24)):
            base_tf = map_spawn[idx]
            truck = _try_spawn_with_jitter(truck_bp, base_tf)
            if truck is None:
                continue
            obs_tf = _forward_offset_tf(base_tf, params.obstacle_ahead_m, carla)
            obstacle = _try_spawn_with_jitter(obstacle_bp, obs_tf)
            if obstacle is None:
                truck.destroy()
                continue
            sedan_tf = _forward_offset_tf(base_tf, -params.headway_m, carla)
            sedan = _try_spawn_with_jitter(sedan_bp, sedan_tf)
            if sedan is None:
                obstacle.destroy()
                truck.destroy()
                continue
            return truck, obstacle, sedan, base_tf

        raise RuntimeError("vehicle spawn failed: could not place truck/obstacle/sedan")

    truck, obstacle, sedan, truck_tf = spawn_triplet()

    speed_ms = params.initial_speed_kmh / 3.6
    # set_velocity()는 물리를 무시할 수 있으므로 사용하지 않고,
    # 초기 구간에서 apply_control을 지속 적용해 목표 속도로 수렴시킨다.

    recv_udp: Optional[socket.socket] = None
    brake_ctrl: Optional[ReceiverBrakeController] = None
    eb_detector: Optional[EmergencyBrakeDetector] = None
    yolo: Optional[YoloRiskPipeline] = None
    udp_send_sock: Optional[socket.socket] = None
    latest = LatestPair()
    rgb_actor = depth_actor = None

    # 사진 캡처(옵션): important event 순간에만 저장
    record_photos = bool(params.params.get("record_photos")) if isinstance(getattr(params, "params", None), dict) else False

    # 동기 모드에서는 벽시계 기반으로 rate-limit 하면 재현성이 깨질 수 있어
    # tick 기반으로 추론 주기를 결정한다.
    infer_every_n = max(1, int(round(1.0 / (max(YOLO_TARGET_HZ, 0.1) * params.fixed_delta_seconds))))
    infer_dt_s = infer_every_n * params.fixed_delta_seconds

    # --- 이벤트 사진용 멀티캠 준비 ---
    photo_cams: Dict[str, LatestCarlaImage] = {}
    photo_actors: List[Any] = []
    photo_out_dir: Optional[str] = None
    if getattr(params, "record_photos", False):
        photo_out_dir = str(getattr(params, "photo_out_dir", os.path.join(LOG_DIR, "photos")))
        photo_w = int(getattr(params, "photo_width", 1280))
        photo_h = int(getattr(params, "photo_height", 720))
        photo_fov = float(getattr(params, "photo_fov", CAMERA_FOV_DEG))
        # 고화질 멀티캠은 렌더 부담이 크므로 sensor_tick으로 캡처 주기를 낮춰 server stall을 방지
        photo_sensor_tick = float(getattr(params, "photo_sensor_tick", 0.2))
        want = list(getattr(params, "photo_cams", ["front", "driver", "top"]))

        def spawn_photo_cam(name: str, tf: Any) -> None:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(photo_w))
            bp.set_attribute("image_size_y", str(photo_h))
            bp.set_attribute("fov", str(photo_fov))
            bp.set_attribute("sensor_tick", str(photo_sensor_tick))
            buf = LatestCarlaImage()
            a = world.spawn_actor(bp, tf, attach_to=truck)
            a.listen(buf.set)
            photo_cams[name] = buf
            photo_actors.append(a)

        if "front" in want:
            spawn_photo_cam(
                "front",
                carla.Transform(
                    carla.Location(x=2.3, z=1.6), carla.Rotation(pitch=-8.0)
                ),
            )
        if "driver" in want:
            spawn_photo_cam(
                "driver",
                carla.Transform(carla.Location(x=0.4, z=1.3), carla.Rotation()),
            )
        if "top" in want:
            spawn_photo_cam(
                "top",
                carla.Transform(
                    carla.Location(x=-8.0, z=6.0),
                    carla.Rotation(pitch=-20.0),
                ),
            )

    if params.v2v_enabled:
        udp_send_sock = make_udp_socket()
        recv_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_udp.bind(("", UDP_PORT))
        recv_udp.setblocking(False)
        brake_ctrl = ReceiverBrakeController()
        eb_detector = EmergencyBrakeDetector(
            EMERGENCY_TTC_WINDOW_FRAMES,
            EMERGENCY_DTTC_DT_THRESHOLD,
            EMERGENCY_BBOX_AREA_RATE_THRESHOLD,
            infer_dt_s,
            EMERGENCY_DETECTION_COOLDOWN_S,
        )
        yolo = YoloRiskPipeline()

        cam_rgb_bp = bp_lib.find("sensor.camera.rgb")
        cam_rgb_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        cam_rgb_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        cam_rgb_bp.set_attribute("fov", str(CAMERA_FOV_DEG))
        cam_depth_bp = bp_lib.find("sensor.camera.depth")
        cam_depth_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        cam_depth_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        cam_depth_bp.set_attribute("fov", str(CAMERA_FOV_DEG))
        cam_mount = carla.Transform(
            carla.Location(x=2.3, z=1.6), carla.Rotation(pitch=-8.0)
        )

        def on_rgb(img: Any) -> None:
            latest.set_rgb(carla_image_to_bgr(img))

        def on_depth(img: Any) -> None:
            latest.set_depth(carla_depth_to_bgra(img))

        rgb_actor = world.spawn_actor(cam_rgb_bp, cam_mount, attach_to=truck)
        depth_actor = world.spawn_actor(cam_depth_bp, cam_mount, attach_to=truck)
        rgb_actor.listen(on_rgb)
        depth_actor.listen(on_depth)

    try:
        sedan_vel_before = None
        obs_vel_before = None
        first_emergency: Optional[float] = None
        first_rear_brake: Optional[float] = None
        last_pkt_ttc_rear: Optional[float] = None
        trigger_armed = False
        obstacle_braking = False

        # 워밍업: 목표 속도로 수렴 (센서·YOLO 없이 물리만 안정화)
        warmup_ticks = int(round(2.0 / max(params.fixed_delta_seconds, 1e-6)))
        for _ in range(max(20, warmup_ticks)):
            world.tick()
            for a in (truck, obstacle, sedan):
                cur = _speed_ms(a)
                th, br = _cruise_control(speed_ms, cur)
                a.apply_control(carla.VehicleControl(throttle=th, brake=br))

        for tick_idx in range(params.max_ticks):
            world.tick()
            res.ticks_ran = tick_idx + 1
            snap = world.get_snapshot()
            sim_time = float(snap.timestamp.elapsed_seconds)

            dist_to_obs = truck.get_location().distance(obstacle.get_location())
            if not trigger_armed and dist_to_obs < params.emergency_trigger_distance_m:
                trigger_armed = True
                obstacle_braking = True
                logger.info("장애물 급제동 트리거 dist=%.2f", dist_to_obs)
                if photo_out_dir and photo_cams:
                    _capture_event_photos("trigger", sim_time, photo_out_dir, photo_cams)

            # 차량 제어는 tick마다 지속 적용 (apply_control 1회로는 다음 tick에 덮일 수 있음)
            if obstacle_braking:
                obstacle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
                )
            else:
                cur = _speed_ms(obstacle)
                th, br = _cruise_control(speed_ms, cur)
                obstacle.apply_control(
                    carla.VehicleControl(throttle=th, brake=br, hand_brake=False)
                )

            # 트럭은 시나리오 동안 목표 속도를 유지
            cur_t = _speed_ms(truck)
            th_t, br_t = _cruise_control(speed_ms, cur_t)
            truck.apply_control(carla.VehicleControl(throttle=th_t, brake=br_t))

            if params.v2v_enabled and recv_udp and brake_ctrl and yolo and eb_detector and udp_send_sock:
                rgb, depth_bgra = latest.get_copy()
                if (
                    rgb is not None
                    and depth_bgra is not None
                    and (tick_idx % infer_every_n == 0)
                ):
                    t_em = carla_inference_tick(
                        rgb,
                        depth_bgra,
                        truck,
                        rgb_actor,
                        yolo,
                        eb_detector,
                        udp_send_sock,
                        sim_time,
                    )
                    if t_em is not None and first_emergency is None:
                        first_emergency = t_em
                        if photo_out_dir and photo_cams:
                            _capture_event_photos("detect", sim_time, photo_out_dir, photo_cams)

                pkts = _drain_udp(recv_udp)
                for p in pkts:
                    if p.get("risk_type") == "emergency_brake":
                        last_pkt_ttc_rear = float(p.get("ttc", 0.0))
                # 수신 측 시간도 simulation time으로 통일 (monotonic/wallclock 혼용 방지)
                brake_ctrl.ingest_packets(pkts, sim_time, sim_time)
                brake_ctrl.step_vehicle(sedan, carla, sim_time)
                if (
                    first_rear_brake is None
                    and brake_ctrl.brake_cmd > 0.05
                ):
                    first_rear_brake = sim_time
                    res.ttc_at_rear_brake_start = last_pkt_ttc_rear
                    if photo_out_dir and photo_cams:
                        _capture_event_photos("rear_brake", sim_time, photo_out_dir, photo_cams)
            else:
                # V2V 미사용 시 뒤차도 목표 속도 유지
                cur_s = _speed_ms(sedan)
                th_s, br_s = _cruise_control(speed_ms, cur_s)
                sedan.apply_control(carla.VehicleControl(throttle=th_s, brake=br_s))

            # 충돌 근사: 승용·장애물 거리
            d_coll = sedan.get_location().distance(obstacle.get_location())
            if d_coll < 3.0:
                sv = sedan.get_velocity()
                ov = obstacle.get_velocity()
                sedan_vel_before = math.sqrt(sv.x ** 2 + sv.y ** 2 + sv.z ** 2)
                obs_vel_before = math.sqrt(ov.x ** 2 + ov.y ** 2 + ov.z ** 2)
                res.collision = True
                res.delta_v_ms = abs(sedan_vel_before - obs_vel_before)
                if photo_out_dir and photo_cams:
                    _capture_event_photos("collision", sim_time, photo_out_dir, photo_cams)
                break

        if photo_out_dir and photo_cams:
            _capture_event_photos("end", sim_time, photo_out_dir, photo_cams)

        # 결과 시간도 simulation time(초) 기준
        res.first_emergency_detect_wall_s = first_emergency
        res.first_rear_brake_wall_s = first_rear_brake
        if first_emergency is not None and first_rear_brake is not None:
            res.perception_delay_s = max(0.0, first_rear_brake - first_emergency)

        log_scenario_result(logger, asdict(res))
        return res

    finally:
        # 서버가 stall/timeout 상태이면 apply_settings도 timeout이 날 수 있어 보호
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except RuntimeError:
            pass
        if rgb_actor is not None:
            rgb_actor.destroy()
        if depth_actor is not None:
            depth_actor.destroy()
        for a in photo_actors:
            try:
                # sensor는 stop 후 destroy (경고/잔존 방지)
                a.stop()
                a.destroy()
            except Exception:
                pass
        if recv_udp is not None:
            recv_udp.close()
        if udp_send_sock is not None:
            udp_send_sock.close()
        for a in (obstacle, sedan, truck):
            try:
                a.destroy()
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--speed-kmh", type=float, default=SCENARIO_INITIAL_SPEED_KMH)
    p.add_argument("--headway-m", type=float, default=SCENARIO_HEADWAY_M)
    p.add_argument("--trigger-m", type=float, default=SCENARIO_TRIGGER_DISTANCE_M)
    p.add_argument("--obstacle-ahead-m", type=float, default=SCENARIO_OBSTACLE_AHEAD_M)
    p.add_argument("--v2v", type=int, default=1, help="1=V2V 파이프라인 포함")
    p.add_argument("--max-ticks", type=int, default=8000)
    p.add_argument("--record-photos", type=int, default=0, help="1=중요 시점 사진 저장")
    p.add_argument("--photo-out-dir", type=str, default=os.path.join(LOG_DIR, "photos"))
    p.add_argument("--photo-width", type=int, default=1280)
    p.add_argument("--photo-height", type=int, default=720)
    p.add_argument("--photo-sensor-tick", type=float, default=0.2)
    p.add_argument("--photo-cams", type=str, default="front,driver,top")
    args = p.parse_args()

    sp = ScenarioParams(
        initial_speed_kmh=args.speed_kmh,
        headway_m=args.headway_m,
        emergency_trigger_distance_m=args.trigger_m,
        obstacle_ahead_m=args.obstacle_ahead_m,
        v2v_enabled=bool(args.v2v),
        max_ticks=args.max_ticks,
    )
    # dataclass 확장 없이 런타임 속성으로 사진 옵션 전달
    setattr(sp, "record_photos", bool(args.record_photos))
    setattr(sp, "photo_out_dir", str(args.photo_out_dir))
    setattr(sp, "photo_width", int(args.photo_width))
    setattr(sp, "photo_height", int(args.photo_height))
    setattr(sp, "photo_sensor_tick", float(args.photo_sensor_tick))
    cams = [c.strip() for c in str(args.photo_cams).split(",") if c.strip()]
    setattr(sp, "photo_cams", cams)
    setattr(sp, "photo_fov", float(CAMERA_FOV_DEG))
    out = run_scenario(sp)
    outp = os.path.join(LOG_DIR, "scenario_last_result.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(asdict(out), f, ensure_ascii=False, indent=2)
    logger.info("결과 저장 %s", outp)


if __name__ == "__main__":
    main()
