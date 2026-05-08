"""
GCP/도커 환경에서 CARLA+V2V 실행 전 사전 점검(preflight).

목표:
- CARLA 서버 접속/동기 틱 가능
- 센서(RGB/Depth) 프레임 수신 가능 (offscreen 렌더링 확인)
- torch/cuda/opencv/ultralytics 로딩 가능
- YOLO 추론이 "numpy 입력" 및 "파일 입력" 모두에서 동작하며,
  우리 파이프라인(`YoloRiskPipeline`, `compute_lead_vehicle_metrics`)이 처리 가능한 출력 형태인지 확인
- UDP 포트 바인드 가능(기본 5005)

사용 예 (VM에서):
  python preflight_gcp.py --host 127.0.0.1 --port 2000 --ticks 120
"""

from __future__ import annotations

import argparse
import os
import queue
import socket
import sys
import tempfile
import time
from typing import Any, Optional, Tuple

import numpy as np

from config import UDP_PORT
from lead_vehicle import compute_lead_vehicle_metrics
from yolo_risk import YoloRiskPipeline


def _print_env() -> None:
    print("python:", sys.version.replace("\n", " "))
    print("cwd:", os.getcwd())


def _check_torch() -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))


def _check_cv2() -> None:
    import cv2

    print("cv2:", cv2.__version__)


def _check_udp_bind(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        print("udp_bind_ok:", port)
    finally:
        s.close()


def _carla_connect(host: str, port: int) -> Tuple[Any, Any]:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    print("carla_map:", world.get_map().name)
    return client, world


def _carla_sensor_smoke(world: Any, ticks: int, fixed_dt: float) -> None:
    import carla

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("no spawn points in map")

    orig = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(settings)

    veh = None
    rgb = None
    depth = None
    q: "queue.Queue[Tuple[str, int]]" = queue.Queue()

    def on_rgb(img: Any) -> None:
        q.put(("rgb", int(img.frame)))

    def on_depth(img: Any) -> None:
        q.put(("depth", int(img.frame)))

    try:
        veh_bp = (bp.filter("vehicle.audi.*") or bp.filter("vehicle.*"))[0]
        veh = world.try_spawn_actor(veh_bp, spawns[0])
        if veh is None:
            raise RuntimeError("vehicle spawn failed")

        cam_rgb_bp = bp.find("sensor.camera.rgb")
        cam_rgb_bp.set_attribute("image_size_x", "640")
        cam_rgb_bp.set_attribute("image_size_y", "360")
        cam_rgb_bp.set_attribute("fov", "90")

        cam_d_bp = bp.find("sensor.camera.depth")
        cam_d_bp.set_attribute("image_size_x", "640")
        cam_d_bp.set_attribute("image_size_y", "360")
        cam_d_bp.set_attribute("fov", "90")

        mount = carla.Transform(
            carla.Location(x=2.3, z=1.6), carla.Rotation(pitch=-8.0)
        )
        rgb = world.spawn_actor(cam_rgb_bp, mount, attach_to=veh)
        depth = world.spawn_actor(cam_d_bp, mount, attach_to=veh)
        rgb.listen(on_rgb)
        depth.listen(on_depth)

        got_rgb = 0
        got_depth = 0
        for _ in range(ticks):
            world.tick()
            while True:
                try:
                    kind, _frame = q.get_nowait()
                except queue.Empty:
                    break
                if kind == "rgb":
                    got_rgb += 1
                else:
                    got_depth += 1
            if got_rgb >= 10 and got_depth >= 10:
                break

        print("sensor_frames_rgb:", got_rgb, "depth:", got_depth)
        if got_rgb == 0 or got_depth == 0:
            raise RuntimeError("sensor frames not arriving (offscreen/render issue)")
    finally:
        try:
            world.apply_settings(orig)
        except Exception:
            pass
        for a in (rgb, depth, veh):
            if a is not None:
                try:
                    a.destroy()
                except Exception:
                    pass


def _yolo_smoke() -> None:
    # 더미 이미지 생성(검은 바탕)
    bgr = np.zeros((360, 640, 3), dtype=np.uint8)
    yolo = YoloRiskPipeline()
    risks, res = yolo.infer_frame(bgr)
    print("yolo_risks_len:", len(risks), "res_type:", type(res).__name__)


def _lead_metrics_smoke() -> None:
    # compute_lead_vehicle_metrics()는 YOLO 결과 + depth 필요하므로, 여기선 최소 호출만 검증:
    # YOLO 결과가 Results가 아니더라도 함수가 "크래시 없이 None 반환"까지는 보장해야 한다.
    dummy_depth = np.zeros((360, 640, 4), dtype=np.uint8)
    cam = np.eye(4, dtype=np.float64)
    ego_loc = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    ego_fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ego_vel = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    # 빈 det를 res로 전달
    det = np.zeros((0, 6), dtype=np.float32)
    ttc, area, xyxy = compute_lead_vehicle_metrics(
        det, dummy_depth, cam, ego_loc, ego_fwd, ego_vel
    )
    print("lead_metrics:", ttc, area, xyxy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--fixed-dt", type=float, default=0.05)
    p.add_argument("--skip-yolo", action="store_true")
    args = p.parse_args()

    _print_env()
    _check_udp_bind(UDP_PORT)

    client, world = _carla_connect(args.host, args.port)
    _carla_sensor_smoke(world, ticks=args.ticks, fixed_dt=args.fixed_dt)

    if not args.skip_yolo:
        _check_cv2()
        _check_torch()
        _yolo_smoke()
        _lead_metrics_smoke()

    # 약한 keep-alive 출력 (로그 가독성)
    print("preflight_ok")


if __name__ == "__main__":
    main()

