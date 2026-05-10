"""
CARLA 포스터용 스토리보드: 정적 배치(간격만 변경)로 7개 컷을 순서대로 촬영.

요구사항:
- carla_shot.py의 로직을 재사용(_spawn_abc_same_lane, _compute_camera_tf 등)
- load_world는 최초 1회만
- 각 컷마다 차량 destroy 후 재스폰
- 실패한 컷은 건너뛰고 다음 컷 진행 (전체가 죽으면 안 됨)
- 컷 번호와 저장 경로를 콘솔에 출력

Shots (ab/bc는 m):
공통 3장:
  1) shot1_cruise.png        ab=30 bc=20
  2) shot2_a_brake.png       ab=15 bc=20
  3) shot3_b_brake.png       ab=10 bc=12
기기 없음 2장:
  4) shot4_no_device.png     ab=10 bc=6
  5) shot5_collision.png     ab=8  bc=2
기기 있음 2장:
  6) shot4_with_device.png   ab=10 bc=14
  7) shot5_safe.png          ab=8  bc=12
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, List, Optional, Tuple

from config import CARLA_HOST, CARLA_PORT

# carla_shot.py 내부 로직 재사용 (Python 3.7 호환)
from carla_shot import (  # noqa: F401
    _count_abc_outside_frame,
    _compute_camera_tf,
    _spawn_abc_same_lane,
    _spawn_camera,
    _wait_for_frame,
)


ShotSpec = Tuple[str, float, float, Optional[float], Optional[float]]  # (filename, ab_m, bc_m, cam_z, cam_offset_back)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _destroy_quiet(actor: Any) -> None:
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def _run_one_shot(
    *,
    world: Any,
    carla: Any,
    bp_lib: Any,
    out_path: str,
    ab_m: float,
    bc_m: float,
    fixed_dt: float,
    stable_ticks: int,
    img_w: int,
    img_h: int,
    fov: float,
    cam_z: float,
    cam_offset_back: float,
) -> bool:
    """
    1) A/B/C 스폰(같은 lane)
    2) 카메라 스폰(월드 고정)
    3) 안정화 tick N회
    4) 안정화 후 카메라 TF 재계산+적용
    5) 다음 tick에서 프레임 정합 확인 후 저장
    return: 성공 여부
    """
    cam = None
    a = b = c = None
    try:
        print(f"[shot] spawn abc ab={ab_m} bc={bc_m} -> {out_path}", flush=True)
        a, b, c = _spawn_abc_same_lane(world, carla, float(ab_m), float(bc_m))

        cam, buf = _spawn_camera(world, bp_lib, int(img_w), int(img_h), float(fov))

        # 안정화 tick
        for _ in range(int(stable_ticks)):
            world.tick()

        # 안정화 후 위치 기준 카메라 재계산
        cam_tf = _compute_camera_tf(
            carla, a, b, c, z=float(cam_z), back_offset_m=float(cam_offset_back)
        )
        cam.set_transform(cam_tf)

        # 촬영 tick (0.9.13에서는 world.tick()이 frame id(int)를 반환하는 빌드가 있음)
        frame_id = world.tick()
        snap = world.get_snapshot()
        target_frame = int(getattr(snap, "frame", frame_id))
        if not _wait_for_frame(buf, target_frame, timeout_s=2.0):
            print(f"[WARN] frame wait timeout (frame={target_frame})", flush=True)

        outside, total = _count_abc_outside_frame(
            carla, cam_tf, int(img_w), int(img_h), float(fov), a, b, c
        )
        if outside > 0:
            print(
                f"[WARN] bbox corners outside frame: {outside}/{total} "
                f"(ab={ab_m} bc={bc_m} z={cam_z} fov={fov})",
                flush=True,
            )

        img, _fr = buf.get()
        if img is None:
            raise RuntimeError("no image received from camera")

        _ensure_dir(os.path.dirname(os.path.abspath(out_path)))
        img.save_to_disk(str(out_path), carla.ColorConverter.Raw)
        print(f"[ok] saved {out_path}", flush=True)
        return True

    except Exception as e:
        print(f"[FAIL] {out_path}: {type(e).__name__}: {e}", flush=True)
        return False
    finally:
        try:
            if cam is not None:
                cam.stop()
                cam.destroy()
        except Exception:
            pass
        for actor in (a, b, c):
            _destroy_quiet(actor)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/output/storyboard")
    p.add_argument("--host", default=CARLA_HOST)
    p.add_argument("--port", type=int, default=int(CARLA_PORT))
    p.add_argument("--map", default="Town04")
    p.add_argument("--fixed-dt", type=float, default=0.05)
    p.add_argument("--stable-ticks", type=int, default=10)
    p.add_argument("--img-w", type=int, default=1920)
    p.add_argument("--img-h", type=int, default=1080)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--cam-z", type=float, default=20.0)
    p.add_argument("--cam-offset-back", type=float, default=40.0)
    args = p.parse_args()

    import carla

    out_dir = os.path.abspath(str(args.output_dir))
    _ensure_dir(out_dir)

    # 좁은 간격(겹침에 가까운 값)은 스폰 자체가 물리적으로 거부될 수 있음.
    # 포스터 목적상 "위험해 보이는" 앵글로 보완하고, 간격은 안전한 값으로 둔다.
    shots: List[ShotSpec] = [
        ("shot1_cruise.png", 30.0, 20.0, None, None),
        ("shot2_a_brake.png", 15.0, 20.0, None, None),
        ("shot3_b_brake.png", 10.0, 12.0, None, None),
        ("shot4_no_device.png", 10.0, 10.0, 12.0, 25.0),  # bc 6→10, 앵글로 위험감 강화
        ("shot5_collision.png", 8.0, 8.0, 12.0, 25.0),    # bc 2→8, 앵글로 근접 강조
        ("shot4_with_device.png", 10.0, 14.0, None, None),
        ("shot5_safe.png", 8.0, 12.0, None, None),
    ]

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(60.0)
    client.load_world(str(args.map))
    time.sleep(3.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    settings = world.get_settings()
    try:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_dt)
        world.apply_settings(settings)

        ok_count = 0
        for idx, (fname, ab_m, bc_m, cam_z_override, cam_back_override) in enumerate(shots, start=1):
            out_path = os.path.join(out_dir, fname)
            print(f"\n=== Shot {idx}/7: {fname} ===", flush=True)
            _cz = float(args.cam_z) if cam_z_override is None else float(cam_z_override)
            _cb = (
                float(args.cam_offset_back)
                if cam_back_override is None
                else float(cam_back_override)
            )
            print(f"[shot_cfg] ab={ab_m} bc={bc_m} cam_z={_cz} cam_back={_cb}", flush=True)
            ok = _run_one_shot(
                world=world,
                carla=carla,
                bp_lib=bp_lib,
                out_path=out_path,
                ab_m=float(ab_m),
                bc_m=float(bc_m),
                fixed_dt=float(args.fixed_dt),
                stable_ticks=int(args.stable_ticks),
                img_w=int(args.img_w),
                img_h=int(args.img_h),
                fov=float(args.fov),
                cam_z=_cz,
                cam_offset_back=_cb,
            )
            ok_count += 1 if ok else 0

            # 컷 사이 물리 정리 시간(스폰 충돌/잔상 완화)
            for _ in range(3):
                world.tick()

        print(f"\n[done] storyboard saved to {out_dir} (ok={ok_count}/7)", flush=True)

    finally:
        try:
            s2 = world.get_settings()
            s2.synchronous_mode = False
            world.apply_settings(s2)
        except Exception:
            pass


if __name__ == "__main__":
    main()

