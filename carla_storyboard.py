"""
CARLA 포스터용 스토리보드: 정적 배치(간격만 변경)로 7개 컷을 순서대로 촬영.

요구사항:
- carla_shot.py의 로직을 재사용(_spawn_abc_same_lane, _compute_camera_tf 등)
- load_world는 최초 1회만
- 각 컷마다 차량 destroy 후 재스폰
- 실패한 컷은 건너뛰고 다음 컷 진행 (전체가 죽으면 안 됨)
- 컷 번호와 저장 경로를 콘솔에 출력

간격(ab/bc)을 바꿀 때: "A 급정지 → B 급정지 → C 반응" 순으로 장면이 이어질 때
각 컷의 ab·bc가 물리적으로 자연스럽게 변하는지 먼저 확인한다
(예: 동일 shot5 시점에서 기기 있음이면 B가 일찍 반응했으면 collision 컷보다 AB 여유가 커야 함).
shot3→shot4는 연속 장면이므로 AB가 시간 경과로 벌어지면 역행처럼 보임 → shot4의 ab는 shot3 이하로 맞춘다.

Shots (ab/bc는 m, "C=0 기준 절대 위치"에서 변환):
  1) shot1_cruise.png        C:0   B:20  A:50  → ab=30 bc=20
  2) shot2_a_brake.png       C:20  B:40  A:60  → ab=20 bc=20
  3) shot3_b_brake.png       C:40  B:50  A:70  → ab=20 bc=10
  4) shot4_no_device.png     C:50  B:60  A:80  → ab=20 bc=10
  5) shot4_with_device.png   C:45  B:60  A:80  → ab=20 bc=15
  6) shot5_collision.png     A:80  B:60  C:60  → ab=20 bc=0 (스폰 안정 위해 bc는 최소값으로 보정)
  7) shot5_safe.png          C:50  B:60  A:80  → ab=20 bc=10
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


ShotSpec = Tuple[
    str,
    float,
    float,
    Optional[float],
    Optional[float],
    Optional[float],
]  # (filename, ab_m, bc_m, cam_z, cam_offset_back, cam_pitch_deg)


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
    cam_pitch_deg: float,
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
            carla,
            a,
            b,
            c,
            z=float(cam_z),
            back_offset_m=float(cam_offset_back),
            pitch_deg=float(cam_pitch_deg),
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
    p.add_argument("--map", default="Town05")
    p.add_argument("--fixed-dt", type=float, default=0.05)
    p.add_argument("--stable-ticks", type=int, default=10)
    p.add_argument("--img-w", type=int, default=1920)
    p.add_argument("--img-h", type=int, default=1080)
    # 목적 고정: B가 화면 중앙 + B 위에서 수직 탑다운
    p.add_argument("--fov", type=float, default=120.0)
    # 탑다운에서는 z가 프레이밍을 좌우함. A/C를 안정적으로 담기 위해 기본을 높게 둔다.
    p.add_argument("--cam-z", type=float, default=40.0)
    p.add_argument("--cam-offset-back", type=float, default=0.0)
    p.add_argument(
        "--only",
        default="",
        help=(
            "빠른 검증용: 특정 컷만 실행. "
            "예) --only shot5_collision 또는 --only shot5_collision,shot4_no_device 또는 --only 5"
        ),
    )
    p.add_argument(
        "--list-shots",
        action="store_true",
        help="컷 목록만 출력하고 종료 (빠른 확인용).",
    )
    p.add_argument(
        "--cam-pitch",
        type=float,
        default=-90.0,
        help="카메라 pitch(deg). -90이면 B 위에서 수직 탑다운(백오프 무시).",
    )
    args = p.parse_args()

    import carla

    out_dir = os.path.abspath(str(args.output_dir))
    _ensure_dir(out_dir)

    # 카메라: B 중심 + 수직 탑다운(pitch=-90)으로 고정. 이후 샷 간 차이는 ab/bc만으로 표현.
    shots: List[ShotSpec] = [
        # (filename, ab_m=A-B, bc_m=B-C, cam_z, cam_offset_back, cam_pitch_deg)
        ("shot1_cruise.png", 30.0, 20.0, None, None, None),  # C:0   B:20  A:50
        ("shot2_a_brake.png", 20.0, 20.0, None, None, None),  # C:20  B:40  A:60
        ("shot3_b_brake.png", 20.0, 10.0, None, None, None),  # C:40  B:50  A:70
        ("shot4_no_device.png", 20.0, 10.0, None, None, None),  # C:50  B:60  A:80
        ("shot4_with_device.png", 20.0, 15.0, None, None, None),  # C:45  B:60  A:80
        # collision: bc=0은 스폰이 실패할 수 있어 최소값으로 보정(시각적으로는 거의 겹침)
        ("shot5_collision.png", 20.0, 4.0, None, None, None),  # C:60  B:60  A:80
        ("shot5_safe.png", 20.0, 10.0, None, None, None),  # C:50  B:60  A:80
    ]

    if bool(args.list_shots):
        for idx, (fname, *_rest) in enumerate(shots, start=1):
            print(f"{idx}: {fname}", flush=True)
        return

    only_raw = str(args.only or "").strip()
    if only_raw:
        tokens = [t.strip() for t in only_raw.split(",") if t.strip()]
        by_index = {int(t) for t in tokens if t.isdigit()}
        by_name = {t for t in tokens if not t.isdigit()}
        filtered: List[ShotSpec] = []
        for idx, spec in enumerate(shots, start=1):
            fname = spec[0]
            stem = os.path.splitext(os.path.basename(fname))[0]
            if idx in by_index or fname in by_name or stem in by_name:
                filtered.append(spec)
        shots = filtered
        if not shots:
            raise SystemExit(f"[error] --only matched no shots: {only_raw}")

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
        for idx, (
            fname,
            ab_m,
            bc_m,
            cam_z_override,
            cam_back_override,
            cam_pitch_override,
        ) in enumerate(shots, start=1):
            out_path = os.path.join(out_dir, fname)
            print(f"\n=== Shot {idx}/7: {fname} ===", flush=True)
            _cz = float(args.cam_z) if cam_z_override is None else float(cam_z_override)
            _cb = (
                float(args.cam_offset_back)
                if cam_back_override is None
                else float(cam_back_override)
            )
            _cp = float(args.cam_pitch) if cam_pitch_override is None else float(cam_pitch_override)
            print(
                f"[shot_cfg] ab={ab_m} bc={bc_m} cam_z={_cz} cam_back={_cb} cam_pitch={_cp}",
                flush=True,
            )
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
                cam_pitch_deg=_cp,
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

