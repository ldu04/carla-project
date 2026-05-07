"""
CARLA 수신 차량: UDP V2V 패킷 수신 후 tier별 반응 지연을 두고 브레이크로 감속.

동기 모드에서는 송신측(sender.py)이 world.tick()을 호출합니다.
이 스크립트는 wait_for_tick()으로 동기화만 맞추고 apply_control만 수행합니다.

실행 순서 (예시):
  1) CARLA 서버 → 2) python sender.py → 3) python receiver_carla.py

환경 변수: V2V_REACTION_*, V2V_BRAKE_*, V2V_RECEIVER_*, V2V_REACTION_SEED 등 (config.py)
"""
from __future__ import annotations

import json
import random
import queue
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from config import (
    BRAKE_CRITICAL_MAX,
    BRAKE_CRITICAL_MIN,
    BRAKE_HOLD_SECONDS,
    BRAKE_WARNING_MAX,
    BRAKE_WARNING_MIN,
    CARLA_HOST,
    CARLA_PORT,
    RECEIVER_CRUISE_THROTTLE,
    RECEIVER_VEHICLE_ACTOR_ID,
    REACTION_CRITICAL_MAX_S,
    REACTION_CRITICAL_MIN_S,
    REACTION_RNG_SEED,
    REACTION_WARNING_MAX_S,
    REACTION_WARNING_MIN_S,
    SPEED_STOPPED_MS,
    UDP_PORT,
)
from driver_response import (
    ScheduledBrake,
    consume_due_schedules,
    schedule_brake_from_packet,
    speed_ms_from_velocity_xyz,
)
from v2v_logger import setup_logger

logger = setup_logger("receiver_carla")


def udp_listener(pkt_q: "queue.Queue[Dict[str, Any]]", stop_evt: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", UDP_PORT))
    sock.settimeout(0.25)
    logger.info("UDP 수신 (CARLA 제어) port=%s", UDP_PORT)
    while not stop_evt.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            pkt = json.loads(data.decode("utf-8"))
            pkt_q.put(pkt)
            logger.info(
                "패킷 수신 from=%s tier=%s",
                addr,
                (pkt.get("meta") or {}).get("tier"),
            )
        except json.JSONDecodeError:
            logger.warning("JSON 실패")
    sock.close()


def pick_receiver_vehicle(world: Any) -> Optional[Any]:
    """수신용 승용차 선택: 환경변수 ID 우선, 없으면 audi 블루프린트."""
    import carla

    if RECEIVER_VEHICLE_ACTOR_ID.strip():
        aid = int(RECEIVER_VEHICLE_ACTOR_ID)
        for a in world.get_actors():
            if a.id == aid and str(a.type_id).startswith("vehicle."):
                logger.info("수신 차량 actor id=%s type=%s", aid, a.type_id)
                return a

    for a in world.get_actors().filter("vehicle.audi.*"):
        logger.info("수신 차량 자동 선택 type=%s id=%s", a.type_id, a.id)
        return a

    vehicles = list(world.get_actors().filter("vehicle.*"))
    if len(vehicles) >= 2:
        v = vehicles[-1]
        logger.warning(
            "audi 미발견 — 마지막 vehicle 사용 id=%s type=%s", v.id, v.type_id
        )
        return v
    if vehicles:
        logger.warning("차량 1대만 있음 — 해당 차량으로 제어 시도")
        return vehicles[0]
    logger.error("vehicle 액터 없음")
    return None


def run() -> None:
    import carla

    vehicle = None
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)
    world = client.get_world()

    vehicle = pick_receiver_vehicle(world)
    if vehicle is None:
        raise RuntimeError("수신 차량을 찾지 못했습니다. sender를 먼저 실행했는지 확인하세요.")

    try:
        vehicle.set_autopilot(False)
    except Exception as exc:
        logger.warning("autopilot 해제 실패(무시): %s", exc)

    pkt_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=500)
    stop_evt = threading.Event()
    udp_th = threading.Thread(target=udp_listener, args=(pkt_q, stop_evt), daemon=True)
    udp_th.start()

    pending: List[ScheduledBrake] = []
    brake_cmd = 0.0
    hold_until_mono = 0.0

    react_rng = random.Random()
    if REACTION_RNG_SEED != "":
        react_rng.seed(int(REACTION_RNG_SEED))

    logger.info(
        "반응지연 [crit %.2f–%.2f]s [warn %.2f–%.2f]s | 브레이크 [crit %.2f–%.2f] [warn %.2f–%.2f] 시드=%s",
        REACTION_CRITICAL_MIN_S,
        REACTION_CRITICAL_MAX_S,
        REACTION_WARNING_MIN_S,
        REACTION_WARNING_MAX_S,
        BRAKE_CRITICAL_MIN,
        BRAKE_CRITICAL_MAX,
        BRAKE_WARNING_MIN,
        BRAKE_WARNING_MAX,
        REACTION_RNG_SEED or "(비고정)",
    )

    try:
        while True:
            world.wait_for_tick(timeout=5.0)

            while True:
                try:
                    pkt = pkt_q.get_nowait()
                except queue.Empty:
                    break
                recv_m = time.monotonic()
                recv_w = time.time()
                sch = schedule_brake_from_packet(
                    pkt,
                    recv_m,
                    recv_w,
                    REACTION_CRITICAL_MIN_S,
                    REACTION_CRITICAL_MAX_S,
                    REACTION_WARNING_MIN_S,
                    REACTION_WARNING_MAX_S,
                    BRAKE_CRITICAL_MIN,
                    BRAKE_CRITICAL_MAX,
                    BRAKE_WARNING_MIN,
                    BRAKE_WARNING_MAX,
                    react_rng,
                )
                if sch is not None:
                    pending.append(sch)
                    logger.info(
                        "제동 예약 tier=%s delay=%.3fs fire_in=%.3fs brake=%.2f",
                        sch.tier,
                        sch.sampled_reaction_s,
                        sch.fire_monotonic - recv_m,
                        sch.sampled_brake_pedal,
                    )

            now_m = time.monotonic()

            new_pedal, fired = consume_due_schedules(pending, now_m)
            if new_pedal > 0:
                brake_cmd = max(brake_cmd, new_pedal)
                hold_until_mono = max(hold_until_mono, now_m + BRAKE_HOLD_SECONDS)
                logger.info(
                    "운전자 반응 후 브레이크 적용 tiers=%s cmd=%.2f",
                    fired,
                    brake_cmd,
                )

            vel = vehicle.get_velocity()
            speed = speed_ms_from_velocity_xyz(vel.x, vel.y, vel.z)

            if speed < SPEED_STOPPED_MS:
                brake_cmd = 0.0

            if now_m > hold_until_mono and brake_cmd > 0:
                brake_cmd = max(0.0, brake_cmd - 0.06)

            ctrl = carla.VehicleControl()
            if brake_cmd > 0.02:
                ctrl.throttle = 0.0
                ctrl.brake = min(1.0, brake_cmd)
                ctrl.hand_brake = False
            else:
                ctrl.throttle = RECEIVER_CRUISE_THROTTLE
                ctrl.brake = 0.0
                ctrl.hand_brake = False

            vehicle.apply_control(ctrl)

    except KeyboardInterrupt:
        logger.info("종료")
    finally:
        stop_evt.set()
        try:
            if vehicle is not None:
                import carla as _carla

                c = _carla.VehicleControl()
                c.throttle = 0.0
                c.brake = 0.5
                vehicle.apply_control(c)
        except Exception:
            pass


if __name__ == "__main__":
    run()
