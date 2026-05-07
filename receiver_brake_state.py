"""
UDP 패킷 리스트 → 수신 차량 브레이크 (receiver_carla 와 동일 물리 규칙).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config import (
    BRAKE_CRITICAL_MAX,
    BRAKE_CRITICAL_MIN,
    BRAKE_HOLD_SECONDS,
    BRAKE_WARNING_MAX,
    BRAKE_WARNING_MIN,
    REACTION_CRITICAL_MAX_S,
    REACTION_CRITICAL_MIN_S,
    REACTION_RNG_SEED,
    REACTION_WARNING_MAX_S,
    REACTION_WARNING_MIN_S,
    RECEIVER_CRUISE_THROTTLE,
    SPEED_STOPPED_MS,
)
from driver_response import (
    consume_due_schedules,
    schedule_brake_from_packet,
    speed_ms_from_velocity_xyz,
)


@dataclass
class ReceiverBrakeController:
    pending: List[Any] = field(default_factory=list)
    brake_cmd: float = 0.0
    hold_until_mono: float = 0.0
    react_rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if REACTION_RNG_SEED.strip():
            self.react_rng.seed(int(REACTION_RNG_SEED))

    def ingest_packets(
        self, packets: List[Dict[str, Any]], recv_mono: float, recv_wall: float
    ) -> None:
        for pkt in packets:
            sch = schedule_brake_from_packet(
                pkt,
                recv_mono,
                recv_wall,
                REACTION_CRITICAL_MIN_S,
                REACTION_CRITICAL_MAX_S,
                REACTION_WARNING_MIN_S,
                REACTION_WARNING_MAX_S,
                BRAKE_CRITICAL_MIN,
                BRAKE_CRITICAL_MAX,
                BRAKE_WARNING_MIN,
                BRAKE_WARNING_MAX,
                self.react_rng,
            )
            if sch is not None:
                self.pending.append(sch)

    def step_vehicle(self, vehicle: Any, carla: Any, now_mono: float) -> None:
        new_pedal, _ = consume_due_schedules(self.pending, now_mono)
        if new_pedal > 0:
            self.brake_cmd = max(self.brake_cmd, new_pedal)
            self.hold_until_mono = max(
                self.hold_until_mono, now_mono + BRAKE_HOLD_SECONDS
            )

        vel = vehicle.get_velocity()
        speed = speed_ms_from_velocity_xyz(vel.x, vel.y, vel.z)

        if speed < SPEED_STOPPED_MS:
            self.brake_cmd = 0.0

        if now_mono > self.hold_until_mono and self.brake_cmd > 0:
            self.brake_cmd = max(0.0, self.brake_cmd - 0.06)

        ctrl = carla.VehicleControl()
        if self.brake_cmd > 0.02:
            ctrl.throttle = 0.0
            ctrl.brake = min(1.0, self.brake_cmd)
            ctrl.hand_brake = False
        else:
            ctrl.throttle = RECEIVER_CRUISE_THROTTLE
            ctrl.brake = 0.0
            ctrl.hand_brake = False

        vehicle.apply_control(ctrl)
