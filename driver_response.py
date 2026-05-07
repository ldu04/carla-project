"""
V2V tier별 운전자 인지·반응 지연 및 제동 강도 (순수 로직, CARLA 없이 검증 가능).

반응 시간·브레이크 페달은 tier별 [min,max] 구간에서 균등 샘플(uniform)로 뽑을 수 있다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

Tier = Literal["critical", "warning", "info"]


def reaction_delay_sample(
    tier: str,
    crit_min: float,
    crit_max: float,
    warn_min: float,
    warn_max: float,
    rng: Optional[random.Random] = None,
) -> float:
    """tier별 [min,max] 에서 균등 무작위 반응 지연(초). rng 로 재현 가능."""
    r = rng if rng is not None else random.Random()
    t = (tier or "").strip().lower()
    if t == "critical":
        lo, hi = crit_min, crit_max
    elif t == "warning":
        lo, hi = warn_min, warn_max
    else:
        lo, hi = warn_min, warn_max
    if lo > hi:
        lo, hi = hi, lo
    return r.uniform(lo, hi)


def reaction_delay_s(
    tier: str,
    critical_s: float,
    warning_s: float,
) -> float:
    """단일 스칼라(고정 지연) — 예전 API·테스트 호환."""
    t = (tier or "").strip().lower()
    if t == "critical":
        return critical_s
    if t == "warning":
        return warning_s
    return warning_s


def brake_pedal_sample(
    tier: str,
    brake_crit_min: float,
    brake_crit_max: float,
    brake_warn_min: float,
    brake_warn_max: float,
    rng: Optional[random.Random] = None,
) -> float:
    """tier별 브레이크 페달 강도 [0,1] 균등 샘플."""
    r = rng if rng is not None else random.Random()
    t = (tier or "").strip().lower()
    if t == "critical":
        lo, hi = brake_crit_min, brake_crit_max
    elif t == "warning":
        lo, hi = brake_warn_min, brake_warn_max
    else:
        lo, hi = brake_warn_min, brake_warn_max
    if lo > hi:
        lo, hi = hi, lo
    v = r.uniform(lo, hi)
    return min(1.0, max(0.0, v))


def brake_pedal_for_tier(
    tier: str,
    pedal_critical: float,
    pedal_warning: float,
) -> float:
    """단일 스칼라 페달 — 예전 API·호환."""
    t = (tier or "").strip().lower()
    if t == "critical":
        return min(1.0, max(0.0, pedal_critical))
    if t == "warning":
        return min(1.0, max(0.0, pedal_warning))
    return pedal_warning


@dataclass(order=True)
class ScheduledBrake:
    """fire_monotonic 시점에 목표 브레이크를 적용하기 위한 예약."""

    fire_monotonic: float
    brake_pedal: float = field(compare=False)
    tier: str = field(compare=False)
    recv_wallclock: float = field(compare=False, default=0.0)
    sampled_reaction_s: float = field(compare=False, default=0.0)
    sampled_brake_pedal: float = field(compare=False, default=0.0)


def schedule_brake_from_packet(
    pkt: Dict[str, Any],
    recv_monotonic: float,
    recv_wallclock: float,
    crit_min: float,
    crit_max: float,
    warn_min: float,
    warn_max: float,
    brake_crit_min: float,
    brake_crit_max: float,
    brake_warn_min: float,
    brake_warn_max: float,
    rng: Optional[random.Random] = None,
) -> Optional[ScheduledBrake]:
    """UDP 패킷 한 건에서 ScheduledBrake 생성. tier 없거나 info면 None."""
    meta = pkt.get("meta") or {}
    tier = str(meta.get("tier", "")).lower()
    if tier not in ("critical", "warning"):
        return None
    delay = reaction_delay_sample(
        tier, crit_min, crit_max, warn_min, warn_max, rng
    )
    pedal = brake_pedal_sample(
        tier,
        brake_crit_min,
        brake_crit_max,
        brake_warn_min,
        brake_warn_max,
        rng,
    )
    return ScheduledBrake(
        fire_monotonic=recv_monotonic + delay,
        brake_pedal=pedal,
        tier=tier,
        recv_wallclock=recv_wallclock,
        sampled_reaction_s=delay,
        sampled_brake_pedal=pedal,
    )


def speed_ms_from_velocity_xyz(vx: float, vy: float, vz: float) -> float:
    """CARLA get_velocity() 크기 (m/s)."""
    return (vx * vx + vy * vy + vz * vz) ** 0.5


def consume_due_schedules(
    pending: List[ScheduledBrake], now_mono: float
) -> Tuple[float, List[str]]:
    """
    발화 시각이 지난 예약을 제거하고, 그중 최대 브레이크 강도와 tier 목록을 반환.
    """
    max_pedal = 0.0
    fired_tiers: List[str] = []
    kept: List[ScheduledBrake] = []
    for ev in pending:
        if now_mono >= ev.fire_monotonic:
            max_pedal = max(max_pedal, ev.brake_pedal)
            fired_tiers.append(ev.tier)
        else:
            kept.append(ev)
    pending[:] = kept
    return max_pedal, fired_tiers


def timeline_fire_times(
    packets: List[Tuple[float, str]],
    crit_min: float,
    crit_max: float,
    warn_min: float,
    warn_max: float,
    rng: Optional[random.Random] = None,
) -> List[Tuple[float, str, float]]:
    """
    검증용: (수신 시각, tier) 목록 -> (브레이크 발판 시각, tier, 샘플된 delay) 정렬.
    """
    r = rng if rng is not None else random.Random(0)
    out: List[Tuple[float, str, float]] = []
    for recv_t, tier in packets:
        d = reaction_delay_sample(tier, crit_min, crit_max, warn_min, warn_max, r)
        out.append((recv_t + d, tier, d))
    out.sort(key=lambda x: x[0])
    return out


def timeline_fire_times_fixed_delay(
    packets: List[Tuple[float, str]],
    critical_s: float,
    warning_s: float,
) -> List[Tuple[float, str, float]]:
    """고정 지연만 쓰는 검증용 (구 reaction_delay_s)."""
    out: List[Tuple[float, str, float]] = []
    for recv_t, tier in packets:
        d = reaction_delay_s(tier, critical_s, warning_s)
        out.append((recv_t + d, tier, d))
    out.sort(key=lambda x: x[0])
    return out
