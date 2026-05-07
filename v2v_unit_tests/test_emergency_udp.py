"""급정거 UDP 패킷 risk_type·meta 필드 (소켓 목)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from brake_detector import EmergencyBrakeSignal
from sender import build_packet, process_emergency_brake_send


class _CaptureUdp:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def sendto(self, data: bytes, addr) -> None:  # noqa: ANN001
        self.payloads.append(json.loads(data.decode("utf-8")))


def test_build_packet_emergency_risk_type():
    pkt = build_packet(
        "emergency_brake",
        2.5,
        np.array([1.0, 2.0, 3.0]),
        extra={"tier": "critical", "emergency_methods": ["A"]},
    )
    assert pkt["risk_type"] == "emergency_brake"
    assert pkt["meta"]["emergency_methods"] == ["A"]


def test_process_emergency_brake_send_includes_risk_type_and_methods():
    sock = _CaptureUdp()
    sig = EmergencyBrakeSignal(
        methods=["A", "B"],
        ttc=1.25,
        dttc_dt=-3.3,
        bbox_area_rate=0.22,
        bbox_area=512.0,
    )
    ego_loc = np.array([10.0, 20.0, 0.5])
    ego_fwd = np.array([1.0, 0.0, 0.0])
    ego_vel = np.array([12.0, 0.0, 0.0])
    cam = np.eye(4)
    process_emergency_brake_send(
        sig,
        None,
        None,
        cam,
        ego_loc,
        ego_fwd,
        ego_vel,
        sock,
        mock_fixed_depth=14.0,
        sender_heading_deg=90.0,
    )
    assert len(sock.payloads) == 1
    p = sock.payloads[0]
    assert p["risk_type"] == "emergency_brake"
    assert p["ttc"] == pytest.approx(1.25)
    meta = p["meta"]
    assert meta["emergency_methods"] == ["A", "B"]
    assert meta["detection_dttc_dt"] == pytest.approx(-3.3)
    assert meta["bbox_area_rate"] == pytest.approx(0.22)
    assert meta["bbox_area"] == pytest.approx(512.0)
