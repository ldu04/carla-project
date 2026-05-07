"""
1단계 통합 예제: CARLA 없이 송·수신 차량 위치만 UDP로 주고받기 (GPS 시뮬).
실행:
  터미널1: python step1_gps_udp.py sender
  터미널2: python step1_gps_udp.py recv
"""
from __future__ import annotations

import argparse
import json
import socket
import time

from config import UDP_BROADCAST_ADDR, UDP_PORT


def sender() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    x = 0.0
    while True:
        pkt = {
            "mode": "gps_only",
            "sender_xy": [round(x, 2), 0.0],
            "ts": time.time(),
        }
        sock.sendto(
            json.dumps(pkt).encode("utf-8"),
            (UDP_BROADCAST_ADDR, UDP_PORT),
        )
        print("sent", pkt)
        x += 1.0
        time.sleep(0.5)


def recv() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))
    while True:
        data, a = sock.recvfrom(4096)
        print(a, data.decode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("sender", "recv"))
    args = p.parse_args()
    if args.role == "sender":
        sender()
    else:
        recv()


if __name__ == "__main__":
    main()
