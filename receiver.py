"""
V2V 수신측: UDP 비동기 수신 + Pygame 내비게이션형 경고 UI.
탑다운 미니맵에 위험 위치(빨간 점)·앞차 사각지대(반투명 웨지) 표시.

실행: python receiver.py
"""
from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pygame

from config import UDP_PORT
from v2v_logger import setup_logger

logger = setup_logger("receiver")

# --- 화면 ---
WIN_W, WIN_H = 960, 540
MAP_RECT = pygame.Rect(520, 40, 400, 400)
ALERT_RECT = pygame.Rect(40, 120, 440, 200)
FPS = 60

RISK_LABEL_KO = {
    "pedestrian": "보행자",
    "sudden_stop": "급정거/제동",
    "emergency_brake": "급정거(V2V)",
    "lane_change": "차선 변경 차량",
}


def udp_listener(pkt_q: "queue.Queue[Dict[str, Any]]", stop_evt: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", UDP_PORT))
    sock.settimeout(0.3)
    logger.info("UDP 수신 대기 port=%s", UDP_PORT)
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
                "UDP 수신 risk=%s ttc=%s from=%s",
                pkt.get("risk_type"),
                pkt.get("ttc"),
                addr,
            )
        except json.JSONDecodeError:
            logger.warning("JSON 파싱 실패 bytes=%d", len(data))
    sock.close()


def hazard_to_minimap_px(
    hazard_xy: Tuple[float, float],
    sender_xy: Tuple[float, float],
    sender_heading_deg: float,
    map_rect: pygame.Rect,
    scale_px_per_m: float = 4.0,
) -> Tuple[int, int]:
    """송신 차량 기준 상대 좌표를 미니맵 픽셀로 (전방=화면 위)."""
    hx, hy = hazard_xy
    sx, sy = sender_xy
    dx = hx - sx
    dy = hy - sy
    rad = math.radians(-sender_heading_deg)
    lx = dx * math.cos(rad) - dy * math.sin(rad)
    ly = dx * math.sin(rad) + dy * math.cos(rad)
    cx = map_rect.centerx
    cy = map_rect.centery - 40
    px = int(cx + lx * scale_px_per_m)
    py = int(cy - ly * scale_px_per_m)
    return px, py


def draw_blind_zone(
    surf: pygame.Surface, map_rect: pygame.Rect, sender_heading_deg: float
) -> None:
    """앞차에 가려진 사각지대(투시/탑다운) 시각 강조."""
    overlay = pygame.Surface(map_rect.size, pygame.SRCALPHA)
    cx = map_rect.width // 2
    cy = map_rect.height // 4
    w, h = map_rect.width, map_rect.height
    spread = 110
    base_y = cy + 50
    poly = [
        (cx, cy - 20),
        (cx - spread, h - 30),
        (cx + spread, h - 30),
    ]
    pygame.draw.polygon(overlay, (255, 60, 40, 70), poly)
    pygame.draw.polygon(overlay, (255, 80, 60, 220), poly, 2)
    surf.blit(overlay, map_rect.topleft)

    try:
        font = pygame.font.SysFont("malgungothic", 18)
    except Exception:
        font = pygame.font.Font(None, 20)
    txt = font.render("사각지대 (앞차 차단)", True, (255, 120, 100))
    surf.blit(txt, (map_rect.x + 12, map_rect.y + 8))


def draw_sender_receiver_icons(surf: pygame.Surface, map_rect: pygame.Rect) -> None:
    cx = map_rect.centerx
    top = map_rect.centery - 80
    pygame.draw.rect(
        surf, (90, 90, 90), pygame.Rect(cx - 35, top, 70, 36), border_radius=4
    )
    bot = map_rect.bottom - 55
    pygame.draw.rect(
        surf, (50, 120, 200), pygame.Rect(cx - 22, bot, 44, 28), border_radius=4
    )


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("V2V 수신 — 내비게이션 경고")
    clock = pygame.time.Clock()

    pkt_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=200)
    stop_evt = threading.Event()
    th = threading.Thread(
        target=udp_listener, args=(pkt_q, stop_evt), daemon=True
    )
    th.start()

    active_alerts: List[Dict[str, Any]] = []
    hazard_dots: List[Tuple[int, int, str, float]] = []
    last_packet_ts = 0.0

    running = True
    while running:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        while True:
            try:
                pkt = pkt_q.get_nowait()
            except queue.Empty:
                break
            meta = pkt.get("meta") or {}
            tier = str(meta.get("tier", "")).lower()
            tier_tag = ""
            if tier == "critical":
                tier_tag = "[긴급] "
            elif tier == "warning":
                tier_tag = "[주의] "
            active_alerts.insert(
                0,
                {
                    "msg": f"{tier_tag}{RISK_LABEL_KO.get(pkt.get('risk_type'), pkt.get('risk_type'))} · TTC {pkt.get('ttc')}s",
                    "until": now + 4.0,
                    "raw": pkt,
                    "tier": tier or "unknown",
                },
            )
            loc = pkt.get("location") or {}
            sl = meta.get("sender_location") or {}
            if sl and loc.get("x") is not None:
                try:
                    px, py = hazard_to_minimap_px(
                        (float(loc["x"]), float(loc["y"])),
                        (float(sl["x"]), float(sl["y"])),
                        float(meta.get("sender_heading_deg", 0.0)),
                        MAP_RECT,
                    )
                    hazard_dots.append(
                        (
                            px,
                            py,
                            str(pkt.get("risk_type", "")),
                            now + 6.0,
                        )
                    )
                except (TypeError, ValueError, KeyError):
                    pass
            last_packet_ts = now

        active_alerts = [a for a in active_alerts if a["until"] > now]
        hazard_dots = [h for h in hazard_dots if h[3] > now]

        screen.fill((18, 22, 28))

        # 미니맵 패널
        pygame.draw.rect(screen, (35, 42, 52), MAP_RECT.inflate(8, 8), border_radius=8)
        pygame.draw.rect(screen, (55, 62, 74), MAP_RECT, border_radius=6)
        try:
            title = pygame.font.SysFont("malgungothic", 22)
        except Exception:
            title = pygame.font.Font(None, 24)
        screen.blit(
            title.render("탑다운 / 사각지대 투시", True, (220, 225, 235)),
            (MAP_RECT.x, MAP_RECT.y - 32),
        )

        draw_blind_zone(screen, MAP_RECT, 0.0)
        draw_sender_receiver_icons(screen, MAP_RECT)

        for px, py, rtype, _exp in hazard_dots:
            pygame.draw.circle(screen, (255, 60, 60), (px, py), 10)
            pygame.draw.circle(screen, (255, 200, 100), (px, py), 10, 2)

        # 경고 팝업 (tier: 긴급=빨강 테두리, 주의=주황)
        pygame.draw.rect(screen, (48, 38, 30), ALERT_RECT, border_radius=10)
        top_tier = active_alerts[0].get("tier", "") if active_alerts else ""
        if top_tier == "critical":
            border_rgb = (255, 55, 55)
            title_rgb = (255, 70, 70)
            title_txt = "! 긴급 위험"
        elif top_tier == "warning":
            border_rgb = (255, 160, 60)
            title_rgb = (255, 190, 90)
            title_txt = "! 주의 경고"
        else:
            border_rgb = (255, 140, 60)
            title_rgb = (255, 90, 40)
            title_txt = "! 위험 경고"
        pygame.draw.rect(screen, border_rgb, ALERT_RECT, 3, border_radius=10)
        try:
            warn_font = pygame.font.SysFont("malgungothic", 26)
        except Exception:
            warn_font = pygame.font.Font(None, 28)
        if active_alerts:
            screen.blit(
                warn_font.render(title_txt, True, title_rgb),
                (ALERT_RECT.x + 20, ALERT_RECT.y + 16),
            )
            try:
                body = pygame.font.SysFont("malgungothic", 20)
            except Exception:
                body = pygame.font.Font(None, 22)
            for i, al in enumerate(active_alerts[:4]):
                screen.blit(
                    body.render(al["msg"], True, (245, 240, 230)),
                    (ALERT_RECT.x + 24, ALERT_RECT.y + 56 + i * 30),
                )
        else:
            screen.blit(
                warn_font.render("정상 주행", True, (120, 200, 140)),
                (ALERT_RECT.x + 20, ALERT_RECT.y + 70),
            )

        try:
            hud = pygame.font.SysFont("consolas", 16)
        except Exception:
            hud = pygame.font.Font(None, 18)
        screen.blit(
            hud.render(f"UDP :{UDP_PORT}  last_rx={last_packet_ts:.3f}", True, (160, 170, 180)),
            (40, WIN_H - 36),
        )

        pygame.display.flip()
        clock.tick(FPS)

    stop_evt.set()
    pygame.quit()


if __name__ == "__main__":
    main()
