"""타임스탬프가 있는 이벤트 로깅 (YOLO → UDP → UI 추적)."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, List, Optional

from config import LOG_DIR


def setup_logger(name: str = "v2v", log_file: Optional[str] = None) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = log_file or os.path.join(LOG_DIR, "v2v_events.log")
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_emergency_brake_detection(
    logger: logging.Logger,
    methods: List[str],
    ttc: float,
    dttc_dt: Optional[float],
    bbox_area_rate: Optional[float],
    bbox_area: Optional[float],
) -> None:
    """급정거 감지 시각·방법·TTC·bbox 변화율 로그."""
    logger.info(
        "EMERGENCY_BRAKE methods=%s ttc=%.4f dttc_dt=%s bbox_area_rate=%s bbox_area=%s",
        "+".join(methods),
        ttc,
        f"{dttc_dt:.4f}" if dttc_dt is not None else "—",
        f"{bbox_area_rate:.4f}" if bbox_area_rate is not None else "—",
        f"{bbox_area:.1f}" if bbox_area is not None else "—",
    )


def log_rear_brake_start(
    logger: logging.Logger, wall_time: float, ttc_at_start: Optional[float]
) -> None:
    """뒤차 제동 시작(인지 지연 측정용)."""
    logger.info(
        "REAR_BRAKE_START wall=%.4f ttc_at_start=%s",
        wall_time,
        f"{ttc_at_start:.4f}" if ttc_at_start is not None else "—",
    )


def log_scenario_result(logger: logging.Logger, payload: Any) -> None:
    import json

    logger.info("SCENARIO_RESULT %s", json.dumps(payload, ensure_ascii=False))
