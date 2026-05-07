"""공통 설정 (CARLA 호스트, UDP, YOLO 주파수 등)."""
from __future__ import annotations

import os

# CARLA
CARLA_HOST: str = os.environ.get("CARLA_HOST", "127.0.0.1")
CARLA_PORT: int = int(os.environ.get("CARLA_PORT", "2000"))

# UDP V2V (브로드캐스트)
UDP_PORT: int = int(os.environ.get("V2V_UDP_PORT", "5005"))
# Windows에서는 "<broadcast>" 대신 서브넷 브로드캐스트가 필요할 수 있음
UDP_BROADCAST_ADDR: str = os.environ.get("V2V_BROADCAST", "255.255.255.255")

# 송신 차량 ID (JSON v_id)
SENDER_VEHICLE_ID: str = os.environ.get("V2V_SENDER_ID", "HV-001")

# YOLO 추론 목표 주파수 (Hz) — 시뮬 부하 완화
YOLO_TARGET_HZ: float = float(os.environ.get("YOLO_HZ", "7.0"))

# 목(Mock) 모드: True면 CARLA 없이 영상 파일로 송신 테스트
USE_MOCK_VIDEO: bool = os.environ.get("V2V_MOCK", "0") in ("1", "true", "True")
MOCK_VIDEO_PATH: str = os.environ.get(
    "V2V_MOCK_VIDEO",
    os.path.join(os.path.dirname(__file__), "assets", "mock_drive.mp4"),
)

# 카메라
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 360
CAMERA_FOV_DEG: float = 90.0

# TTC 임계 (초) — 낮을수록 위험
TTC_CRITICAL: float = 2.0
TTC_WARNING: float = 4.0

# 로그
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs")

def _env_range(
    legacy_key: str,
    min_key: str,
    max_key: str,
    default_lo: float,
    default_hi: float,
) -> tuple[float, float]:
    """
    legacy_key 가 설정되면 (v,v) 단일값.
    아니면 min_key / max_key, 기본은 [default_lo, default_hi] (lo>hi 는 스왑).
    """
    if os.environ.get(legacy_key, "").strip():
        v = float(os.environ[legacy_key])
        return v, v
    lo = float(os.environ.get(min_key, str(default_lo)))
    hi = float(os.environ.get(max_key, str(default_hi)))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


# 운전자 반응 시간 (초) — tier마다 [min, max] 에서 무작위 샘플 (receiver_carla + driver_response)
# 단일값만 쓰려면 V2V_REACTION_CRITICAL / V2V_REACTION_WARNING 만 설정 (호환)
(
    REACTION_CRITICAL_MIN_S,
    REACTION_CRITICAL_MAX_S,
) = _env_range(
    "V2V_REACTION_CRITICAL",
    "V2V_REACTION_CRITICAL_MIN",
    "V2V_REACTION_CRITICAL_MAX",
    0.28,
    0.48,
)
(
    REACTION_WARNING_MIN_S,
    REACTION_WARNING_MAX_S,
) = _env_range(
    "V2V_REACTION_WARNING",
    "V2V_REACTION_WARNING_MIN",
    "V2V_REACTION_WARNING_MAX",
    0.58,
    1.05,
)

# 반복 재현용 시드 (비우면 매 실행·매 패킷마다 비결정적)
REACTION_RNG_SEED: str = os.environ.get("V2V_REACTION_SEED", "").strip()

# 참고용 중앙값 (로그·외부 코드 호환)
REACTION_TIME_CRITICAL_S: float = (
    REACTION_CRITICAL_MIN_S + REACTION_CRITICAL_MAX_S
) / 2.0
REACTION_TIME_WARNING_S: float = (
    REACTION_WARNING_MIN_S + REACTION_WARNING_MAX_S
) / 2.0

# tier별 브레이크 페달 강도 (0~1) — [min,max] 에서 무작위 샘플
# 단일값: V2V_BRAKE_CRITICAL / V2V_BRAKE_WARNING 만 설정 (호환)
(
    BRAKE_CRITICAL_MIN,
    BRAKE_CRITICAL_MAX,
) = _env_range(
    "V2V_BRAKE_CRITICAL",
    "V2V_BRAKE_CRITICAL_MIN",
    "V2V_BRAKE_CRITICAL_MAX",
    0.72,
    1.0,
)
(
    BRAKE_WARNING_MIN,
    BRAKE_WARNING_MAX,
) = _env_range(
    "V2V_BRAKE_WARNING",
    "V2V_BRAKE_WARNING_MIN",
    "V2V_BRAKE_WARNING_MAX",
    0.32,
    0.62,
)

# 참고용 중앙값 (로그·호환)
BRAKE_PEDAL_CRITICAL: float = (BRAKE_CRITICAL_MIN + BRAKE_CRITICAL_MAX) / 2.0
BRAKE_PEDAL_WARNING: float = (BRAKE_WARNING_MIN + BRAKE_WARNING_MAX) / 2.0

# 제동 유지·완화 (CARLA 수신 제어)
BRAKE_HOLD_SECONDS: float = float(os.environ.get("V2V_BRAKE_HOLD_S", "2.5"))
SPEED_STOPPED_MS: float = float(os.environ.get("V2V_STOPPED_SPEED_MS", "1.2"))

# 수신 차량 순항 스로틀 (데모용, CARLA에서 움직임 확보)
RECEIVER_CRUISE_THROTTLE: float = float(
    os.environ.get("V2V_RECEIVER_CRUISE", "0.42")
)

# CARLA 수신 차량 액터 ID (비우면 audi 우선 자동 선택)
RECEIVER_VEHICLE_ACTOR_ID: str = os.environ.get("V2V_RECEIVER_ACTOR_ID", "")

# 급정거 감지 (A: TTC 변화율, B: bbox 면적 증가율)
EMERGENCY_TTC_WINDOW_FRAMES: int = int(
    os.environ.get("V2V_EMERGENCY_TTC_WINDOW", "5")
)
# dTTC/dt (s/s) 이 값 이하(더 음수)이면 급격한 위험 증가로 판단
EMERGENCY_DTTC_DT_THRESHOLD: float = float(
    os.environ.get("V2V_EMERGENCY_DTTC_DT", "-2.0")
)
# 연속 프레임 간 bbox 면적 상대 증가율이 이 값 이상이면 급정거 후보 (B)
EMERGENCY_BBOX_AREA_RATE_THRESHOLD: float = float(
    os.environ.get("V2V_EMERGENCY_BBOX_RATE", "0.15")
)
EMERGENCY_DETECTION_COOLDOWN_S: float = float(
    os.environ.get("V2V_EMERGENCY_COOLDOWN", "0.25")
)

# scenario_emergency_brake.py 기본 실험 파라미터 (환경변수로 덮어쓰기 가능)
SCENARIO_INITIAL_SPEED_KMH: float = float(
    os.environ.get("SCENARIO_SPEED_KMH", "45.0")
)
SCENARIO_HEADWAY_M: float = float(os.environ.get("SCENARIO_HEADWAY_M", "28.0"))
SCENARIO_TRIGGER_DISTANCE_M: float = float(
    os.environ.get("SCENARIO_TRIGGER_DIST_M", "40.0")
)
SCENARIO_OBSTACLE_AHEAD_M: float = float(
    os.environ.get("SCENARIO_OBSTACLE_AHEAD_M", "18.0")
)
SCENARIO_V2V_ENABLED: bool = os.environ.get("SCENARIO_V2V", "1") in (
    "1",
    "true",
    "True",
)
