"""
YOLOv8 기반 탐지 + 휴리스틱(급정거·차선변경 근사).
COCO: person, car, truck, bus 등.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# COCO class ids (ultralytics yolov8n.pt)
COCO_PERSON = 0
COCO_CAR = 2
COCO_BUS = 5
COCO_TRUCK = 7


@dataclass
class DetectionRisk:
    risk_type: str  # pedestrian | sudden_stop | lane_change
    xyxy: Tuple[float, float, float, float]
    score: float
    lateral_pixel_delta: float = 0.0


def red_tail_ratio_in_bbox(
    bgr: np.ndarray, xyxy: Tuple[float, float, float, float]
) -> float:
    """후미 급정거 힌트: bbox 내 붉은 픽셀 비율 (단순 HSV)."""
    # V2V 시나리오에서 V2V를 끄고 실행할 때(OpenCV 미설치)도 모듈 import가 깨지지 않도록
    # cv2는 여기서 지연 import 한다.
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low1 = (0, 70, 50)
    high1 = (10, 255, 255)
    low2 = (170, 70, 50)
    high2 = (180, 255, 255)
    m1 = cv2.inRange(hsv, low1, high1)
    m2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(mask.size + 1e-6)


class YoloRiskPipeline:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self._prev_vehicle_centers: List[Tuple[float, float]] = []

    def infer_frame(self, bgr: np.ndarray) -> Tuple[List[DetectionRisk], object]:
        results = self.model.predict(
            source=bgr, verbose=False, imgsz=640, conf=0.35
        )
        res = results[0]
        risks: List[DetectionRisk] = []
        if res.boxes is None or len(res.boxes) == 0:
            self._prev_vehicle_centers = []
            return risks, res

        xyxys = res.boxes.xyxy.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        vehicle_centers: List[Tuple[float, float]] = []
        vehicle_rows: List[Tuple[Tuple[float, float, float, float], float, float, float]] = []

        for xyxy, cid, sc in zip(xyxys, cls_ids, scores):
            x1, y1, x2, y2 = xyxy.tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if cid == COCO_PERSON:
                risks.append(
                    DetectionRisk(
                        risk_type="pedestrian",
                        xyxy=(x1, y1, x2, y2),
                        score=float(sc),
                    )
                )
            elif cid in (COCO_CAR, COCO_BUS, COCO_TRUCK):
                vehicle_centers.append((cx, cy))
                vehicle_rows.append(((x1, y1, x2, y2), float(sc), cx, cy))

        prev = self._prev_vehicle_centers
        for (xyxy, sc, cx, cy) in vehicle_rows:
            lat_delta = 0.0
            if prev:
                best = min(prev, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                dist2 = (best[0] - cx) ** 2 + (best[1] - cy) ** 2
                if dist2 < 150.0 ** 2:
                    lat_delta = abs(cx - best[0])

            rr = red_tail_ratio_in_bbox(bgr, xyxy)
            if rr > 0.08:
                risks.append(
                    DetectionRisk(
                        risk_type="sudden_stop",
                        xyxy=xyxy,
                        score=float(sc) * (0.5 + rr),
                    )
                )
            elif lat_delta > 12.0 and cy > bgr.shape[0] * 0.22:
                risks.append(
                    DetectionRisk(
                        risk_type="lane_change",
                        xyxy=xyxy,
                        score=float(sc),
                        lateral_pixel_delta=float(lat_delta),
                    )
                )

        self._prev_vehicle_centers = vehicle_centers
        return risks, res
