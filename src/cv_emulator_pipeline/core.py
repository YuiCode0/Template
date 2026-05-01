from __future__ import annotations

import math
import subprocess
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mss
import numpy as np


@dataclass
class CaptureRegion:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_mss_monitor(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def as_dxcam_region(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


class MSSCapture:
    def __init__(self, region: CaptureRegion):
        self.region = region
        self.sct = mss.mss()

    def start(self) -> None:
        return None

    def read(self) -> Optional[np.ndarray]:
        raw = self.sct.grab(self.region.as_mss_monitor())
        frame = np.asarray(raw)
        return frame[:, :, :3]

    def stop(self) -> None:
        self.sct.close()


class DXCamCapture:
    def __init__(self, region: CaptureRegion, target_fps: int = 120):
        import dxcam  # Windows-only optional dependency

        self.region = region
        self.target_fps = target_fps
        self.camera = dxcam.create(output_idx=0, output_color="BGR")

    def start(self) -> None:
        self.camera.start(region=self.region.as_dxcam_region(), target_fps=self.target_fps)

    def read(self) -> Optional[np.ndarray]:
        return self.camera.get_latest_frame()

    def stop(self) -> None:
        self.camera.stop()


def create_capture(backend: str, region: CaptureRegion, target_fps: int = 120):
    backend = backend.lower().strip()
    if backend == "dxcam":
        return DXCamCapture(region, target_fps=target_fps)
    if backend == "mss":
        return MSSCapture(region)
    raise ValueError(f"Unsupported capture backend: {backend}")


class YOLOTracker:
    def __init__(self, weights: str, tracker_config: str = "bytetrack.yaml", imgsz: int = 512,
                 conf: float = 0.25, iou: float = 0.5, device: int | str = 0):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.tracker_config = tracker_config
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

    def update(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        return [
            {"track_id": int(tid), "class_id": int(cls), "confidence": float(conf), "xyxy": box}
            for box, conf, cls, tid in zip(xyxy, confs, cls_ids, track_ids)
        ]


class HealthBarColorClassifier:
    def __init__(self):
        self.ranges = {
            "ally_green": [((35, 60, 60), (90, 255, 255))],
            "ally_blue": [((90, 60, 60), (130, 255, 255))],
            "enemy_red": [((0, 70, 70), (10, 255, 255)), ((170, 70, 70), (180, 255, 255))],
        }

    def classify_crop(self, bgr_crop: np.ndarray) -> tuple[str, float]:
        if bgr_crop is None or bgr_crop.size == 0:
            return "unknown", 0.0

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        total = max(hsv.shape[0] * hsv.shape[1], 1)
        scores: dict[str, float] = {}

        for label, ranges in self.ranges.items():
            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                mask_total = cv2.bitwise_or(mask_total, mask)
            scores[label] = cv2.countNonZero(mask_total) / total

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        if best_score < 0.08:
            return "unknown", best_score
        if "enemy" in best_label:
            return "enemy", best_score
        if "ally" in best_label:
            return "ally", best_score
        return "unknown", best_score


def crop_health_bar_above(frame: np.ndarray, xyxy: np.ndarray, top_ratio: float = 0.25) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    box_h = y2 - y1
    bar_y1 = max(0, y1 - int(box_h * top_ratio))
    bar_y2 = min(h, max(0, y1 + int(box_h * 0.10)))
    x1 = max(0, x1)
    x2 = min(w, x2)
    return frame[bar_y1:bar_y2, x1:x2]


def extract_geometry_features(xyxy: np.ndarray, frame_shape: tuple[int, ...]) -> dict[str, float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return {
        "cx_norm": cx / w,
        "cy_norm": cy / h,
        "w_norm": bw / w,
        "h_norm": bh / h,
        "aspect": bw / max(bh, 1.0),
        "area_norm": (bw * bh) / max(w * h, 1.0),
    }


class TeamSmoother:
    def __init__(self, history: int = 8):
        self.history = defaultdict(lambda: deque(maxlen=history))

    def update(self, track_id: int, team: str) -> str:
        if team != "unknown":
            self.history[track_id].append(team)
        if not self.history[track_id]:
            return "unknown"
        return Counter(self.history[track_id]).most_common(1)[0][0]


class ObjectFeaturePipeline:
    def __init__(self):
        self.color_classifier = HealthBarColorClassifier()
        self.team_smoother = TeamSmoother()

    def enrich(self, frame: np.ndarray, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for track in tracks:
            xyxy = track["xyxy"]
            bar_crop = crop_health_bar_above(frame, xyxy)
            team, score = self.color_classifier.classify_crop(bar_crop)
            team = self.team_smoother.update(track["track_id"], team)
            item = dict(track)
            item.update({
                "team": team,
                "team_score": score,
                "geometry": extract_geometry_features(xyxy, frame.shape),
            })
            enriched.append(item)
        return enriched


class SimpleDecisionModule:
    def __init__(self, frame_width: int, frame_height: int):
        self.center = (frame_width / 2, frame_height / 2)

    def choose_target(self, objects: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        candidates = [obj for obj in objects if obj.get("team") == "enemy" and obj.get("class_id") == 0]
        if not candidates:
            return None
        cx0, cy0 = self.center

        def score(obj: dict[str, Any]) -> float:
            x1, y1, x2, y2 = obj["xyxy"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = math.hypot(cx - cx0, cy - cy0)
            return dist - 200.0 * obj.get("team_score", 0.0)

        return min(candidates, key=score)


class CoordinateMapper:
    def __init__(self, capture_left: int, capture_top: int, scale_x: float = 1.0, scale_y: float = 1.0):
        self.capture_left = capture_left
        self.capture_top = capture_top
        self.scale_x = scale_x
        self.scale_y = scale_y

    def frame_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.capture_left + x * self.scale_x, self.capture_top + y * self.scale_y


def target_point_from_box(xyxy: np.ndarray, vertical_anchor: float = 0.65) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, y1 + (y2 - y1) * vertical_anchor


class ADBInputController:
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id

    def _base_cmd(self) -> list[str]:
        return ["adb", "-s", self.device_id] if self.device_id else ["adb"]

    def tap(self, x: float, y: float) -> None:
        cmd = self._base_cmd() + ["shell", "input", "tap", str(int(x)), str(int(y))]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 100) -> None:
        cmd = self._base_cmd() + [
            "shell", "input", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms))
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def hold(self, x: float, y: float, duration_ms: int = 300) -> None:
        self.swipe(x, y, x, y, duration_ms)


def draw_objects(frame: np.ndarray, objects: list[dict[str, Any]], selected: Optional[dict[str, Any]] = None) -> None:
    selected_id = selected["track_id"] if selected else None
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["xyxy"])
        team = obj.get("team", "unknown")
        track_id = obj.get("track_id", -1)
        conf = obj.get("confidence", 0.0)
        color = (180, 180, 180)
        if team == "ally":
            color = (80, 220, 80)
        elif team == "enemy":
            color = (80, 80, 255)
        thickness = 3 if track_id == selected_id else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"id={track_id} {team} {conf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


class FPSCounter:
    def __init__(self):
        self.last = time.perf_counter()
        self.frames = 0
        self.fps = 0.0

    def tick(self) -> float:
        self.frames += 1
        now = time.perf_counter()
        elapsed = now - self.last
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.last = now
        return self.fps
