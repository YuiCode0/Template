from __future__ import annotations

import argparse
import time

import cv2
import yaml

from .core import (
    ADBInputController,
    CaptureRegion,
    CoordinateMapper,
    FPSCounter,
    ObjectFeaturePipeline,
    SimpleDecisionModule,
    YOLOTracker,
    create_capture,
    draw_objects,
    target_point_from_box,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    r = cfg["capture"]["region"]
    region = CaptureRegion(r["left"], r["top"], r["right"], r["bottom"])
    capture = create_capture(cfg["capture"]["backend"], region, cfg["capture"].get("target_fps", 120))
    capture.start()

    detector_cfg = cfg["detector"]
    tracker = YOLOTracker(
        weights=detector_cfg["weights"],
        tracker_config=cfg["tracker"].get("config", "bytetrack.yaml"),
        imgsz=detector_cfg.get("imgsz", 512),
        conf=detector_cfg.get("conf", 0.30),
        iou=detector_cfg.get("iou", 0.50),
        device=detector_cfg.get("device", 0),
    )

    features = ObjectFeaturePipeline()
    mapper = CoordinateMapper(region.left, region.top)
    input_enabled = bool(cfg.get("input", {}).get("enabled", False))
    input_controller = ADBInputController(cfg.get("input", {}).get("adb_device"))
    cooldown = float(cfg.get("input", {}).get("action_cooldown_sec", 0.25))
    last_action = 0.0
    fps = FPSCounter()

    try:
        while True:
            frame = capture.read()
            if frame is None:
                continue

            tracks = tracker.update(frame)
            objects = features.enrich(frame, tracks)
            h, w = frame.shape[:2]
            decision = SimpleDecisionModule(w, h)
            target = decision.choose_target(objects)

            if input_enabled and target is not None and time.perf_counter() - last_action > cooldown:
                tx, ty = target_point_from_box(target["xyxy"], vertical_anchor=0.65)
                sx, sy = mapper.frame_to_screen(tx, ty)
                input_controller.tap(sx, sy)
                last_action = time.perf_counter()

            debug = frame.copy()
            draw_objects(debug, objects, target)
            cv2.putText(debug, f"FPS {fps.tick():.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("cv-emulator-pipeline", debug)
            if cv2.waitKey(1) == 27:
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
