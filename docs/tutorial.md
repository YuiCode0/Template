# Real-time CV Pipeline for Dynamic Mobile-like Environment on Android Emulator

This document summarizes the tutorial roadmap from the ChatGPT answer.

## 1. Capture

Use DXcam on Windows for high-FPS ROI capture. Use MSS as a portable baseline. Avoid ADB screencap for real-time capture unless you only need debugging frames.

## 2. Detection

Use a small Ultralytics model first: YOLO11n, YOLO11s, YOLOv8n, or YOLOv8s. Start with classes such as `character`, `health_bar`, `projectile`, `obstacle`, and `minimap_marker`.

## 3. Visual classification

Do not force YOLO to learn every team/state distinction. Detect objects first, then classify visual properties such as health/status bar color in HSV space.

## 4. Tracking

Start with ByteTrack through Ultralytics `model.track(..., tracker="bytetrack.yaml")`. Move to BoT-SORT if occlusion or identity switches become a problem.

## 5. Input

Use ADB `input tap` and `input swipe` for controlled experiments. Keep input disabled until perception is stable, and use cooldowns.

## 6. Architecture

ScreenCapture -> Preprocess -> Detector/Tracker -> Feature Pipeline -> Decision Module -> Input Controller.

## 7. Performance

Target capture above 100 FPS and detector/tracker around 20-60 FPS depending on GPU, image size, and model size.

## 8. Common issues

Low FPS usually means ROI is too large, debug rendering is too expensive, detector is too heavy, or too many copies/conversions are happening.

## 9. Open resources

Useful references: Ultralytics YOLO docs, DXcam, python-mss, ByteTrack, SORT, BoT-SORT, and Roboflow Universe gaming datasets.
