# Real-time CV Pipeline for Dynamic Mobile-like Environment on Android Emulator

Educational Python project for building a high-FPS computer-vision pipeline over an Android emulator window.

The project is intended for controlled sandbox scenes, QA automation, accessibility prototypes, and CV research where you have permission to capture the screen and send input.

## Features

- High-FPS screen capture via `dxcam` or `mss`
- YOLO detector wrapper for Ultralytics models
- ByteTrack / BoT-SORT wrapper via Ultralytics track mode
- HSV-based visual feature classifier for health/status bars
- Geometry and minimap helpers
- ADB input controller with tap, swipe, hold, and persistent-shell mode
- Config-driven app entrypoint
- Dataset extraction and YOLO training scripts

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/benchmark_capture.py --backend mss
python -m cv_emulator_pipeline.app --config configs/default.yaml
```

For Windows high-FPS capture, install and use `dxcam`:

```bash
pip install dxcam
python scripts/benchmark_capture.py --backend dxcam
```

## Minimal workflow

1. Benchmark capture for your emulator window.
2. Record video or capture frames from a stable ROI.
3. Annotate objects in CVAT, Label Studio, or Roboflow.
4. Train a small YOLO model, for example YOLO11n or YOLOv8n.
5. Run tracking with ByteTrack.
6. Add color/geometry/minimap filters.
7. Add cautious input actions only after perception is stable.

## Notes

- Process only the emulator ROI, not the whole monitor.
- Do not run detector on every captured frame unless necessary.
- Use tracker IDs and temporal smoothing to reduce noisy decisions.
- Keep input rate-limited with cooldowns.

## License

MIT
