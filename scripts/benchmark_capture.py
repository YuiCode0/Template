import argparse
import time

import cv2

from cv_emulator_pipeline.core import CaptureRegion, create_capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mss", "dxcam"], default="mss")
    parser.add_argument("--left", type=int, default=100)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--right", type=int, default=1380)
    parser.add_argument("--bottom", type=int, default=820)
    parser.add_argument("--target-fps", type=int, default=120)
    args = parser.parse_args()

    region = CaptureRegion(args.left, args.top, args.right, args.bottom)
    cap = create_capture(args.backend, region, args.target_fps)
    cap.start()

    frames = 0
    last = time.perf_counter()

    try:
        while True:
            frame = cap.read()
            if frame is None:
                continue
            frames += 1
            now = time.perf_counter()
            if now - last >= 1.0:
                print(f"{args.backend} capture FPS: {frames / (now - last):.1f}")
                frames = 0
                last = now
            cv2.imshow("capture benchmark", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
