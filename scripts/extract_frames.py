import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str, every_n: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % every_n == 0:
            cv2.imwrite(str(out / f"frame_{saved_id:06d}.jpg"), frame)
            saved_id += 1
        frame_id += 1
    cap.release()
    print(f"Saved {saved_id} frames to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output_dir")
    parser.add_argument("--every-n", type=int, default=10)
    args = parser.parse_args()
    extract_frames(args.video, args.output_dir, args.every_n)


if __name__ == "__main__":
    main()
