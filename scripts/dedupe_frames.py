import argparse
import shutil
from pathlib import Path

import imagehash
from PIL import Image


def dedupe(input_dir: str, output_dir: str, threshold: int = 4) -> None:
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    hashes = []
    kept = 0
    scanned = 0

    for path in sorted(src.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        scanned += 1
        try:
            img = Image.open(path).convert("RGB").resize((256, 256))
            h = imagehash.phash(img)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue

        if any(abs(h - old) <= threshold for old in hashes):
            continue

        hashes.append(h)
        shutil.copy2(path, dst / path.name)
        kept += 1

    print(f"scanned={scanned} kept={kept} output={dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--threshold", type=int, default=4)
    args = parser.parse_args()
    dedupe(args.input_dir, args.output_dir, args.threshold)
