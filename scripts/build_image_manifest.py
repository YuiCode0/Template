import argparse
import csv
from pathlib import Path


def build_manifest(image_dir: str, output_csv: str) -> None:
    root = Path(image_dir)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            rows.append({"path": str(p), "split": "unassigned", "source": root.name})

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    build_manifest(args.image_dir, args.output_csv)
