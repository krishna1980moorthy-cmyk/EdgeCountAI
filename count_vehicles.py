"""Count vehicles in input_images using a local YOLO model."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=1600, help="Inference resolution; use 640 for faster runs")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--output", default="outputs/latest", help="Output folder, relative to the project or absolute")
    parser.add_argument("--model", default="yolo11n.pt", help="Local model file, e.g. yolo11s.pt")
    parser.add_argument("--iou", type=float, default=0.7, help="Overlap threshold; lower values suppress more overlapping boxes")
    parser.add_argument("--exclude", action="append", default=[], help="Exact input filename to skip; repeat for multiple files")
    args = parser.parse_args()
    if args.imgsz <= 0 or not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("imgsz must be positive; conf and iou must be in (0, 1]")
    input_dir = BASE_DIR / "input_images"
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder missing: {input_dir}")
    images = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.name not in args.exclude)
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")
    model_path = BASE_DIR / args.model
    if not model_path.is_file() and Path(args.model).name == args.model:
        model_path = BASE_DIR / "models" / args.model
    if not model_path.is_file():
        raise SystemExit(f"Model missing: {model_path}")

    print(f"Loading {model_path.name}; found {len(images)} image(s).", flush=True)
    model = YOLO(str(model_path))
    output_dir = BASE_DIR / args.output
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vehicle_counts.csv"
    fields = ["image", *VEHICLES.values(), "total", "inference_ms", "annotated_image"]
    failed = 0
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(images, 1):
            image = cv2.imread(str(path))
            if image is None:
                print(f"ERROR: Cannot read {path.name}", flush=True)
                failed += 1
                continue
            result = model.predict(image, classes=list(VEHICLES), conf=args.conf, iou=args.iou, imgsz=args.imgsz, agnostic_nms=True, device="cpu", verbose=False)[0]
            counts = Counter(int(class_id) for class_id in result.boxes.cls.tolist())
            # Keep output filenames short enough for Windows, even for long input names.
            annotated_path = annotated_dir / f"{index:03d}_{path.stem[:60]}.jpg"
            if not cv2.imwrite(str(annotated_path), result.plot(line_width=1, font_size=10, conf=False)):
                raise OSError(f"Could not save annotated image: {annotated_path}")
            row = {"image": path.name, **{name: counts[class_id] for class_id, name in VEHICLES.items()}}
            row.update(total=sum(counts.values()), inference_ms=round(result.speed["inference"], 2),
                       annotated_image=str(annotated_path.relative_to(output_dir)))
            writer.writerow(row)
            handle.flush()
            print(f"[{index}/{len(images)}] {path.name}: {row['total']} vehicles "
                  f"({', '.join(f'{name}={row[name]}' for name in VEHICLES.values())})", flush=True)
    print(f"Saved CSV: {csv_path}\nAnnotated images: {annotated_dir}", flush=True)
    if failed:
        raise SystemExit(f"Failed to read {failed} image(s); see errors above.")


if __name__ == "__main__":
    main()
