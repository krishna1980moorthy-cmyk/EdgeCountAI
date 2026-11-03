# EdgeCountAI

A local vehicle-detection proof of concept that counts cars, motorcycles, buses, and trucks in still images using Ultralytics YOLO11. It writes a CSV summary and an annotated image for each input.

## Current status

- Working Windows/Python image-counting pipeline with YOLO11n and YOLO11s.
- Configurable image size, confidence threshold, and overlap filtering.
- Local CPU inference after dependencies and model weights have been downloaded.
- Raspberry Pi and ExecuTorch deployment are planned, not implemented or verified.
- Video tracking and unique vehicle counting across frames are not implemented.

## Setup on Windows

Python 3.13 was used for local testing. From the project folder in PowerShell:

```powershell
py -3.13 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
New-Item -ItemType Directory -Force models, input_images | Out-Null
Push-Location models
& "..\.venv\Scripts\python.exe" -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11s.pt')"
Pop-Location
```

Dependency installation and the first model download require internet access. Model files, personal inputs, and generated results are excluded from Git.

## Run detection

Place original, unmarked images directly in `input_images/`. Supported formats: JPG, JPEG, PNG, WEBP, BMP, TIF, and TIFF. Images in subfolders are not processed.

```powershell
& ".\.venv\Scripts\python.exe" ".\count_vehicles.py" --model yolo11s.pt --imgsz 1600 --conf 0.25 --output outputs/latest
```

For a lighter configuration:

```powershell
& ".\.venv\Scripts\python.exe" ".\count_vehicles.py" --model yolo11n.pt --imgsz 640 --output outputs/nano_640
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `yolo11n.pt` | Model filename in `models/`, a project-relative path, or an absolute path |
| `--imgsz` | `1600` | Inference image size |
| `--conf` | `0.25` | Minimum confidence threshold |
| `--iou` | `0.7` | Overlap threshold; lower values suppress more overlapping boxes |
| `--output` | `outputs/latest` | Output folder relative to the project, or an absolute path |
| `--exclude` | none | Exact input filename to skip; repeat to exclude multiple files |

Paths are resolved relative to the script, so running it from another working directory is supported. The script uses CPU inference and class-agnostic overlap suppression.

## Results

Each run creates:

- `vehicle_counts.csv`: input filename, car/motorcycle/bus/truck counts, total, inference time in milliseconds, and annotated-image path.
- `annotated/`: images with detection boxes and class labels. Shortened filenames avoid overly long Windows paths.

```powershell
Import-Csv ".\outputs\latest\vehicle_counts.csv" | Format-Table car,motorcycle,bus,truck,total,inference_ms
ii ".\outputs\latest\annotated"
```

The inference time excludes image loading, annotation, and file writing; it is not total processing time. Reusing an output folder replaces its CSV and matching image files. Use a new output folder to preserve a run. Close the CSV in Excel before rerunning.

## Folder layout

```text
EdgeCountAI/
  count_vehicles.py
  requirements.txt
  README.md
  .gitignore
  .venv/          # local Python environment, ignored
  input_images/  # original images, ignored
  models/        # downloaded weights, ignored
  outputs/       # current and comparison runs, ignored
  archive/       # older local samples and runs, ignored
  results/       # legacy local output, ignored
```

## Accuracy and limitations

Counts represent model detections, not verified ground truth. Distant or occluded vehicles may be missed; vehicles can be misclassified or counted more than once. This pipeline only requests car, motorcycle, bus, and truck classes, so it does not provide a dedicated auto-rickshaw category.

Local comparisons showed that higher resolution recovered more distant detections, while stricter confidence thresholds also removed real buses. More detections do not necessarily mean greater accuracy. Validate on manually labelled images before claiming accuracy or selecting production settings. Raspberry Pi performance has not been measured.

## Dependencies

Built with Python, Ultralytics, OpenCV, and the Python standard library. Dependency versions in requirements.txt match the installed direct packages used for this proof of concept; the full transitive environment is not locked.

Third-party code and pretrained weights retain their own licenses. See the Ultralytics package and model distribution for their terms. Sample photos and model weights are not distributed in this repository.
