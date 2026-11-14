# Annotator

Local tool for image annotation, YOLO training and automatic pre-annotation.

The workflow is a loop. A human annotates a small number of images, the tool
trains a YOLO model on them, and the model pre-annotates the remaining images.
Where a prediction is not unambiguous, the image is put into a review queue for
a human instead of being accepted silently.

Source images are never copied, moved or modified. The database stores the
absolute path and the SHA-256 digest only. The web server binds to `127.0.0.1`
exclusively.

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.12 (managed by uv) |
| uv | 0.12 or newer |
| GPU | Optional. NVIDIA with CUDA for training, otherwise `--device cpu` |
| Network | Only for the first `--init pretrained` run (downloads `yolov8n.pt`) |
| Disk | One SQLite file for state; images stay in their source folder |

## Setup

```
uv sync
uv run annotator --help
```

## Commands

Every command takes `--db <path>` before the subcommand. Default is
`annotator.db`. All steps must use the same database file.

### `import` — read an image folder

```
annotator --db annotator.db import ~/images --recursive
annotator --db annotator.db import ~/images --labels
```

| Option | Effect |
|---|---|
| `--recursive` | Include subfolders |
| `--labels` | Adopt existing YOLO `.txt` labels next to the images. A class index outside the valid range rejects the file and leaves it untouched |

Supported extensions: `.jpg .jpeg .png .bmp .webp .tif .tiff`, case-insensitive.
Symbolic links are not followed. Duplicates are detected by SHA-256 and skipped.

### `serve` — browser interface

```
annotator --db annotator.db serve --port 8765
```

Open `http://127.0.0.1:8765/`. `--host` accepts `127.0.0.1`, `localhost` and
`::1`; any other address is rejected with exit code 1. The interface itself is
in German.

| Control | Effect |
|---|---|
| "Bilder importieren" plus folder path | Imports a folder; checkboxes for subfolders and existing labels |
| "Datensatz schreiben" plus target path | Exports to YOLO, COCO or both; checkbox to overwrite |
| Drag on the image | Draw a new box |
| Click on a box | Select the box |
| Drag inside a box | Move the box |
| Drag a corner | Resize the box |
| `1`–`9` | Assign a class to the selected box |
| `Del` | Delete the box |
| `Ctrl+Z` or "Rückgängig" | Undo the last change of this session, up to 100 steps |
| `A` / `D` or arrow keys | Previous / next image |
| `Space` | Accept the proposal (status becomes `labeled`) |
| `R` | Reload the list |
| "Ansicht" | Filter by status, for example "Nur Prüfung" |

Every change is written immediately and logged as one line in `annotator.log`.
The undo history lives in the browser tab and is lost when the page is closed.

### `review` — review queue on the console

```
annotator --db annotator.db review
```

Prints count and path of every image with status `needs_review`, plus the number
of images per status.

### `train` — train YOLO

```
annotator --db annotator.db train --dataset ./dataset --epochs 100 --init scratch
```

| Option | Default | Effect |
|---|---|---|
| `--cwd` | `.` | Working folder; weights are written to `<cwd>/models/run_<id>/` |
| `--dataset` | `<cwd>/dataset` | Folder containing `data.yaml` from `export` |
| `--epochs` | 100 | Training epochs |
| `--init` | `scratch` | `scratch` uses `yolov8n.yaml`: no external weights, no network. `pretrained` uses `yolov8n.pt` (COCO weights) |
| `--imgsz` | 640 | Training image size |
| `--batch` | 4 | Batch size; 4 is safe on 8 GB VRAM |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dry-run` | off | Prints parameters and the hint, does not train |

Before each run the tool prints which init mode fits the current data volume.
Below 200 labeled images, `pretrained` usually performs better. A model trained
from scratch on a few dozen images reports `mAP50 = 0.0` even after 80 epochs;
that follows from the data volume, not from a defect.

Output: `models/run_<id>/best.pt`, `models/run_<id>/train.log`, and one row in
the `runs` table with parameters and metrics.

### `auto` — automatic pre-annotation

```
annotator --db annotator.db auto --model models/run_4/best.pt --cwd ./work
```

| Option | Default | Effect |
|---|---|---|
| `--model` | latest `models/run_*/best.pt` | Weights file |
| `--cwd` | `.` | Target folder for `review_queue.json` and `annotator.log` |
| `--conf-accept` | 0.50 | At or above this confidence a detection counts as certain |
| `--conf-review` | 0.25 | Below this confidence a detection counts as uncertain |
| `--device` | `cuda` | `cuda` or `cpu` |

Only images with status `new` are processed. Images that are already labeled are
never overwritten.

**When an image goes to human review** (first matching rule wins):

| Rule | Condition | Reported reason |
|---|---|---|
| 1 | No detection | "Kein Objekt gefunden" |
| 2 | Any box below `--conf-review` | "Erkennung unsicher" |
| 3 | Two boxes of the same class, IoU ≥ 0.70 | "Doppelte Erkennung" |
| 4 | More than 50 boxes | "Zu viele Objekte" |
| 5 | Any box below `--conf-accept` | "Grenzfall" |
| 6 | Otherwise | Certain, status `auto_ok` |

**Human notification** — four channels, always all of them:

1. Console: number of processed, certain and flagged images, plus the reasons.
2. `annotator.log` in the working folder: one JSON line per image with reason.
3. `review_queue.json` in the working folder: list of path and reason.
4. Interface: banner with the count and the "Nur Prüfung" filter.

A model error on a single image does not abort the run. The image is set to
`needs_review` and the run continues; the process then exits with code 3.

### `export` — write a dataset

```
annotator --db annotator.db export --out ./dataset --format both --val-fraction 0.2
```

| Option | Default | Effect |
|---|---|---|
| `--out` | required | Target folder |
| `--format` | `coco` | `coco` (RF-DETR), `yolo`, or `both` |
| `--val-fraction` | 0.2 | Share of validation images |
| `--seed` | 42 | Same value produces the same split |
| `--force` | off | Overwrite an existing target folder |

Only images with status `labeled` are exported. Unlabeled images would become
false negatives and degrade training; their count is recorded in
`<out>/export_meta.json` as `skipped_unlabeled`.

| Format | Result |
|---|---|
| `yolo` | `images/{train,val}`, `labels/{train,val}`, `data.yaml` with absolute paths |
| `coco` | `train/_annotations.coco.json`, `val/_annotations.coco.json`, images alongside |
| `both` | Both structures in the same folder |

Class indices are converted on export: the database uses 1-based class ids, the
YOLO format uses 0-based indices, COCO keeps the database id.

## Image states

```
new --auto, all boxes certain------> auto_ok --human accepts--------> labeled
new --auto, uncertain or empty-----> needs_review --human-----------> labeled
new --human annotates--------------> labeled
```

`labeled` is only ever set by a human action. `auto_ok` is a high-confidence
proposal, not a finished label.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Error (message on the console) |
| 2 | Nothing to do (no model, too few labeled images) |
| 3 | Partial failure: `import` with unreadable image files, `auto` with images the model could not process |

On exit code 3 the message states how many files were affected. No data is lost:
unreadable files are left untouched, failed pre-annotations enter the review
queue.

## Files

| File | Content |
|---|---|
| `annotator.db` | Database: images, classes, boxes, runs |
| `annotator.log` | JSON lines for changes and pre-annotation runs |
| `review_queue.json` | Current review queue |
| `models/run_<id>/` | Weights and training log per run |
| `<export>/` | Dataset in YOLO and/or COCO layout |

## Limits

- One operator, one machine. No user management, no authentication; the loopback
  binding is the only protection.
- Bounding boxes on single images only. No videos, no masks.
- A model trained from scratch needs substantially more than 100 images to be
  usable. For small datasets use `--init pretrained`.
- `labeled` is enforced by the program path, not by a signature; direct database
  access can bypass it.
- `app.js` has no automated test. The contract is the HTTP API.
- `ultralytics` is licensed under **AGPL-3.0**. Distributing this tool triggers
  AGPL obligations; own code is MIT-compatible.

## Contract and evidence

- `SPEC.md` — approved specification with acceptance criteria A1–G4.
- `EVIDENCE.md` — verification report: 111 tests, one real GPU training run,
  7/7 hand-written mutants killed, 90 % line coverage, end-to-end run on synthetic
  images.
