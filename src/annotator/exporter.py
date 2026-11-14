"""Export nach YOLO und COCO."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .db import ImageRow, Store
from .geometry import Box, to_yolo_line, validate_box

FORMATS = ("yolo", "coco", "both")


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    train: int
    val: int
    files: list[str]


def split_ids(ids: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction muss in [0, 1) liegen")
    if not ids:
        return [], []
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    if val_fraction > 0 and len(shuffled) > 1 and n_val == 0:
        n_val = 1
    n_val = min(n_val, max(len(shuffled) - 1, 0)) if len(shuffled) > 1 else 0
    validation = shuffled[:n_val]
    training = shuffled[n_val:]
    return sorted(training), sorted(validation)


def _prepare(out: Path, force: bool) -> None:
    if out.exists():
        if not force:
            raise FileExistsError(f"Zielordner existiert bereits: {out}. Nutzen Sie --force.")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def _write_yolo_part(store: Store, out: Path, part: str, images: list[ImageRow]) -> list[str]:
    images_dir = out / "images" / part
    labels_dir = out / "labels" / part
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_ids = [item.id for item in store.classes()]
    written: list[str] = []
    for image in images:
        source = Path(image.path)
        target = images_dir / source.name
        shutil.copy2(source, target)
        lines: list[str] = []
        for row in store.boxes(image.id):
            box = Box(row.class_id, row.cx, row.cy, row.w, row.h)
            validate_box(box, str(source))
            if row.class_id not in class_ids:
                raise ValueError(f"Klasse {row.class_id} fehlt in der Klassenliste: {source}")
            index = class_ids.index(row.class_id)
            lines.append(to_yolo_line(Box(index, box.cx, box.cy, box.w, box.h)))
        (labels_dir / f"{source.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        written.append(target.name)
    return written


def _write_yolo_yaml(store: Store, out: Path) -> None:
    names = ", ".join(_yaml_quote(item.name) for item in store.classes())
    text = (
        f"path: {out.resolve()}\n"
        f"train: {(out / 'images' / 'train').resolve()}\n"
        f"val: {(out / 'images' / 'val').resolve()}\n"
        f"nc: {len(store.classes())}\n"
        f"names: [{names}]\n"
    )
    (out / "data.yaml").write_text(text, encoding="utf-8")


def _yaml_quote(name: str) -> str:
    return '"' + name.replace('"', '\\"') + '"'


def _write_coco_part(store: Store, out: Path, part: str, images: list[ImageRow]) -> list[str]:
    part_dir = out / part
    part_dir.mkdir(parents=True, exist_ok=True)
    images_json: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    files: list[str] = []
    annotation_id = 1
    for local_id, image in enumerate(images, start=1):
        source = Path(image.path)
        target = part_dir / source.name
        shutil.copy2(source, target)
        files.append(target.name)
        images_json.append(
            {
                "id": local_id,
                "file_name": source.name,
                "width": image.width,
                "height": image.height,
            }
        )
        for box in store.boxes(image.id):
            width_px = box.w * image.width
            height_px = box.h * image.height
            x_min = (box.cx - box.w / 2) * image.width
            y_min = (box.cy - box.h / 2) * image.height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": local_id,
                    "category_id": box.class_id,
                    "bbox": [x_min, y_min, width_px, height_px],
                    "area": width_px * height_px,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "info": {"description": "annotator Export", "version": "1.0"},
        "licenses": [],
        "images": images_json,
        "annotations": annotations,
        "categories": [
            {"id": item.id, "name": item.name, "supercategory": "object"}
            for item in store.classes()
        ],
    }
    (part_dir / "_annotations.coco.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return files


def export_dataset(
    store: Store,
    out: Path,
    fmt: str = "coco",
    val_fraction: float = 0.2,
    seed: int = 42,
    force: bool = False,
) -> ExportResult:
    if fmt not in FORMATS:
        raise ValueError(f"Unbekanntes Format: {fmt}. Erlaubt: {', '.join(FORMATS)}")
    if out.exists() and not force:
        raise FileExistsError(f"Zielordner existiert bereits: {out}. Nutzen Sie --force.")
    if not store.classes():
        raise ValueError("Keine Klassen angelegt. Export nicht möglich.")
    _prepare(out, force)
    all_images = store.images("labeled")
    if not all_images:
        raise ValueError("Keine fertig annotierten Bilder. Export nicht möglich.")
    train_ids, val_ids = split_ids([image.id for image in all_images], val_fraction, seed)
    by_id = {image.id: image for image in all_images}
    train = [by_id[image_id] for image_id in train_ids]
    val = [by_id[image_id] for image_id in val_ids]
    files: list[str] = []
    if fmt in {"yolo", "both"}:
        files += _write_yolo_part(store, out, "train", train)
        files += _write_yolo_part(store, out, "val", val)
        _write_yolo_yaml(store, out)
    if fmt in {"coco", "both"}:
        files += _write_coco_part(store, out, "train", train)
        files += _write_coco_part(store, out, "val", val)
    meta = {
        "format": fmt,
        "train": len(train),
        "val": len(val),
        "seed": seed,
        "only_labeled": True,
        "skipped_unlabeled": len(store.images()) - len(all_images),
    }
    (out / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ExportResult(out_dir=out, train=len(train), val=len(val), files=files)
