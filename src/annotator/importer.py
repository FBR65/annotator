"""Import von Bildordnern und vorhandenen YOLO-Labels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .db import Store
from .geometry import Box, BoxError, from_yolo_line

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


@dataclass(frozen=True)
class ImportResult:
    added: int
    duplicates: int
    skipped: int
    unreadable: int = 0


@dataclass(frozen=True)
class LabelImportResult:
    files: int
    boxes: int
    rejected: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(root: Path, recursive: bool) -> tuple[list[Path], int]:
    if not root.is_dir():
        raise NotADirectoryError(f"Kein Ordner: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    images: list[Path] = []
    skipped = 0
    for candidate in sorted(candidates, key=lambda item: str(item)):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(candidate)
        else:
            skipped += 1
    return images, skipped


def iter_images(root: Path, recursive: bool) -> list[Path]:
    images, _ = _scan(root, recursive)
    return images


def _size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as handle:
            return handle.size
    except (UnidentifiedImageError, OSError):
        return None


def import_folder(store: Store, root: Path, recursive: bool = False) -> ImportResult:
    images, skipped = _scan(root, recursive)
    added = 0
    duplicates = 0
    unreadable = 0
    for path in images:
        size = _size(path)
        if size is None:
            skipped += 1
            unreadable += 1
            continue
        width, height = size
        new_id = store.add_image(str(path.resolve()), sha256_file(path), width, height)
        if new_id is None:
            duplicates += 1
        else:
            added += 1
    return ImportResult(added=added, duplicates=duplicates, skipped=skipped, unreadable=unreadable)


def read_label_file(path: Path, n_classes: int) -> list[Box]:
    """Liest eine YOLO-Labeldatei. Klasse außerhalb 0..n-1 wird abgelehnt."""
    boxes: list[Box] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            box = from_yolo_line(line)
        except BoxError as exc:
            raise BoxError(f"{path}: Zeile {number}: {exc}") from exc
        if not 0 <= box.class_id < n_classes:
            raise BoxError(
                f"{path}: Zeile {number}: Klasse {box.class_id} außerhalb 0..{n_classes - 1}"
            )
        boxes.append(box)
    return boxes


def import_labels(store: Store, root: Path, recursive: bool = False) -> LabelImportResult:
    """Liest vorhandene .txt-Labels neben Bildern und legt sie als menschliche an."""
    if not root.is_dir():
        raise NotADirectoryError(f"Kein Ordner: {root}")
    classes = store.classes()
    files = 0
    boxes = 0
    rejected = 0
    labels = root.rglob("*.txt") if recursive else root.glob("*.txt")
    for label_path in sorted(labels, key=lambda item: str(item)):
        files += 1
        image_path = label_path.with_suffix(".png")
        if not image_path.is_file():
            rejected += 1
            continue
        image = store.image_by_path(str(image_path.resolve()))
        if image is None:
            rejected += 1
            continue
        try:
            parsed = read_label_file(label_path, len(classes))
        except BoxError:
            rejected += 1
            continue
        for box in parsed:
            store.add_box(
                image.id,
                Box(classes[box.class_id].id, box.cx, box.cy, box.w, box.h),
                source="human",
                state="accepted",
            )
            boxes += 1
        store.set_status(image.id, "labeled", by="human")
    return LabelImportResult(files=files, boxes=boxes, rejected=rejected)
