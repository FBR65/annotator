"""Box-Geometrie und YOLO-Textformat."""

from __future__ import annotations

import math
from dataclasses import dataclass


class BoxError(ValueError):
    """Ungültige Box."""


@dataclass(frozen=True)
class Box:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


def to_yolo_line(box: Box) -> str:
    """Eine Zeile im YOLO-Format: class cx cy w h, normalisiert, 6 Stellen."""
    return f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"


def from_yolo_line(line: str) -> Box:
    """Liest eine YOLO-Zeile. Wirft BoxError bei falscher Feldzahl oder Zahl."""
    parts = line.split()
    if len(parts) != 5:
        raise BoxError(f"YOLO-Zeile braucht 5 Felder, gefunden {len(parts)}: {line!r}")
    try:
        class_id = int(parts[0])
        cx, cy, w, h = (float(value) for value in parts[1:])
    except ValueError as exc:
        raise BoxError(f"YOLO-Zeile nicht lesbar: {line!r}") from exc
    return Box(class_id, cx, cy, w, h)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _min_extent(low: float, high: float, limit: float, min_size: float) -> tuple[float, float]:
    if high - low >= min_size:
        return low, high
    center = (low + high) / 2
    low = center - min_size / 2
    high = center + min_size / 2
    if low < 0:
        high -= low
        low = 0.0
    if high > limit:
        low -= high - limit
        high = limit
    return max(low, 0.0), min(high, limit)


def clamp_box(box: Box, width: int, height: int, min_px: int = 1) -> Box:
    """Beschneidet eine Box auf das Bild. Mindestgröße min_px Pixel."""
    if width <= 0 or height <= 0:
        raise BoxError(f"Bildmaße ungültig: {width}x{height}")
    x0 = _clamp((box.cx - box.w / 2) * width, 0.0, float(width))
    x1 = _clamp((box.cx + box.w / 2) * width, 0.0, float(width))
    y0 = _clamp((box.cy - box.h / 2) * height, 0.0, float(height))
    y1 = _clamp((box.cy + box.h / 2) * height, 0.0, float(height))
    x0, x1 = _min_extent(x0, x1, float(width), float(min_px))
    y0, y1 = _min_extent(y0, y1, float(height), float(min_px))
    return Box(
        class_id=box.class_id,
        cx=(x0 + x1) / 2 / width,
        cy=(y0 + y1) / 2 / height,
        w=(x1 - x0) / width,
        h=(y1 - y0) / height,
    )


def validate_box(box: Box, image_path: str) -> None:
    """Prüft eine Box vor dem Speichern. Die Meldung nennt die Datei."""
    if box.class_id < 0:
        raise BoxError(f"Ungültige Klasse {box.class_id} in {image_path}")
    values = (box.cx, box.cy, box.w, box.h)
    if not all(math.isfinite(value) for value in values):
        raise BoxError(f"Ungültige Box (kein endlicher Wert) in {image_path}")
    if box.w <= 0 or box.h <= 0:
        raise BoxError(f"Ungültige Box (Breite oder Höhe 0) in {image_path}")
    if not (0.0 <= box.cx <= 1.0 and 0.0 <= box.cy <= 1.0):
        raise BoxError(f"Ungültige Box (Mittelpunkt außerhalb 0..1) in {image_path}")
    if box.w > 1.0 or box.h > 1.0:
        raise BoxError(f"Ungültige Box (Breite oder Höhe über 1) in {image_path}")


def iou(a: Box, b: Box) -> float:
    """Schnittmenge geteilt durch Vereinigung, Klassen bleiben außen vor."""
    ax0, ax1 = a.cx - a.w / 2, a.cx + a.w / 2
    ay0, ay1 = a.cy - a.h / 2, a.cy + a.h / 2
    bx0, bx1 = b.cx - b.w / 2, b.cx + b.w / 2
    by0, by1 = b.cy - b.h / 2, b.cy + b.h / 2
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union
