"""Gemeinsame Fixtures: temporärer Speicher und synthetische Bilder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from annotator.db import Store


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "db" / "test.db")
    s.add_class("Messer")
    s.add_class("Waffe")
    yield s
    s.close()


def make_image(path: Path, size: tuple[int, int] = (64, 48), color: str = "black") -> Path:
    """Erzeugt ein PNG. Ein Pixel hängt am Dateinamen, damit Bilder nicht gleich sind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    marker = int(hashlib.sha256(path.name.encode()).hexdigest()[:6], 16)
    image.putpixel((marker % size[0], (marker // size[0]) % size[1]), (marker % 256, 17, 33))
    image.save(path)
    return path
