"""Angriffspass: kaputte Dateien, riesige Bilder, gleichzeitige Schreiber."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image

from annotator.db import Store
from annotator.geometry import Box
from annotator.importer import import_folder, sha256_file
from tests.conftest import make_image


def test_empty_file_is_skipped(tmp_path: Path, store: Store) -> None:
    (tmp_path / "leer.png").write_bytes(b"")
    make_image(tmp_path / "gut.png")
    result = import_folder(store, tmp_path)
    assert result.added == 1
    assert result.skipped == 1
    assert len(store.images()) == 1


def test_fake_png_is_skipped(tmp_path: Path, store: Store) -> None:
    (tmp_path / "attrappe.png").write_text("das ist kein Bild", encoding="utf-8")
    result = import_folder(store, tmp_path)
    assert result.added == 0
    assert result.skipped == 1


def test_huge_image_is_imported(tmp_path: Path, store: Store) -> None:
    ziel = tmp_path / "riesig.png"
    Image.new("L", (4000, 3000), 0).save(ziel)
    result = import_folder(store, tmp_path)
    assert result.added == 1
    gespeichert = store.images()[0]
    assert (gespeichert.width, gespeichert.height) == (4000, 3000)


def test_concurrent_writes_same_image(store: Store, tmp_path: Path) -> None:
    path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    fehler: list[Exception] = []

    def schreiben(x: float) -> None:
        try:
            store.add_box(image_id, Box(1, x, 0.5, 0.1, 0.1))
        except Exception as exc:  # Test sammelt echte Fehler
            fehler.append(exc)

    threads = [threading.Thread(target=schreiben, args=(0.1 + index * 0.1,)) for index in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert fehler == []
    assert len(store.boxes(image_id)) == 5


def test_box_outside_image_is_rejected(store: Store, tmp_path: Path) -> None:
    path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    with pytest.raises(ValueError):
        store.add_box(image_id, Box(1, 1.5, 0.5, 0.2, 0.2))
