"""Kriterien B6, B7, A4: Import von Ordnern und vorhandenen YOLO-Labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from annotator.db import Store
from annotator.geometry import BoxError
from annotator.importer import (
    import_folder,
    import_labels,
    iter_images,
    read_label_file,
    sha256_file,
)
from tests.conftest import make_image


def test_b7_recursive_only_with_flag(tmp_path: Path, store: Store) -> None:
    make_image(tmp_path / "a.png")
    make_image(tmp_path / "unter" / "b.png")
    assert len(iter_images(tmp_path, recursive=False)) == 1
    assert len(iter_images(tmp_path, recursive=True)) == 2


def test_b7_extension_case_insensitive_and_others_counted(tmp_path: Path, store: Store) -> None:
    make_image(tmp_path / "gross.PNG")
    make_image(tmp_path / "klein.jpg")
    (tmp_path / "notiz.txt").write_text("kein Bild", encoding="utf-8")
    result = import_folder(store, tmp_path)
    assert result.added == 2
    assert result.skipped == 1


def test_b7_symlink_not_followed(tmp_path: Path, store: Store) -> None:
    real = make_image(tmp_path / "echt.png")
    link = tmp_path / "verknuepfung.png"
    link.symlink_to(real)
    found = iter_images(tmp_path, recursive=False)
    assert [p.name for p in found] == ["echt.png"]


def test_b6_import_twice_adds_nothing(tmp_path: Path, store: Store) -> None:
    for name in ("a.png", "b.png", "c.png"):
        make_image(tmp_path / name)
    first = import_folder(store, tmp_path)
    assert (first.added, first.duplicates) == (3, 0)
    assert all(image.status == "new" for image in store.images())
    second = import_folder(store, tmp_path)
    assert (second.added, second.duplicates) == (0, 3)
    assert len(store.images()) == 3


def test_b7_umlaut_and_space_in_path(tmp_path: Path, store: Store) -> None:
    target = tmp_path / "Prüfung Ordner"
    make_image(target / "Bild äöü.png")
    result = import_folder(store, target)
    assert result.added == 1
    assert "Prüfung" in store.images()[0].path


def test_sha256_is_stable(tmp_path: Path) -> None:
    path = make_image(tmp_path / "a.png")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_a4_out_of_range_class_rejected_file_untouched(tmp_path: Path, store: Store) -> None:
    label = tmp_path / "a.txt"
    content = "5 0.500000 0.500000 0.100000 0.100000\n"
    label.write_text(content, encoding="utf-8")
    n_classes = len(store.classes())
    with pytest.raises(BoxError) as exc:
        read_label_file(label, n_classes)
    assert "a.txt" in str(exc.value)
    assert label.read_text(encoding="utf-8") == content


def test_import_labels_creates_boxes_and_logs_rejects(tmp_path: Path, store: Store) -> None:
    image_path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(image_path), sha256_file(image_path), 64, 48)
    assert image_id is not None
    (tmp_path / "bild.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
    (tmp_path / "kaputt.txt").write_text("7 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    result = import_labels(store, tmp_path)
    assert result.boxes == 1
    assert result.rejected == 1
    boxes = store.boxes(image_id)
    assert len(boxes) == 1
    assert (boxes[0].cx, boxes[0].cy) == pytest.approx((0.5, 0.5))
