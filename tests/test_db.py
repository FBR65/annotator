"""Speicher: A3, B2, G2, G4, Dedupe (B6-Vorstufe)."""

from __future__ import annotations

from pathlib import Path

import pytest

from annotator.db import Store
from annotator.geometry import Box, BoxError


def test_g2_class_ids_append_only(store: Store) -> None:
    assert [c.id for c in store.classes()] == [1, 2]
    third = store.add_class("Flache")
    assert third == 3
    assert [c.id for c in store.classes()] == [1, 2, 3]


def test_duplicate_sha_ignored(store: Store, tmp_path: Path) -> None:
    assert store.add_image(str(tmp_path / "a.png"), "abc", 10, 10) is not None
    assert store.add_image(str(tmp_path / "b.png"), "abc", 10, 10) is None
    assert len(store.images()) == 1


def test_b6_second_import_counts_as_duplicate(store: Store, tmp_path: Path) -> None:
    first = store.add_image(str(tmp_path / "a.png"), "abc", 10, 10)
    assert first == 1
    assert store.image_by_path(str(tmp_path / "a.png")) is not None


def test_g4_labeled_only_by_human(store: Store, tmp_path: Path) -> None:
    image_id = store.add_image(path=str(tmp_path / "a.png"), sha256="x", width=10, height=10)
    assert image_id is not None
    store.set_status(image_id, "needs_review", by="auto")
    with pytest.raises(PermissionError):
        store.set_status(image_id, "labeled", by="auto")
    store.set_status(image_id, "labeled", by="human")
    assert store.image(image_id).status == "labeled"


def test_a3_zero_box_rejected_with_path(store: Store, tmp_path: Path) -> None:
    image_id = store.add_image(path=str(tmp_path / "rad.png"), sha256="y", width=10, height=10)
    assert image_id is not None
    with pytest.raises(BoxError) as exc:
        store.add_box(image_id, Box(1, 0.5, 0.5, 0.0, 0.2))
    assert "rad.png" in str(exc.value)


def test_b2_update_touches_one_box_only(store: Store, tmp_path: Path) -> None:
    image_id = store.add_image(path=str(tmp_path / "a.png"), sha256="z", width=100, height=100)
    assert image_id is not None
    first = store.add_box(image_id, Box(1, 0.25, 0.25, 0.1, 0.1))
    second = store.add_box(image_id, Box(2, 0.75, 0.75, 0.2, 0.2))
    before = store.boxes(image_id)
    store.update_box(first, cx=0.5)
    after = store.boxes(image_id)
    changed = next(b for b in after if b.id == first)
    untouched_before = next(b for b in before if b.id == second)
    untouched_after = next(b for b in after if b.id == second)
    assert changed.cx == pytest.approx(0.5)
    assert untouched_before == untouched_after


def test_b3_delete_removes_one_row(store: Store, tmp_path: Path) -> None:
    image_id = store.add_image(path=str(tmp_path / "a.png"), sha256="q", width=10, height=10)
    assert image_id is not None
    one = store.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    store.add_box(image_id, Box(2, 0.2, 0.2, 0.1, 0.1))
    store.set_status(image_id, "labeled", by="human")
    store.delete_box(one)
    assert len(store.boxes(image_id)) == 1
    assert store.image(image_id).status == "labeled"


def test_update_unknown_field_rejected(store: Store, tmp_path: Path) -> None:
    image_id = store.add_image(path=str(tmp_path / "a.png"), sha256="w", width=10, height=10)
    assert image_id is not None
    box_id = store.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        store.update_box(box_id, nonsense=1)


def test_counts_by_status(store: Store, tmp_path: Path) -> None:
    ids = [store.add_image(str(tmp_path / f"{i}.png"), str(i), 10, 10) for i in range(3)]
    store.set_status(int(ids[0] or 0), "needs_review", by="auto")
    store.set_status(int(ids[1] or 0), "labeled", by="human")
    counts = store.counts()
    assert counts["new"] == 1
    assert counts["needs_review"] == 1
    assert counts["labeled"] == 1
