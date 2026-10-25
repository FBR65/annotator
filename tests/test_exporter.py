"""Kriterien D1 bis D9: Export nach YOLO und COCO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from annotator.db import Store
from annotator.exporter import export_dataset, split_ids
from annotator.geometry import Box
from annotator.importer import sha256_file
from tests.conftest import make_image


def seed_store(store: Store, tmp_path: Path, n: int = 20, box_every: int = 1) -> None:
    for i in range(n):
        path = make_image(tmp_path / "bilder" / f"bild_{i:02d}.png", size=(80, 60))
        image_id = store.add_image(str(path), sha256_file(path), 80, 60)
        assert image_id is not None
        if i % box_every == 0:
            store.add_box(image_id, Box(i % 2 + 1, 0.5, 0.5, 0.25, 0.25))
        store.set_status(image_id, "labeled", by="human")


def test_d1_split_counts_and_structure(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=20)
    out = tmp_path / "export"
    result = export_dataset(store, out, fmt="yolo", val_fraction=0.2, seed=42)
    assert (result.train, result.val) == (16, 4)
    for part in ("train", "val"):
        assert (out / "images" / part).is_dir()
        assert (out / "labels" / part).is_dir()
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 2" in yaml_text
    assert str(out / "images" / "train") in yaml_text


def test_d2_every_image_has_label_file(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=10, box_every=2)
    out = tmp_path / "export"
    export_dataset(store, out, fmt="yolo", val_fraction=0.2, seed=7)
    images = sorted(p.stem for part in ("train", "val") for p in (out / "images" / part).glob("*"))
    labels = sorted(
        p.stem for part in ("train", "val") for p in (out / "labels" / part).glob("*.txt")
    )
    assert images == labels
    assert len(images) == 10
    empty = [
        p
        for part in ("train", "val")
        for p in (out / "labels" / part).glob("*.txt")
        if p.stat().st_size == 0
    ]
    assert len(empty) == 5


def test_d3_split_is_deterministic(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=20)
    first = export_dataset(store, tmp_path / "e1", fmt="yolo", seed=1)
    second = export_dataset(store, tmp_path / "e2", fmt="yolo", seed=1)
    third = export_dataset(store, tmp_path / "e3", fmt="yolo", seed=2)
    assert first.files == second.files
    assert (first.train, first.val) == (second.train, second.val)
    assert (third.train, third.val) == (first.train, first.val)
    val_first = sorted(p.name for p in (tmp_path / "e1" / "images" / "val").glob("*"))
    val_second = sorted(p.name for p in (tmp_path / "e2" / "images" / "val").glob("*"))
    val_third = sorted(p.name for p in (tmp_path / "e3" / "images" / "val").glob("*"))
    assert val_first == val_second
    assert val_first != val_third


def test_d4_originals_untouched(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=4)
    before = {
        p: (sha256_file(p), p.stat().st_mtime_ns) for p in (tmp_path / "bilder").glob("*.png")
    }
    export_dataset(store, tmp_path / "export", fmt="coco", force=True)
    after = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in (tmp_path / "bilder").glob("*.png")}
    assert before == after


def test_d5_existing_target_needs_force(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=2)
    out = tmp_path / "export"
    out.mkdir()
    (out / "alt.txt").write_text("alt", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_dataset(store, out, fmt="yolo")
    result = export_dataset(store, out, fmt="yolo", force=True)
    assert result.train + result.val == 2


def test_d6_coco_structure(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=10)
    out = tmp_path / "export"
    result = export_dataset(store, out, fmt="coco", val_fraction=0.2, seed=3)
    assert (result.train, result.val) == (8, 2)
    for part in ("train", "val"):
        data = json.loads((out / part / "_annotations.coco.json").read_text(encoding="utf-8"))
        assert set(data) >= {"images", "annotations", "categories"}
        for ann in data["annotations"]:
            assert ann["iscrowd"] == 0
            x, y, w, h = ann["bbox"]
            assert w > 0 and h > 0
            assert ann["area"] == pytest.approx(w * h)
            assert x >= 0 and y >= 0
        assert all(image["file_name"] for image in data["images"])
        assert (out / part / data["images"][0]["file_name"]).is_file()


def test_d7_bbox_back_to_yolo(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=6)
    out = tmp_path / "export"
    export_dataset(store, out, fmt="coco", val_fraction=0.0, seed=1)
    data = json.loads((out / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_id = {image["id"]: image for image in data["images"]}
    image = store.images()[0]
    ann = next(a for a in data["annotations"] if a["image_id"] == 1)
    x, y, w, h = ann["bbox"]
    assert image.width == by_id[1]["width"]
    assert x + w / 2 == pytest.approx(store.boxes(image.id)[0].cx * image.width, abs=1.0)
    assert y + h / 2 == pytest.approx(store.boxes(image.id)[0].cy * image.height, abs=1.0)
    assert w == pytest.approx(store.boxes(image.id)[0].w * image.width, abs=1.0)


def test_d8_category_ids_start_at_one(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=4)
    out = tmp_path / "export"
    export_dataset(store, out, fmt="coco", val_fraction=0.0, seed=1)
    data = json.loads((out / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in data["categories"]] == [1, 2]
    assert {a["image_id"] for a in data["annotations"]} <= {i["id"] for i in data["images"]}


def test_d9_both_formats_same_folder(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=6)
    out = tmp_path / "export"
    result = export_dataset(store, out, fmt="both", val_fraction=0.0, seed=1)
    assert result.train == 6
    assert (out / "data.yaml").is_file()
    assert (out / "train" / "_annotations.coco.json").is_file()
    yolo_names = {p.name for p in (out / "images" / "train").glob("*")}
    coco_names = {
        image["file_name"]
        for image in json.loads(
            (out / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
        )["images"]
    }
    assert yolo_names == coco_names


def test_split_ids_edge_cases() -> None:
    assert split_ids([], 0.2, 1) == ([], [])
    train, val = split_ids(list(range(1)), 0.2, 1)
    assert len(train) + len(val) == 1
    train, val = split_ids(list(range(10)), 0.0, 1)
    assert (len(train), len(val)) == (10, 0)
    train, val = split_ids(list(range(10)), 0.5, 1)
    assert (len(train), len(val)) == (5, 5)


def test_export_only_uses_database(store: Store, tmp_path: Path) -> None:
    seed_store(store, tmp_path, n=3)
    orphan = make_image(tmp_path / "bilder" / "fremd.png")
    out = tmp_path / "export"
    result = export_dataset(store, out, fmt="yolo", val_fraction=0.0, seed=1)
    names = {Path(f).name for f in result.files}
    assert orphan.name not in names
    assert len(names) == 3
    assert hashlib.sha256(b"x").hexdigest()
