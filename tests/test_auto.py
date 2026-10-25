"""Kriterien C7, F1 bis F4: automatische Vorannotation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotator.autolabel import run_auto
from annotator.db import Store
from annotator.decide import DecideConfig, Detection
from annotator.geometry import Box
from annotator.importer import sha256_file
from tests.conftest import make_image


def seed_new(store: Store, tmp_path: Path, n: int) -> list[int]:
    ids: list[int] = []
    for i in range(n):
        path = make_image(tmp_path / "bilder" / f"a_{i:02d}.png", size=(64, 48))
        image_id = store.add_image(str(path), sha256_file(path), 64, 48)
        assert image_id is not None
        ids.append(image_id)
    return ids


def test_f2_nothing_found_goes_to_review(store: Store, tmp_path: Path) -> None:
    ids = seed_new(store, tmp_path, n=3)
    result = run_auto(
        store,
        lambda path: [],
        queue_path=tmp_path / "queue.json",
        log_path=tmp_path / "annotator.log",
    )
    assert result.processed == 3
    assert result.review == 3
    assert result.reasons == {"kein_objekt": 3}
    for image_id in ids:
        assert store.image(image_id).status == "needs_review"
        assert store.boxes(image_id) == []


def test_f1_labeled_images_untouched(store: Store, tmp_path: Path) -> None:
    ids = seed_new(store, tmp_path, n=2)
    store.add_box(ids[0], Box(1, 0.5, 0.5, 0.2, 0.2))
    store.set_status(ids[0], "labeled", by="human")
    before_boxes = store.boxes(ids[0])
    result = run_auto(
        store,
        lambda path: [Detection(1, 0.5, 0.5, 0.3, 0.3, 0.99)],
        queue_path=tmp_path / "queue.json",
        log_path=tmp_path / "annotator.log",
    )
    assert result.processed == 1
    assert store.boxes(ids[0]) == before_boxes
    assert store.image(ids[0]).status == "labeled"
    assert store.image(ids[1]).status == "auto_ok"


def test_c7_counts_match_everywhere(store: Store, tmp_path: Path) -> None:
    ids = seed_new(store, tmp_path, n=4)

    def predictor(path: str) -> list[Detection]:
        if path.endswith("a_00.png"):
            return [Detection(1, 0.5, 0.5, 0.2, 0.2, 0.9)]
        if path.endswith("a_01.png"):
            return [Detection(1, 0.5, 0.5, 0.2, 0.2, 0.35)]
        return []

    queue = tmp_path / "queue.json"
    log = tmp_path / "annotator.log"
    result = run_auto(store, predictor, queue_path=queue, log_path=log)
    assert result.accepted == 1
    assert result.review == 3
    entries = json.loads(queue.read_text(encoding="utf-8"))
    assert len(entries) == 3
    assert {entry["image_id"] for entry in entries} == {ids[1], ids[2], ids[3]}
    assert all(entry["reason"] for entry in entries)
    db_review = [i for i in store.images() if i.status == "needs_review"]
    assert len(db_review) == result.review
    log_lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len([line for line in log_lines if line["event"] == "auto_review"]) == result.review


def test_accepted_boxes_are_proposals_not_labels(store: Store, tmp_path: Path) -> None:
    ids = seed_new(store, tmp_path, n=1)
    run_auto(
        store,
        lambda path: [Detection(1, 0.4, 0.6, 0.2, 0.2, 0.95)],
        queue_path=tmp_path / "queue.json",
        log_path=tmp_path / "annotator.log",
    )
    assert store.image(ids[0]).status == "auto_ok"
    boxes = store.boxes(ids[0])
    assert len(boxes) == 1
    assert boxes[0].state == "proposed"
    assert boxes[0].source == "model"
    assert boxes[0].confidence == pytest.approx(0.95)


def test_f4_interrupt_leaves_no_partial_image(store: Store, tmp_path: Path) -> None:
    ids = seed_new(store, tmp_path, n=2)

    def predictor(path: str) -> list[Detection]:
        if path.endswith("a_01.png"):
            raise KeyboardInterrupt
        return [Detection(1, 0.5, 0.5, 0.2, 0.2, 0.95)]

    with pytest.raises(KeyboardInterrupt):
        run_auto(
            store, predictor, queue_path=tmp_path / "queue.json", log_path=tmp_path / "log.txt"
        )
    assert store.image(ids[0]).status in {"auto_ok", "new"}
    assert store.boxes(ids[1]) == []


def test_custom_thresholds_used(store: Store, tmp_path: Path) -> None:
    seed_new(store, tmp_path, n=1)
    cfg = DecideConfig(conf_accept=0.99, conf_review=0.9)
    result = run_auto(
        store,
        lambda path: [Detection(1, 0.5, 0.5, 0.2, 0.2, 0.95)],
        cfg=cfg,
        queue_path=tmp_path / "queue.json",
        log_path=tmp_path / "log.txt",
    )
    assert result.review == 1
    assert store.boxes(store.images()[0].id)[0].state == "proposed"
