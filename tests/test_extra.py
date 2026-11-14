"""Zusätzliche Randfälle: Fehlerpfade in Geometrie, API, Kommandozeile, Trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from annotator.api import create_app
from annotator.cli import main
from annotator.db import Store
from annotator.geometry import Box, BoxError, clamp_box, from_yolo_line, iou, validate_box
from annotator.importer import sha256_file
from annotator.trainer import TrainingError, build_params, train
from tests.conftest import make_image


def test_yolo_line_with_text_rejected() -> None:
    with pytest.raises(BoxError):
        from_yolo_line("eins 0.5 0.5 0.1 0.1")


def test_yolo_line_with_nan_rejected_by_validate() -> None:
    box = from_yolo_line("0 nan 0.5 0.1 0.1")
    with pytest.raises(BoxError):
        validate_box(box, "x.png")


def test_clamp_rejects_empty_image() -> None:
    with pytest.raises(BoxError):
        clamp_box(Box(1, 0.5, 0.5, 0.1, 0.1), width=0, height=10)


def test_validate_rejects_center_outside() -> None:
    with pytest.raises(BoxError):
        validate_box(Box(1, 1.2, 0.5, 0.1, 0.1), "x.png")


def test_validate_rejects_too_wide_box() -> None:
    with pytest.raises(BoxError):
        validate_box(Box(1, 0.5, 0.5, 1.5, 0.1), "x.png")


def test_iou_of_degenerate_boxes_is_zero() -> None:
    assert iou(Box(1, 0.5, 0.5, 0.0, 0.0), Box(1, 0.5, 0.5, 0.0, 0.0)) == 0.0


def test_build_params_rejects_unknown_init() -> None:
    with pytest.raises(TrainingError):
        build_params("zufall", 1, 64, 1, "cpu")


def test_train_rejects_unknown_init(store: Store, tmp_path: Path) -> None:
    with pytest.raises(TrainingError):
        train(store, tmp_path / "models", init="zufall")


def test_api_serves_image_file(store: Store, tmp_path: Path) -> None:
    path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    client = TestClient(create_app(store, tmp_path / "log.txt"))
    antwort = client.get(f"/api/images/{image_id}/file")
    assert antwort.status_code == 200
    assert antwort.headers["content-type"].startswith("image/png")


def test_api_creates_class_and_rejects_empty(store: Store, tmp_path: Path) -> None:
    client = TestClient(create_app(store, tmp_path / "log.txt"))
    angelegt = client.post("/api/classes", json={"name": "Schere", "color": "#00ff00"})
    assert angelegt.status_code == 201
    assert angelegt.json()["id"] == 3
    leer = client.post("/api/classes", json={"name": "   "})
    assert leer.status_code == 400


def test_api_missing_box_returns_404(store: Store, tmp_path: Path) -> None:
    client = TestClient(create_app(store, tmp_path / "log.txt"))
    assert client.delete("/api/boxes/999").status_code == 404
    assert client.get("/api/images/999").status_code == 404


def test_api_patch_without_fields_returns_400(store: Store, tmp_path: Path) -> None:
    path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    box_id = store.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    client = TestClient(create_app(store, tmp_path / "log.txt"))
    assert client.patch(f"/api/boxes/{box_id}", json={}).status_code == 400


def test_cli_train_dry_run(
    store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for nummer in range(2):
        path = make_image(tmp_path / "bilder" / f"b_{nummer}.png")
        image_id = store.add_image(str(path), sha256_file(path), 64, 48)
        assert image_id is not None
        store.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
        store.set_status(image_id, "labeled", by="human")
    db = store.path
    store.close()
    code = main(["--db", str(db), "train", "--cwd", str(tmp_path), "--dry-run", "--epochs", "1"])
    assert code == 0
    ausgabe = capsys.readouterr().out
    assert "Trockenlauf" in ausgabe
    assert "pretrained" in ausgabe


def test_cli_import_with_labels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    s = Store(db)
    s.add_class("Kasten")
    s.close()
    ordner = tmp_path / "bilder"
    make_image(ordner / "a.png")
    (ordner / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    code = main(["--db", str(db), "import", str(ordner), "--labels"])
    assert code == 0
    ausgabe = capsys.readouterr().out
    assert "Labels: 1 Boxen" in ausgabe
    s = Store(db)
    assert s.counts()["labeled"] == 1
    s.close()


def test_cli_export_without_images(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    s = Store(db)
    s.add_class("Kasten")
    s.close()
    code = main(["--db", str(db), "export", "--out", str(tmp_path / "ds")])
    assert code == 1
    assert "Fehler" in capsys.readouterr().out
