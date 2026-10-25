"""Oberflächen-Vertrag aus SPEC Abschnitt 6: Import und Export über die API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from annotator.api import create_app
from annotator.db import Store
from annotator.geometry import Box
from annotator.importer import sha256_file
from tests.conftest import make_image


@pytest.fixture()
def client(store: Store, tmp_path: Path) -> TestClient:
    return TestClient(create_app(store, tmp_path / "log.txt"))


def test_import_endpoint_adds_images(client: TestClient, store: Store, tmp_path: Path) -> None:
    ordner = tmp_path / "eingang"
    make_image(ordner / "a.png")
    make_image(ordner / "unter" / "b.png")
    antwort = client.post("/api/import", json={"folder": str(ordner), "recursive": True})
    assert antwort.status_code == 200
    assert antwort.json() == {
        "added": 2,
        "duplicates": 0,
        "skipped": 0,
        "unreadable": 0,
        "labels": None,
    }
    assert len(store.images()) == 2


def test_import_endpoint_rejects_missing_folder(client: TestClient, tmp_path: Path) -> None:
    antwort = client.post("/api/import", json={"folder": str(tmp_path / "gibtsnicht")})
    assert antwort.status_code == 400
    assert "ordner" in antwort.json()["detail"].lower()


def test_export_endpoint_writes_dataset(client: TestClient, store: Store, tmp_path: Path) -> None:
    path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    store.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    store.set_status(image_id, "labeled", by="human")
    ziel = tmp_path / "datensatz"
    antwort = client.post(
        "/api/export",
        json={"out": str(ziel), "format": "both", "val_fraction": 0.0, "seed": 1},
    )
    assert antwort.status_code == 200
    koerper = antwort.json()
    assert koerper["train"] == 1
    assert (ziel / "data.yaml").is_file()
    assert (ziel / "train" / "_annotations.coco.json").is_file()


def test_export_endpoint_needs_force(client: TestClient, store: Store, tmp_path: Path) -> None:
    path = make_image(tmp_path / "bild.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    store.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    store.set_status(image_id, "labeled", by="human")
    ziel = tmp_path / "datensatz"
    ziel.mkdir()
    ohne = client.post("/api/export", json={"out": str(ziel), "format": "coco"})
    assert ohne.status_code == 400
    assert "force" in ohne.json()["detail"].lower()
    mit = client.post("/api/export", json={"out": str(ziel), "format": "coco", "force": True})
    assert mit.status_code == 200
