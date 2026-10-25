"""Kriterien B1 bis B4: HTTP-API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from annotator.api import create_app
from annotator.db import Store


@pytest.fixture()
def client(store: Store, tmp_path: Path) -> tuple[TestClient, Store, Path]:
    log = tmp_path / "annotator.log"
    app = create_app(store, log)
    image_id = store.add_image(str(tmp_path / "bild.png"), "s1", 200, 100)
    assert image_id is not None
    return TestClient(app), store, log


def _image_id(store: Store) -> int:
    return store.images()[0].id


def test_b1_create_and_read_back(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    payload = {"class_id": 1, "cx": 0.4, "cy": 0.6, "w": 0.2, "h": 0.1}
    created = api.post(f"/api/images/{image_id}/boxes", json=payload)
    assert created.status_code == 201
    box_id = created.json()["id"]
    assert box_id == 1
    listed = api.get(f"/api/images/{image_id}").json()["boxes"]
    assert len(listed) == 1
    assert listed[0]["cx"] == pytest.approx(0.4)
    assert listed[0]["cy"] == pytest.approx(0.6)
    assert listed[0]["w"] == pytest.approx(0.2)
    assert listed[0]["h"] == pytest.approx(0.1)


def test_b2_patch_keeps_other_boxes(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    first = api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 1, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1},
    ).json()["id"]
    api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 2, "cx": 0.8, "cy": 0.8, "w": 0.1, "h": 0.1},
    )
    before = {b["id"]: b for b in api.get(f"/api/images/{image_id}").json()["boxes"]}
    patched = api.patch(f"/api/boxes/{first}", json={"cx": 0.31})
    assert patched.status_code == 200
    after = {b["id"]: b for b in api.get(f"/api/images/{image_id}").json()["boxes"]}
    others = [key for key in before if key != first]
    for key in others:
        assert before[key] == after[key]


def test_b3_delete_one_box(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    first = api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 1, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1},
    ).json()["id"]
    api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 1, "cx": 0.7, "cy": 0.7, "w": 0.1, "h": 0.1},
    )
    assert api.delete(f"/api/boxes/{first}").status_code == 200
    assert len(api.get(f"/api/images/{image_id}").json()["boxes"]) == 1


def test_b4_one_log_line_per_mutation(client: tuple[TestClient, Store, Path]) -> None:
    api, store, log = client
    image_id = _image_id(store)
    before = store.image(image_id).updated_at
    created = api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
    )
    box_id = created.json()["id"]
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["event"] == "box_create"
    assert lines[0]["image_id"] == image_id
    assert store.image(image_id).updated_at != before
    api.patch(f"/api/boxes/{box_id}", json={"cx": 0.6})
    api.delete(f"/api/boxes/{box_id}")
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [entry["event"] for entry in lines] == ["box_create", "box_update", "box_delete"]


def test_invalid_box_returns_400(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    response = api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1},
    )
    assert response.status_code == 400
    assert "ungültig" in response.json()["detail"].lower()


def test_unknown_class_returns_400(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    response = api.post(
        f"/api/images/{image_id}/boxes",
        json={"class_id": 99, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
    )
    assert response.status_code == 400


def test_accept_proposal_marks_labeled(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    store.add_box(
        image_id,
        __import__("annotator.geometry", fromlist=["Box"]).Box(1, 0.5, 0.5, 0.2, 0.2),
        source="model",
        confidence=0.9,
        state="proposed",
    )
    store.set_status(image_id, "auto_ok", by="auto")
    response = api.post(f"/api/images/{image_id}/accept")
    assert response.status_code == 200
    assert store.image(image_id).status == "labeled"
    assert all(b.state == "accepted" for b in store.boxes(image_id))


def test_summary_lists_counts_and_queue(client: tuple[TestClient, Store, Path]) -> None:
    api, store, _ = client
    image_id = _image_id(store)
    store.set_status(image_id, "needs_review", by="auto")
    body = api.get("/api/summary").json()
    assert body["counts"]["needs_review"] == 1
    assert body["review_queue"][0]["id"] == image_id
