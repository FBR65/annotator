"""Kriterien E1 bis E7: Training."""

from __future__ import annotations

from pathlib import Path

import pytest

from annotator.db import Store
from annotator.exporter import export_dataset
from annotator.geometry import Box
from annotator.importer import sha256_file
from annotator.trainer import NotEnoughData, TrainingError, build_params, train
from tests.conftest import make_image


def seed_labeled(store: Store, tmp_path: Path, n: int) -> None:
    for i in range(n):
        path = make_image(tmp_path / "bilder" / f"b_{i:02d}.png", size=(64, 48))
        image_id = store.add_image(str(path), sha256_file(path), 64, 48)
        assert image_id is not None
        store.add_box(image_id, Box(1, 0.5, 0.5, 0.3, 0.3))
        store.set_status(image_id, "labeled", by="human")


def fake_runner(calls: list[dict[str, object]], metrics: dict[str, float] | None = None):
    def run(params: dict[str, object], project: Path) -> dict[str, float]:
        calls.append(dict(params))
        project.mkdir(parents=True, exist_ok=True)
        (project / "best.pt").write_bytes(b"weights")
        return metrics if metrics is not None else {"mAP50": 0.42}

    return run


def test_e1_too_few_labeled(store: Store, tmp_path: Path) -> None:
    seed_labeled(store, tmp_path, n=1)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)
    with pytest.raises(NotEnoughData) as exc:
        train(store, tmp_path / "models")
    assert exc.value.exit_code == 2
    assert "mindestens" in exc.value.message.lower()


def test_e2_params_forwarded(store: Store, tmp_path: Path) -> None:
    seed_labeled(store, tmp_path, n=4)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)
    calls: list[dict[str, object]] = []
    train(
        store,
        tmp_path / "models",
        dataset=tmp_path / "ds",
        init="scratch",
        epochs=3,
        imgsz=320,
        batch=2,
        device="cpu",
        runner=fake_runner(calls),
    )
    assert calls
    params = calls[0]
    assert params["epochs"] == 3
    assert params["imgsz"] == 320
    assert params["batch"] == 2
    assert params["device"] == "cpu"
    assert str(params["model"]).endswith("yolov8n.yaml")


def test_e3_result_artifacts(store: Store, tmp_path: Path) -> None:
    seed_labeled(store, tmp_path, n=4)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)
    result = train(
        store, tmp_path / "models", dataset=tmp_path / "ds", epochs=1, runner=fake_runner([])
    )
    assert Path(result.model_path).is_file()
    assert result.metrics["mAP50"] == pytest.approx(0.42)
    assert Path(result.log_path).is_file()
    assert result.run_id >= 1


def test_e5_init_choice_recorded(store: Store, tmp_path: Path) -> None:
    seed_labeled(store, tmp_path, n=4)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)
    calls: list[dict[str, object]] = []
    train(
        store,
        tmp_path / "models",
        init="pretrained",
        epochs=1,
        runner=fake_runner(calls),
        dataset=tmp_path / "ds",
    )
    assert str(calls[0]["model"]).endswith("yolov8n.pt")
    assert calls[0]["init"] == "pretrained"
    assert build_params("scratch", 1, 64, 1, "cpu")["model"] == "yolov8n.yaml"


def test_e6_model_load_error(store: Store, tmp_path: Path) -> None:
    seed_labeled(store, tmp_path, n=4)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)

    def broken(params: dict[str, object], project: Path) -> dict[str, float]:
        raise OSError("kein Netz: yolov8n.pt nicht ladbar")

    with pytest.raises(TrainingError) as exc:
        train(store, tmp_path / "models", init="pretrained", runner=broken, dataset=tmp_path / "ds")
    assert exc.value.exit_code == 1
    assert "Netz" in exc.value.message or "nicht ladbar" in exc.value.message


def test_e7_hint_printed(store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seed_labeled(store, tmp_path, n=4)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)
    train(store, tmp_path / "models", dataset=tmp_path / "ds", epochs=1, runner=fake_runner([]))
    out = capsys.readouterr().out
    assert "pretrained" in out
    assert "4" in out


def test_dry_run_does_not_train(store: Store, tmp_path: Path) -> None:
    seed_labeled(store, tmp_path, n=4)
    export_dataset(store, tmp_path / "ds", fmt="yolo", force=True)
    calls: list[dict[str, object]] = []
    result = train(
        store,
        tmp_path / "models",
        epochs=1,
        dry_run=True,
        dataset=tmp_path / "ds",
        runner=fake_runner(calls),
    )
    assert calls == []
    assert result.model_path == ""
    assert result.run_id == 0
