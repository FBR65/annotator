"""Kriterium E4: reale Ausführung mit ultralytics auf der GPU.

Nicht im Standardlauf (zu langsam):  uv run pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest

from annotator.db import Store
from annotator.exporter import export_dataset
from annotator.geometry import Box
from annotator.importer import sha256_file
from annotator.trainer import train
from tests.conftest import make_image

pytestmark = pytest.mark.slow


def test_e4_real_training_one_epoch(store: Store, tmp_path: Path) -> None:
    for i in range(8):
        path = make_image(tmp_path / "bilder" / f"s_{i:02d}.png", size=(160, 160))
        image_id = store.add_image(str(path), sha256_file(path), 160, 160)
        assert image_id is not None
        store.add_box(image_id, Box(1, 0.5, 0.5, 0.4, 0.4))
        store.set_status(image_id, "labeled", by="human")
    dataset = tmp_path / "ds"
    export_dataset(store, dataset, fmt="yolo", val_fraction=0.25, seed=1, force=True)
    result = train(
        store,
        tmp_path / "models",
        dataset=dataset,
        init="scratch",
        epochs=1,
        imgsz=320,
        batch=2,
        device="cuda",
    )
    assert Path(result.model_path).is_file()
    assert Path(result.log_path).is_file()
    assert result.metrics
