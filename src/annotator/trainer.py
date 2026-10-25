"""YOLO-Training über ultralytics."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .db import Store
from .decide import init_hint

MODELS = {"scratch": "yolov8n.yaml", "pretrained": "yolov8n.pt"}
MIN_LABELED = 2


class TrainingError(RuntimeError):
    """Training nicht möglich. exit_code für die Kommandozeile."""

    exit_code = 1

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if exit_code is not None:
            self.exit_code = exit_code


class NotEnoughData(TrainingError):
    exit_code = 2


@dataclass(frozen=True)
class TrainResult:
    run_id: int
    model_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    log_path: str = ""


Runner = Callable[[dict[str, object], Path], dict[str, float]]


def build_params(init: str, epochs: int, imgsz: int, batch: int, device: str) -> dict[str, object]:
    if init not in MODELS:
        raise TrainingError(f"Unbekannter Init-Weg: {init}. Erlaubt: scratch, pretrained")
    return {
        "model": MODELS[init],
        "init": init,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
    }


def _as_int(value: object) -> int:
    return int(str(value))


def _ultralytics_runner(params: dict[str, object], project: Path) -> dict[str, float]:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(params["model"]))
    model.train(
        data=str(params["data"]),
        epochs=_as_int(params["epochs"]),
        imgsz=_as_int(params["imgsz"]),
        batch=_as_int(params["batch"]),
        device=str(params["device"]),
        project=str(project.resolve()),
        name="yolo",
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", project / "yolo"))
    weights = save_dir / "weights" / "best.pt"
    if weights.is_file():
        shutil.copy2(weights, project / "best.pt")
    raw = getattr(getattr(model, "trainer", None), "metrics", None) or {}
    metrics: dict[str, float] = {}
    for key, value in dict(raw).items():
        short = str(key).split("/")[-1].replace("(B)", "")
        try:
            metrics[short] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def train(
    store: Store,
    out_root: Path,
    dataset: Path | None = None,
    init: str = "scratch",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 4,
    device: str = "cuda",
    dry_run: bool = False,
    runner: Runner | None = None,
    printer: Callable[[str], None] = print,
) -> TrainResult:
    params = build_params(init, epochs, imgsz, batch, device)
    n_labeled = len(store.images("labeled"))
    printer(init_hint(n_labeled, init))
    if n_labeled < MIN_LABELED:
        raise NotEnoughData(
            f"Zu wenige gelabelte Bilder: {n_labeled}. Mindestens {MIN_LABELED} Bilder nötig."
        )
    dataset_path = Path(dataset) if dataset is not None else Path("dataset")
    params["data"] = str((dataset_path / "data.yaml").resolve())
    params["dataset"] = str(dataset_path.resolve())
    if dry_run:
        printer(f"Trockenlauf: {json.dumps(params, ensure_ascii=False, sort_keys=True)}")
        return TrainResult(run_id=0, model_path="", metrics={}, log_path="")

    if not (dataset_path / "data.yaml").is_file():
        raise TrainingError(
            f"Datensatz fehlt: {dataset_path / 'data.yaml'}. "
            "Bitte zuerst 'annotator export' ausführen."
        )

    out_root.mkdir(parents=True, exist_ok=True)
    run_id = store.add_run("train", params)
    project = out_root / f"run_{run_id}"
    project.mkdir(parents=True, exist_ok=True)
    log_path = project / "train.log"
    log_path.write_text(
        "Parameter: " + json.dumps(params, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        metrics = (runner or _ultralytics_runner)(params, project)
    except TrainingError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrainingError(f"Training abgebrochen: {exc}") from exc

    best = project / "best.pt"
    if not best.is_file():
        raise TrainingError("Training lieferte keine Gewichtsdatei best.pt.")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("Metriken: " + json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
    store.finish_run(run_id, metrics, str(best), str(log_path))
    printer(f"Training fertig: {best} (Lauf {run_id})")
    return TrainResult(run_id=run_id, model_path=str(best), metrics=metrics, log_path=str(log_path))
