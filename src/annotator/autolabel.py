"""Automatische Vorannotation: Vorschläge erzeugen und unsichere Bilder melden."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .db import Store
from .decide import ACCEPT, REASON_TEXT, DecideConfig, Detection, decide
from .geometry import BoxError, clamp_box


@dataclass(frozen=True)
class AutoResult:
    processed: int
    accepted: int
    review: int
    reasons: dict[str, int]


Predictor = Callable[[str], list[Detection]]


def _append_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_auto(
    store: Store,
    predictor: Predictor,
    cfg: DecideConfig | None = None,
    queue_path: Path | None = None,
    log_path: Path | None = None,
) -> AutoResult:
    settings = cfg or DecideConfig()
    queue_file = Path(queue_path) if queue_path is not None else Path("review_queue.json")
    log_file = Path(log_path) if log_path is not None else Path("annotator.log")
    reasons: Counter[str] = Counter()
    queue: list[dict[str, object]] = []
    processed = accepted = review = 0

    for image in store.images("new"):
        processed += 1
        detections = predictor(image.path)
        decision = decide(detections, settings)
        with store.transaction():
            for detection in detections:
                try:
                    box = clamp_box(detection.box(), image.width, image.height)
                except BoxError:
                    continue
                store.add_box(
                    image.id,
                    box,
                    source="model",
                    confidence=detection.confidence,
                    state="proposed",
                )
            if decision.verdict == ACCEPT:
                store.set_status(image.id, "auto_ok", by="auto")
            else:
                store.set_status(image.id, "needs_review", by="auto")
        if decision.verdict == ACCEPT:
            accepted += 1
            _append_log(
                log_file,
                {
                    "event": "auto_accept",
                    "image_id": image.id,
                    "path": image.path,
                    "reason": decision.reason,
                    "boxes": len(detections),
                },
            )
        else:
            review += 1
            reasons[decision.reason] += 1
            queue.append(
                {
                    "image_id": image.id,
                    "path": image.path,
                    "reason": decision.reason,
                    "text": REASON_TEXT.get(decision.reason, decision.reason),
                    "boxes": len(detections),
                }
            )
            _append_log(
                log_file,
                {
                    "event": "auto_review",
                    "image_id": image.id,
                    "path": image.path,
                    "reason": decision.reason,
                    "boxes": len(detections),
                },
            )

    queue_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = queue_file.with_suffix(queue_file.suffix + ".tmp")
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(queue_file)
    return AutoResult(
        processed=processed,
        accepted=accepted,
        review=review,
        reasons=dict(reasons),
    )
