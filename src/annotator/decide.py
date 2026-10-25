"""Entscheidung „eindeutig / nicht eindeutig" und Start-Hinweis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .geometry import Box, iou


@dataclass(frozen=True)
class Detection:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float

    def box(self) -> Box:
        return Box(self.class_id, self.cx, self.cy, self.w, self.h)


@dataclass(frozen=True)
class DecideConfig:
    conf_accept: float = 0.50
    conf_review: float = 0.25
    max_boxes: int = 50
    iou_dup: float = 0.70


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str


ACCEPT = "accept"
REVIEW = "review"

REASON_TEXT = {
    "kein_objekt": "Kein Objekt gefunden",
    "unsicher": "Erkennung unsicher (conf unter der Prüfschwelle)",
    "doppelt": "Doppelte Erkennung, Klasse nicht eindeutig",
    "zu_viele": "Zu viele Objekte, Modell überfordert",
    "grenzfall": "Grenzfall (conf zwischen Prüf- und Freigabeschwelle)",
    "sicher": "Alle Erkennungen sicher",
}


def decide(detections: Sequence[Detection], cfg: DecideConfig) -> Decision:
    """Regeln in fester Reihenfolge; erste Übereinstimmung gewinnt."""
    if not detections:
        return Decision(REVIEW, "kein_objekt")
    if any(det.confidence < cfg.conf_review for det in detections):
        return Decision(REVIEW, "unsicher")
    for index, first in enumerate(detections):
        for second in detections[index + 1 :]:
            if first.class_id == second.class_id and iou(first.box(), second.box()) >= cfg.iou_dup:
                return Decision(REVIEW, "doppelt")
    if len(detections) > cfg.max_boxes:
        return Decision(REVIEW, "zu_viele")
    if any(det.confidence < cfg.conf_accept for det in detections):
        return Decision(REVIEW, "grenzfall")
    return Decision(ACCEPT, "sicher")


def init_hint(n_labeled: int, init: str) -> str:
    """Deutsche Empfehlung vor dem Trainingslauf. Ändert nichts."""
    if init not in {"scratch", "pretrained"}:
        raise ValueError(f"Unbekannter Init-Weg: {init}")
    if n_labeled < 200 and init == "scratch":
        return (
            f"Hinweis: {n_labeled} gelabelte Bilder. --init pretrained liefert meist bessere "
            "Ergebnisse, weil der Start auf COCO-Gewichten sitzt."
        )
    if n_labeled < 200:
        return (
            f"Hinweis: {n_labeled} gelabelte Bilder. pretrained ist passend bei kleinen "
            "Datenmengen."
        )
    if init == "scratch":
        return f"Hinweis: {n_labeled} gelabelte Bilder. scratch ist ausreichend Daten."
    return f"Hinweis: {n_labeled} gelabelte Bilder. --init scratch bietet schnellerer Start."
