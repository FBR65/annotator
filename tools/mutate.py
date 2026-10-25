"""Handmutation: baut je einen echten Fehler ein und prüft, ob die Suite ihn tötet.

Aufruf:  .venv/bin/python tools/mutate.py
Ablauf:  Datei sichern -> Fehler einbauen -> pytest -> zurückschreiben.
Exit 0 nur, wenn jeder Mutant getötet wird.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "annotator"


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    old: str
    new: str
    expectation: str


MUTANTS = (
    Mutant(
        name="M1 Breite 0 erlaubt",
        file="geometry.py",
        old="    if box.w <= 0 or box.h <= 0:",
        new="    if box.w < 0 or box.h < 0:",
        expectation="A3 muss sterben (w=0 zugelassen)",
    ),
    Mutant(
        name="M2 Mindestgröße aufgehoben",
        file="geometry.py",
        old="    x0, x1 = _min_extent(",
        new="    x0, x1 = _min_extent_unused(",
        expectation="A2 muss sterben (Box kann 0 Pixel breit werden)",
    ),
    Mutant(
        name="M3 Doppelt-Schwelle verschoben",
        file="decide.py",
        old="iou(first.box(), second.box()) >= cfg.iou_dup",
        new="iou(first.box(), second.box()) >= cfg.iou_dup * 2",
        expectation="C3 muss sterben (0.8 gilt nicht mehr als doppelt)",
    ),
    Mutant(
        name="M4 Grenzfall ignoriert",
        file="decide.py",
        old="    if any(det.confidence < cfg.conf_accept for det in detections):",
        new="    if False:",
        expectation="C2 muss sterben (0.49 wird freigegeben)",
    ),
    Mutant(
        name="M5 Aufteilung zu groß",
        file="exporter.py",
        old="    n_val = round(len(shuffled) * val_fraction)",
        new="    n_val = round(len(shuffled) * val_fraction) + 1",
        expectation="D1 muss sterben (16/4 wird 15/5)",
    ),
    Mutant(
        name="M6 Label ohne Menschen",
        file="db.py",
        old='        if status == "labeled" and by != "human":',
        new="        if False:",
        expectation="G4 muss sterben (auto darf labeled setzen)",
    ),
    Mutant(
        name="M7 Unsicher wird sicher",
        file="autolabel.py",
        old="            if decision.verdict == ACCEPT:",
        new="            if decision.verdict != ACCEPT:",
        expectation="C7 und F2 müssen sterben (Zählung vertauscht)",
    ),
)


def run_suite() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:randomly", "-m", "not slow"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode


def main() -> int:
    survivors: list[str] = []
    print(f"{'Mutant':38} {'Ergebnis':10} Erwartung")
    print("-" * 100)
    for mutant in MUTANTS:
        target = SRC / mutant.file
        original = target.read_text(encoding="utf-8")
        if mutant.old not in original:
            print(f"{mutant.name:38} {'FEHLER':10} Suchtext nicht gefunden")
            survivors.append(mutant.name)
            continue
        try:
            target.write_text(original.replace(mutant.old, mutant.new, 1), encoding="utf-8")
            code = run_suite()
        finally:
            target.write_text(original, encoding="utf-8")
        killed = code != 0
        print(f"{mutant.name:38} {'getötet' if killed else 'ÜBERLEBT':10} {mutant.expectation}")
        if not killed:
            survivors.append(mutant.name)
    print("-" * 100)
    print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} Mutanten getötet")
    if survivors:
        print("Überlebende: " + ", ".join(survivors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
