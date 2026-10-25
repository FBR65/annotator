"""Erzeugt einen kleinen Demo-Datensatz: helle Rechtecke auf dunklem Grund.

Schreibt zusätzlich demo/wahrheit.json mit der echten Box je Bild.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

WURZEL = Path(__file__).resolve().parent.parent / "demo"
AUSGABE = WURZEL / "bilder"
BREITE, HOEHE = 320, 240


def main() -> None:
    AUSGABE.mkdir(parents=True, exist_ok=True)
    zufall = random.Random(7)
    wahrheit: dict[str, list[float]] = {}
    for nummer in range(12):
        bild = Image.new("RGB", (BREITE, HOEHE), (12, 14, 18))
        zeichner = ImageDraw.Draw(bild)
        for _ in range(40):
            x = zufall.randrange(0, BREITE)
            y = zufall.randrange(0, HOEHE)
            zeichner.point((x, y), fill=(60, 62, 70))
        kasten_breite = zufall.randrange(40, 90)
        kasten_hoehe = zufall.randrange(30, 70)
        links = zufall.randrange(10, BREITE - kasten_breite - 10)
        oben = zufall.randrange(10, HOEHE - kasten_hoehe - 10)
        zeichner.rectangle(
            [links, oben, links + kasten_breite, oben + kasten_hoehe],
            outline=(235, 235, 235),
            width=3,
            fill=(120, 140, 160),
        )
        name = f"probe_{nummer:02d}.png"
        bild.save(AUSGABE / name)
        cx = (links + kasten_breite / 2) / BREITE
        cy = (oben + kasten_hoehe / 2) / HOEHE
        wahrheit[name] = [cx, cy, kasten_breite / BREITE, kasten_hoehe / HOEHE]
    (WURZEL / "wahrheit.json").write_text(
        json.dumps(wahrheit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(wahrheit)} Bilder in {AUSGABE}")


if __name__ == "__main__":
    main()
