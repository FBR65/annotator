"""Setzt über die HTTP-API Demo-Boxen und übernimmt sie (menschliche Handlung).

Aufruf: .venv/bin/python tools/annotate_demo.py [--basis http://127.0.0.1:8765] [--anzahl 6]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

WURZEL = Path(__file__).resolve().parent.parent / "demo"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="http://127.0.0.1:8765")
    parser.add_argument("--anzahl", type=int, default=6)
    args = parser.parse_args()

    wahrheit = json.loads((WURZEL / "wahrheit.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.basis, timeout=20.0) as client:
        klassen = client.get("/api/classes").json()
        if not klassen:
            klasse = client.post("/api/classes", json={"name": "Kasten", "color": "#4f9dff"})
            klasse.raise_for_status()
            klassen = client.get("/api/classes").json()
        klassen_id = klassen[0]["id"]

        bilder = client.get("/api/images").json()
        zugeordnet = 0
        for bild in bilder:
            if bild["status"] == "labeled":
                continue
            werte = wahrheit.get(bild["name"])
            if werte is None or zugeordnet >= args.anzahl:
                continue
            cx, cy, w, h = werte
            antwort = client.post(
                f"/api/images/{bild['id']}/boxes",
                json={"class_id": klassen_id, "cx": cx, "cy": cy, "w": w, "h": h},
            )
            antwort.raise_for_status()
            uebernahme = client.post(f"/api/images/{bild['id']}/accept")
            uebernahme.raise_for_status()
            zugeordnet += 1
            print(f"annotiert: {bild['name']} (Klasse {klassen_id})")
        print(f"{zugeordnet} von {len(bilder)} Bildern durch Menschen annotiert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
