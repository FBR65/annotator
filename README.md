# Annotator

Lokales Werkzeug für Bildannotation im YOLO-Format, YOLO-Training und automatische
Vorannotation. Bilder bleiben am Ort. Der Mensch entscheidet am Ende.

## Schnellstart

```
uv sync
uv run annotator --db annotator.db import ~/bilder --recursive
uv run annotator --db annotator.db serve
uv run annotator --db annotator.db train --epochs 100
uv run annotator --db annotator.db auto
uv run annotator --db annotator.db export --out ./datensatz --format both
```

Vertrag: `SPEC.md`.
