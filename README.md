# Annotator

Lokales Werkzeug für Bildannotation, YOLO-Training und automatische Vorannotation.

Ablauf: Der Mensch annotiert wenige Bilder. Das Werkzeug trainiert daraus ein
YOLO-Modell und annotiert damit die übrigen Bilder vor. Ist eine Zuordnung nicht
eindeutig, landet das Bild in einer Warteschlange und ein Mensch entscheidet.

Originalbilder bleiben am Ort. Die Datenbank speichert nur Pfad und SHA-256.
Der Server bindet ausschließlich an `127.0.0.1`.

## Voraussetzungen

| Baustein | Anforderung |
|---|---|
| Python | 3.12 (uv verwaltet die Version) |
| uv | 0.12 oder neuer |
| GPU | optional; NVIDIA mit CUDA für Training, sonst `--device cpu` |
| Netz | nur beim ersten `--init pretrained` (lädt `yolov8n.pt`) |
| Speicher | SQLite-Datei, Bilder bleiben im Quellordner |

## Einrichtung

```
uv sync                      # Umgebung und Abhängigkeiten
uv run annotator --help      # Kurzhilfe
```

## Befehle

Alle Befehle beginnen mit `--db <pfad>` (Vorgabe: `annotator.db`.
Einmal angelegt, gilt dieselbe Datei für alle Schritte.

### `import` — Bilder einlesen

```
annotator --db annotator.db import ~/bilder --recursive
annotator --db annotator.db import ~/bilder --labels
```

| Option | Wirkung |
|---|---|
| `--recursive` | Unterordner mitlesen |
| `--labels` | vorhandene `.txt`-Labels (YOLO) neben den Bildern übernehmen; Klasse außerhalb des gültigen Bereichs wird abgelehnt, Datei bleibt unberührt |

Gelesen werden `.jpg .jpeg .png .bmp .webp .tif .tiff` (Groß- und Kleinschreibung
gleich). Symlinks werden nicht verfolgt. Doppelte Bilder erkennt das Werkzeug am
SHA-256 und überspringt sie.

### `serve` — Oberfläche im Browser

```
annotator --db annotator.db serve --port 8765
```

Danach `http://127.0.0.1:8765/` öffnen. `--host` akzeptiert nur `127.0.0.1`,
`localhost` oder `::1`; jede andere Adresse wird mit Fehler und Exit-Code 1
abgelehnt.

| Bedienung | Wirkung |
|---|---|
| „Bilder importieren“ + Ordnerpfad | liest den Ordner ein; Häkchen für Unterordner und vorhandene Labels |
| „Datensatz schreiben“ + Zielpfad | Export nach YOLO, COCO oder beides; Häkchen „überschreiben“ |
| Ziehen auf dem Bild | neue Box aufziehen |
| Klick auf Box | Box wählen |
| Ziehen in der Box | verschieben |
| Ecke ziehen | Größe ändern |
| `1`–`9` | Klasse der gewählten Box setzen |
| `Entf` | Box löschen |
| `Strg+Z` oder Knopf „Rückgängig“ | letzte Änderung dieser Sitzung zurücknehmen |
| `A` / `D` oder Pfeiltasten | voriges / nächstes Bild |
| `Leertaste` | Vorschlag übernehmen (Status `labeled`) |
| `R` | Liste neu laden |
| Auswahl „Ansicht“ | Filter nach Status, u. a. „Nur Prüfung“ |

Jede Änderung wird sofort gespeichert und als Zeile in `annotator.log`
protokolliert. Rückgängig reicht höchstens 100 Schritte zurück und endet mit
dem Schließen der Seite.

### `review` — Warteschlange auf der Konsole

```
annotator --db annotator.db review
```

Zeigt Anzahl und Pfad aller Bilder im Status `needs_review` sowie den Bestand
nach Status.

### `train` — YOLO trainieren

```
annotator --db annotator.db train --dataset ./datensatz --epochs 100 --init scratch
```

| Option | Vorgabe | Wirkung |
|---|---|---|
| `--cwd` | `.` | Arbeitsordner; Modelle landen in `<cwd>/models/run_<id>/` |
| `--dataset` | `<cwd>/dataset` | Ordner mit `data.yaml` aus `export` |
| `--epochs` | 100 | Epochen |
| `--init` | `scratch` | `scratch` = `yolov8n.yaml`, keine fremden Gewichte, kein Netz. `pretrained` = `yolov8n.pt` (COCO), lädt beim ersten Mal |
| `--imgsz` | 640 | Bildgröße fürs Training |
| `--batch` | 4 | Batch-Größe (8 GB VRAM: 4 ist sicher) |
| `--device` | `cuda` | `cuda` oder `cpu` |
| `--dry-run` | aus | zeigt Parameter und Hinweis, trainiert nicht |

Vor jedem Lauf erscheint ein Hinweis, welcher Init-Weg zur Datenmenge passt
(unter 200 gelabelten Bildern: `pretrained` liefert meist bessere Ergebnisse).
Das Ergebnis liegt unter `models/run_<id>/best.pt`, die Kennzahlen in
`models/run_<id>/train.log` und in der Datenbank.

### `auto` — automatisch vorannotieren

```
annotator --db annotator.db auto --model models/run_4/best.pt --cwd ./arbeit
```

| Option | Vorgabe | Wirkung |
|---|---|---|
| `--model` | jüngstes `models/run_*/best.pt` | Gewichtsdatei |
| `--cwd` | `.` | Ziel für `review_queue.json` und `annotator.log` |
| `--conf-accept` | 0.50 | ab hier gilt eine Erkennung als sicher |
| `--conf-review` | 0.25 | darunter gilt eine Erkennung als unsicher |
| `--device` | `cuda` | `cuda` oder `cpu` |

Verarbeitet werden nur Bilder im Status `new`. Fertig annotierte Bilder werden
nie überschrieben.

**Wann ein Bild zur Prüfung geht** (erste zutreffende Regel gewinnt):

| Regel | Bedingung | Grund in der Meldung |
|---|---|---|
| 1 | keine Erkennung | „Kein Objekt gefunden“ |
| 2 | eine Box unter `--conf-review` | „Erkennung unsicher“ |
| 3 | zwei Boxen gleicher Klasse, IoU ≥ 0.70 | „Doppelte Erkennung“ |
| 4 | mehr als 50 Boxen | „Zu viele Objekte“ |
| 5 | eine Box unter `--conf-accept` | „Grenzfall“ |
| 6 | sonst | sicher, Status `auto_ok` |

**Meldung an den Menschen** — vier Wege, immer alle:

1. Konsole: „Fertig: N Bilder geprüft, A sicher, R zur Prüfung“ plus Gründe.
2. `annotator.log` im Arbeitsordner: eine JSON-Zeile je Bild mit Grund.
3. `review_queue.json` im Arbeitsordner: Liste mit Pfad und Grund.
4. Oberfläche: Banner mit Anzahl, Filter „Nur Prüfung“.

### `export` — Datensatz schreiben

```
annotator --db annotator.db export --out ./datensatz --format both --val-fraction 0.2
```

| Option | Vorgabe | Wirkung |
|---|---|---|
| `--out` | erforderlich | Zielordner |
| `--format` | `coco` | `coco` (RF-DETR), `yolo`, `both` |
| `--val-fraction` | 0.2 | Anteil Prüfbilder |
| `--seed` | 42 | gleicher Wert → gleiche Aufteilung |
| `--force` | aus | vorhandenen Zielordner überschreiben |

Exportiert werden nur Bilder mit Status `labeled`. Anzahl der übergangenen
Bilder steht in `<out>/export_meta.json`.

| Format | Ergebnis |
|---|---|
| `yolo` | `images/{train,val}`, `labels/{train,val}`, `data.yaml` (absolute Pfade) |
| `coco` | `train/_annotations.coco.json`, `val/_annotations.coco.json`, Bilder daneben |
| `both` | beides im selben Ordner |

## Status der Bilder

```
new ──auto, alle Boxen sicher──▶ auto_ok ──Mensch bestätigt──▶ labeled
new ──auto, unsicher/leer─────▶ needs_review ──Mensch────────▶ labeled
new ──Mensch annotiert────────▶ labeled
```

`labeled` entsteht nur durch eine menschliche Handlung. `auto_ok` ist ein
Vorschlag mit hoher Sicherheit, kein fertiges Label.

## Exit-Codes

| Code | Bedeutung |
|---|---|
| 0 | Erfolg |
| 1 | Fehler (Meldung auf der Konsole) |
| 2 | nichts zu tun (kein Modell, zu wenige Bilder) |
| 3 | teilweise fehlgeschlagen: `import` mit unlesbaren Bilddateien, `auto` mit Bildern, die das Modell nicht verarbeiten konnte |

Bei Code 3 steht die Zahl der betroffenen Dateien in der Meldung. Diese Bilder
gehen nicht verloren: unlesbare Dateien bleiben unangetastet, fehlerhafte
Vorannotierungen landen in der Warteschlange.

## Dateien

| Datei | Inhalt |
|---|---|
| `annotator.db` | Datenbank: Bilder, Klassen, Boxen, Läufe |
| `annotator.log` | JSON-Zeilen zu Änderungen und Vorannotation |
| `review_queue.json` | aktuelle Warteschlange |
| `models/run_<id>/` | Gewichte und Trainingsprotokoll je Lauf |
| `datensatz/` | Export |

## Grenzen

- Ein Bediener, ein Rechner. Keine Benutzerverwaltung, keine Authentifizierung.
- Nur Bounding Boxes auf Einzelbildern. Keine Videos, keine Masken.
- Von Null trainiert braucht ein brauchbares Modell deutlich mehr als 100 Bilder.
  Bei kleinen Mengen `--init pretrained` verwenden.
- `ultralytics` steht unter **AGPL-3.0**. Weitergabe des Werkzeugs erbt diese Lizenz.

## Vertrag und Nachweise

- `SPEC.md` — freigegebener Vertrag mit den Abnahmekriterien A1–G4.
- `EVIDENCE.md` — Prüfbericht: 104 Tests, 1 GPU-Test, 7/7 Mutanten getötet,
  89 % Abdeckung, realer Durchlauf.
