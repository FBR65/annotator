# SPEC — Annotator mit YOLO-Training und automatischer Vorannotation

Status: **Entwurf, wartet auf Freigabe.** Kein Code vor Freigabe.
Projekt: `/home/speedy/Dokumente/dev/annotator`
Verfahren: Old-Coder (SPEC → RED → GREEN → REFACTOR → GAUNTLET → EVIDENCE)
Stufe: **Tier 3 (Datenverlust möglich: menschliche Annotationen)**

---

## 1. Ziel

Ein lokales Werkzeug mit drei Fähigkeiten:

1. **Mensch annotiert** Bilder im Browser. Das Werkzeug schreibt YOLO-Labels,
   auf Wunsch zusätzlich COCO-JSON.
2. **Werkzeug trainiert** ein YOLO-Modell mit den vorhandenen Labels.
3. **Werkzeug annotiert automatisch** neue Bilder mit diesem Modell.

Wenn die Zuordnung nicht eindeutig ist, dann meldet das Werkzeug das Bild.
Ein Mensch sieht das Bild in einer Warteschlange und korrigiert es.

Grundsatz: **Mensch entscheidet am Ende.** Das Modell schlägt vor.

## 2. Abgrenzung (nicht im Umfang)

| Nicht im Umfang | Grund |
|---|---|
| Training von RF-DETR | Eigener Auftrag. Dieses Werkzeug liefert die Daten (YOLO und COCO). |
| Cloud-Dienste, Roboflow, CVAT, Label Studio | Local-first. Bilder verlassen den Rechner nicht. |
| Benutzerverwaltung, Mehrbenutzerbetrieb | Ein Rechner, ein Bediener. |
| Videos, Tracking, Polygon-Masken | Nur Bounding Boxes auf Einzelbildern. |
| Automatisches Verschieben oder Löschen von Originalbildern | Unveränderliche Quelle. |

## 3. Architektur

| Baustein | Wahl | Begründung |
|---|---|---|
| Sprache | Python 3.12 (uv) | torch/ultralytics stabil, 3.14 noch nicht abgedeckt |
| Server | FastAPI + Uvicorn, **nur 127.0.0.1** | UI im Browser, gut automatisch testbar |
| UI | HTML/CSS/JS, Canvas 2D, keine Build-Kette | kein npm, kein Node, direkt auslieferbar |
| Datenbank | SQLite (stdlib `sqlite3`), WAL, eine Datei `annotator.db` | transaktional, kein Server |
| Bilder | **bleiben am Ort**, DB speichert Pfad + SHA-256 | schützt Originale, kein Kopieren |
| Training/Inferenz | `ultralytics` (YOLO) | Standard für YOLO, CLI und Python-API |
| Labels | YOLO-TXT, normalisiert `class cx cy w h`; optional COCO-JSON | gefordertes Format; COCO für RF-DETR (D3) |
| Startgewichte | `--init scratch` (Default) oder `--init pretrained` | beide Wege auf Wunsch (D2) |

Prozessgrenze: Training und Vorannotation laufen als **getrennte CLI-Prozesse**.
Der Webserver startet kein Training. Grund: GPU-Speicher und Abstürze bleiben
aus dem Serverprozess heraus.

## 4. Datenmodell (SQLite)

| Tabelle | Felder (wesentlich) |
|---|---|
| `images` | `id`, `path` UNIQUE, `sha256`, `width`, `height`, `status`, `updated_at` |
| `classes` | `id` (stabil, aufsteigend), `name` UNIQUE, `color` |
| `boxes` | `id`, `image_id` → images, `class_id` → classes, `cx`,`cy`,`w`,`h` (float 0..1), `source` ∈ {`human`,`model`}, `confidence`, `state` ∈ {`accepted`,`proposed`,`rejected`} |
| `runs` | `id`, `kind` ∈ {`train`,`auto`}, `started_at`, `finished_at`, `params_json`, `metrics_json`, `model_path`, `log_path` |

`images.status` ∈ {`new`, `auto_ok`, `needs_review`, `labeled`}.
Übergänge:

```
new --auto(alle Boxen sicher)--> auto_ok --Mensch bestätigt--> labeled
new --auto(unsicher)----------> needs_review --Mensch--> labeled
new --auto(keine Box)---------> needs_review
beliebig --Mensch bearbeitet--> labeled
```

Regel: `status='labeled'` nur nach menschlicher Handlung. `auto_ok` ist kein
fertiges Label, sondern ein Vorschlag mit hoher Sicherheit.

## 5. Entscheidung „eindeutig / nicht eindeutig“

Reine Funktion `decide(detections, cfg) -> "accept" | "review"` mit
`cfg = {conf_accept: 0.50, conf_review: 0.25, max_boxes: 50, iou_dup: 0.70}`.
Reihenfolge der Prüfungen, erste Übereinstimmung gewinnt:

| Nr. | Bedingung | Ergebnis | Klartextmeldung |
|---|---|---|---|
| 1 | 0 Erkennungen | `review` | „Kein Objekt gefunden“ |
| 2 | irgendeine Box mit `conf < conf_review` | `review` | „Erkennung unsicher (conf < 0.25)“ |
| 3 | zwei Boxen gleicher Klasse mit IoU ≥ 0.70 | `review` | „Doppelte Erkennung, Klasse nicht eindeutig“ |
| 4 | mehr als `max_boxes` Boxen | `review` | „Zu viele Objekte, Modell überfordert“ |
| 5 | irgendeine Box mit `conf < conf_accept` | `review` | „Grenzfall (conf zwischen 0.25 und 0.50)“ |
| 6 | alle Boxen `conf ≥ conf_accept`, keine Regel 2–5 | `accept` | — |

Revision 2 (nach Freigabe, sichtbar): Regel 5 war in Revision 1 nicht enthalten,
obwohl Kriterium C2 (`conf=0.49` → `review`) sie verlangt. Ohne Regel 5 wäre die
SPEC widersprüchlich gewesen.

Alle Schwellen sind CLI-Parameter und stehen im Run-Datensatz.

**Meldung an den Menschen** (vier Wege, alle gefordert):
1. Konsolenzusammenfassung am Ende des `auto`-Laufs („N von M Bildern zur Prüfung“).
2. Logzeile je Bild mit Grund (`annotator.log`, JSON Lines).
3. Datei `review_queue.json` im Arbeitsverzeichnis (Liste Pfad + Grund).
4. UI: Banner mit Anzahl und Filter „Nur Prüfung“ (Standard nach `auto`).

## 6. Oberfläche (Deutsch)

| Bedienung | Wirkung |
|---|---|
| „Bilder importieren“ | Ordner wählen (Serverpfad), rekursiv, Duplikate per SHA-256 überspringen |
| Bildliste links | Status farbig: neu / Vorschlag / Prüfung / fertig |
| Ziehen auf Bild | Box aufziehen |
| Box ziehen / an Ecken ziehen | verschieben / skalieren |
| `1`–`9` | Klasse der gewählten Box setzen |
| `Entf` | Box löschen |
| `Strg+Z` | letzte Änderung rückgängig (nur laufende Sitzung) |
| `A` / `D` oder Pfeile | voriges / nächstes Bild |
| „Speichern“ | automatisch nach jeder Änderung (500 ms entprellt) |
| „Vorschlag übernehmen“ | alle `proposed` Boxen → `accepted`, Status `labeled` |
| „Export YOLO“ | schreibt Datensatzordner + `data.yaml` |

Jede Meldung und jeder Dialogtext ist deutsch. Keine Framework-Defaults in
 Englisch.

## 7. CLI

```
annotator import <ordner> [--recursive]
annotator serve [--port 8765]
annotator auto --model <best.pt> [--conf-accept 0.5] [--conf-review 0.25] [--device cuda|cpu]
annotator review                      # Warteschlange auf der Konsole
annotator train [--epochs 100] [--init pretrained|scratch] [--imgsz 640] [--batch 4] [--device cuda]
annotator export --out <ordner> [--format yolo|coco|both] [--val-fraction 0.2] [--seed 42] [--force]
```

`--init scratch` (Default) startet aus `yolov8n.yaml` ohne fremde Gewichte.
`--init pretrained` lädt `yolov8n.pt` (COCO-Gewichte, Netz nötig).
`--format coco` (Default) schreibt `train/_annotations.coco.json` und
`val/_annotations.coco.json` (RF-DETR-Format).
`--format yolo` schreibt `images/`, `labels/`, `data.yaml`.
`--format both` schreibt beides.

**Hinweis beim Start:** Jeder `train`- und `auto`-Lauf gibt vor dem Start eine
deutsche Empfehlung aus, welcher Init-Weg bei der aktuellen Datenmenge sinnvoll
ist (Regel in Kriterium E7). Der Hinweis ändert nichts, er informiert nur.

Exit-Codes: `0` Erfolg, `1` Fehler, `2` nichts zu tun, `3` teilweise fehlgeschlagen
(z. B. Bilder ohne Label). Nie stiller Erfolg bei Teilfehlern.

## 8. Abnahmekriterien (ausführbar, je Kriterium ein Test)

### A. Geometrie und YOLO-Format
- **A1** `to_yolo(box, width, height)` gibt `class cx cy w h` mit 6 Nachkommastellen;
  Werte in `[0,1]`; Ergebnis ist auf 6 Stellen gleich der Eingabe nach
  `from_yolo` (Round-Trip, Hypothesis-Property, 500 Beispiele).
- **A2** Box, die über den Bildrand ragt, wird auf `[0,1]` beschnitten; `w` oder `h`
  wird nie 0 oder negativ (Mindestbreite 1 px).
- **A3** Box mit `w=0` oder `h=0` wird beim Speichern abgelehnt (`ValueError`,
  Meldung nennt Datei).
- **A4** Datei mit Klassennummer außerhalb `0..len(classes)-1` wird beim Import
  abgelehnt; die Datei bleibt unverändert, Fehler wird protokolliert.

### B. Menschliche Annotation (API + UI-Vertrag)
- **B1** `POST /api/images/{id}/boxes` speichert Box, Antwort enthält `id` und
  normalisierte Werte; erneutes `GET` liefert dieselben Werte.
- **B2** `PATCH` einer Box ändert nur diese Box; andere Boxen desselben Bildes
  bleiben byte-gleich.
- **B3** `DELETE` einer Box entfernt genau eine Zeile; Bild bleibt `labeled`.
- **B4** Nach jeder Mutation setzt der Server `images.updated_at` und schreibt
  genau eine Zeile in `annotator.log`.
- **B5** Server bindet nur `127.0.0.1`; `--host 0.0.0.0` wird abgelehnt
  (Exit-Code 1, Meldung).
- **B6** Import eines Ordners mit 3 Bildern erzeugt 3 Zeilen `status='new'`;
  zweiter Import derselben Dateien erzeugt 0 neue Zeilen (SHA-256-Dedupe).
- **B7** Import nimmt jeden angegebenen Ordner. Regeln: rekursiv nur mit
  `--recursive`; Dateiendungen `.jpg .jpeg .png .bmp .webp .tif .tiff` (Groß/Klein
  gleich); Symlinks werden nicht verfolgt; andere Dateien werden gezählt und
  gemeldet, nicht importiert; Umlaute und Leerzeichen im Pfad sind zulässig.

### C. Entscheidungslogik (reine Funktion)
- **C1** leere Erkennungsliste → `review`, Grund `kein_objekt`.
- **C2** eine Box `conf=0.49` → `review`, Grund `grenzfall`; eine Box `conf=0.50` → `accept`.
- **C3** zwei Boxen gleicher Klasse, IoU 0.8 → `review`, Grund `doppelt`.
  IoU 0.6 → `accept`.
- **C4** 51 Boxen → `review`, Grund `zu_viele`.
- **C5** Box `conf=0.20` → `review`, Grund `unsicher`.
- **C6** Property: `decide` ist deterministisch (gleiche Eingabe, gleiches
  Ergebnis, 200 Beispiele) und liefert nie `accept` bei irgendeiner Box
  `conf < conf_review`.
- **C7** `auto` schreibt für jedes `review`-Bild genau einen Eintrag in
  `review_queue.json` und in `annotator.log`; Anzahl Konsole == Anzahl JSON ==
  Anzahl DB-`needs_review`. Keine Box mit `state='proposed'` erzeugt
  `status='labeled'`.

### D. Export
- **D1** Export bei 20 Bildern und `val_fraction=0.2` erzeugt 16/4 (gerundet),
  `images/{train,val}`, `labels/{train,val}`, `data.yaml` mit `nc`, `names`,
  absoluten Pfaden.
- **D2** Jede Bilddatei im Export hat genau eine Labeldatei; Bild ohne Box
  erhält leere `.txt` (Negativbeispiel), Bild ohne Datei fehlt nirgends.
- **D3** Gleicher `seed` → gleiche Aufteilung (Property: zweimal exportieren,
  Dateilisten identisch).
- **D4** Originalbilder werden nicht verändert (mtime und SHA-256 vor/nach gleich).
- **D5** Export überschreibt einen vorhandenen Zielordner nur mit `--force`;
  ohne `--force` Exit-Code 1 und Meldung.
- **D6** `--format coco` erzeugt für `train` und `val` je eine Datei
  `_annotations.coco.json` mit den Schlüsseln `images`, `annotations`,
  `categories`. Jede Annotation hat `bbox = [x_min, y_min, w, h]` in Pixeln
  (nicht normalisiert) und `area = w*h`, `iscrowd = 0`.
- **D7** `bbox` aus COCO zurückgerechnet ergibt die YOLO-Werte aus der DB
  (Toleranz 1 Pixel, Property-Test).
- **D8** `category_id` beginnt bei 1 und folgt der DB-Reihenfolge von `classes`;
  `annotations[].image_id` verweist auf eine vorhandene `images[].id`.
- **D9** `--format both` erzeugt YOLO- und COCO-Struktur im selben Ordner, ohne
  Namenskonflikt; beide zählen dieselben Bilder.

### E. Training
- **E1** `train` bricht mit Exit-Code 2 und deutscher Meldung ab, wenn weniger als
  2 Bilder `status='labeled'` haben.
- **E2** `train` ruft `ultralytics` mit den übergebenen Parametern auf (Test mit
  Doppelgänger an der Grenze, nicht mit dem Modell selbst).
- **E3** Nach Erfolg existiert `models/run_<id>/best.pt`, `runs.metrics_json`
  enthält `mAP50` als Zahl, `runs.log_path` existiert.
- **E4** Reale Ausführung: 1 Epoche, `imgsz=320`, 8 synthetische Bilder,
  `--init scratch`, `device=cuda` → Exit-Code 0, `best.pt` vorhanden,
  Laufzeit wird protokolliert.
- **E5** `--init scratch` verwendet `yolov8n.yaml` und lädt keine Gewichtsdatei;
  `--init pretrained` verwendet `yolov8n.pt`. Der Run-Datensatz hält fest, welcher
  Weg benutzt wurde (`params_json.init`).
- **E6** Fehlt beim Laden von `yolov8n.pt` das Netz, dann Exit-Code 1 mit deutscher
  Meldung; kein Trainingslauf ohne diese Meldung.
- **E7** Vor dem Start gibt `train` genau eine Hinweiszeile aus. Reine Funktion
  `init_hint(n_labeled, init) -> str`:
  | n_labeled | init | Hinweis enthält |
  |---|---|---|
  | < 200 | `scratch` | Anzahl N, „pretrained“, „meist bessere Ergebnisse“ |
  | < 200 | `pretrained` | Anzahl N, „pretrained“, „passend bei kleinen Datenmengen“ |
  | ≥ 200 | `scratch` | Anzahl N, „scratch“, „ausreichend Daten“ |
  | ≥ 200 | `pretrained` | Anzahl N, „scratch“, „schnellerer Start“ |
  Der Hinweis startet das Training nicht und bricht es nicht ab; er erscheint auch
  bei `--dry-run`.

### F. Automatische Annotation
- **F1** `auto` verarbeitet nur Bilder mit `status ∈ {new}`; bereits `labeled`
  Bilder werden nicht überschrieben (Test: Label-Hash vor/nach gleich).
- **F2** `auto` mit Modell, das nichts findet, setzt alle Bilder auf
  `needs_review` und meldet Grund `kein_objekt`.
- **F3** `auto` ohne `--model` sucht das jüngste `models/run_*/best.pt`; ohne
  Treffer Exit-Code 2 und deutsche Meldung.
- **F4** Abbruch mit `Strg+C` hinterlässt keine halb geschriebenen Boxen
  (Transaktion je Bild).

### G. Invarianten (dürfen nie brechen)
- **G1** Originalbilder werden nie geschrieben, verschoben oder gelöscht.
- **G2** Klassennummern werden nur angehängt, nie neu vergeben.
- **G3** Exportordner enthält nur Dateien aus der DB, keine Waisen.
- **G4** Ohne menschliche Handlung entsteht nie `status='labeled'`.

## 9. Setup-Plan (durch SPEC-Freigabe mitgenehmigt)

| Schritt | Inhalt |
|---|---|
| Git | `git init`, `.gitignore` (`.venv/`, `*.db`, `models/`, `datasets/`, `*.pt`), Checkpoint-Commits bei SPEC und jedem GREEN/REFACTOR |
| Python | `uv init --python 3.12`, `pyproject.toml`, `uv sync` |
| Laufzeit-Abhängigkeiten | `fastapi`, `uvicorn[standard]` (HTTP-Server der UI) · `ultralytics` (YOLO-Training/Inferenz, bringt torch/torchvision/opencv/numpy) · `pillow` (Bildmaße/Format prüfen) |
| Entwicklungs-Abhängigkeiten | `pytest`, `pytest-cov`, `pytest-randomly`, `hypothesis` (Property-Tests), `httpx` (API-Tests), `ruff` (Lint/Format), `mypy` (Typen) |
| CUDA | torch-Wheel für CUDA 12.x aus PyTorch-Index; tatsächliche Version wird in EVIDENCE festgehalten |
| GPU | RTX 4060 Laptop, 8 GB frei; Trainingsläufe klein halten (`imgsz≤640`, `batch≤4`) |
| Testdaten | 12 synthetische PNG (Rechtecke/Ellipsen) werden im Test erzeugt, keine echten Bilder im Repo |

## 10. GAUNTLET (Stufe Tier 3)

| Ebene | Werkzeug / Verfahren |
|---|---|
| Tests | `uv run pytest` (randomisierte Reihenfolge), 0 neue Fehler |
| Typen | `uv run mypy src` |
| Lint/Format | `uv run ruff check . && ruff format --check .` |
| Abdeckung | `--cov` mit Bericht über geänderte Zeilen, Ziel 100 % der neuen Zeilen |
| Mutation | Handmutation: 5 echte Fehler (Vergleich, Grenze, Bedingung, Rückgabe, IoU) müssen von der Suite getötet werden; Reproduktion per `tools/mutate.py` im Repo |
| Property | Hypothesis für A1, C6, D3 |
| Reale Ausführung | `serve` starten, Bild importieren, Box per API setzen, `export --format both` laufen lassen, 1 Epoche trainieren (`--init scratch`), `auto` laufen lassen, Warteschlange prüfen |
| Angriffspass | kaputte Bilder, 0-Byte-Datei, EXIF-Drehung, 10000×10000 px, Pfad mit Umlauten, gleichzeitige Schreibzugriffe auf ein Bild |
| Umgebung | `pip-audit` (bzw. `uv run pip-audit`), Lizenzprüfung, Suche nach Geheimnissen im Diff |

EVIDENCE-Bericht am Ende: jedes Kriterium A1–G4 auf Test oder Ebene abgebildet,
jede Zahl aus einem frischen Lauf nach der letzten Codeänderung.

## 11. Entscheidungen (getroffen am 25.10.2026)

| Nr. | Frage | Entscheidung |
|---|---|---|
| D1 | Oberfläche | **Lokal im Browser** (FastAPI, nur 127.0.0.1) |
| D2 | Startgewichte | **Default `scratch`** (`yolov8n.yaml`, keine fremden Gewichte). `--init pretrained` als Schalter. Vor jedem Trainingslauf erscheint ein Hinweis, welcher Weg bei der aktuellen Datenmenge sinnvoller ist (Kriterium E7) |
| D3 | COCO-Export | **Ja, Default `coco`**: `--format coco|yolo|both`. YOLO weiter verfügbar (eigene Trainingsdaten) |
| D4 | Schwellen | **Ja**, 0.50 sicher / 0.25 unsicher, beide per CLI änderbar |
| D5 | Bilder | **Jeder angegebene Ordner**; Regeln in Kriterium B7 |

## 12. Freigabe

Ohne ausdrückliche Freigabe von Frank beginnt keine Implementierung.
Änderung der SPEC öffnet die Freigabe erneut.

Stand: Entscheidungen D1–D5 eingearbeitet. **Wartet auf Freigabe.**
