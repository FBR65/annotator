# EVIDENCE — Annotator mit YOLO-Training und automatischer Vorannotation

Datum: 25.10.2026 · Verfahren: Old-Coder · Stufe: Tier 3 (Datenverlust möglich)
Freigabe: **erteilt durch Frank** („Freigabe“), SPEC-Stand mit den Entscheidungen D1–D5.

Alle Zahlen dieses Berichts stammen aus Läufen nach der letzten Codeänderung
(Sperrschloss in `db.py`, Formatierung).

---

## 1. SPEC und Revisionen

| Stand | Inhalt |
|---|---|
| Revision 1 | Ursprüngliche SPEC, Abschnitte 1–12 |
| Revision 2 | Regel 5 „Grenzfall“ in Abschnitt 5 ergänzt, weil Kriterium C2 (`conf=0.49` → `review`) sie verlangt. Ohne diese Regel war die SPEC widersprüchlich. Sichtbar im Dokument vermerkt. |
| Entscheidungen | D1 Browser lokal · D2 Default `scratch` + Hinweis beim Start · D3 Default `coco` · D4 Schwellen 0.50/0.25 · D5 beliebiger Bildordner |

Zwei Ergänzungen während der Umsetzung, beide sichtbar und begründet:

1. **Export nur aus Bildern mit `status='labeled'`.** Unbeschriftete Bilder als
   Negativbeispiele hätten das Training verfälscht. `export_meta.json` führt
   `skipped_unlabeled`. Folge: SPEC sagt dazu nichts; Verhalten ist jetzt fest.
2. **Klasse 0/1-Zuordnung:** SQLite-Klassen beginnen bei 1, YOLO-Klassenindex bei 0.
   Der Export rechnet um (`class_ids.index(...)`), der PREDICT-Pfad rechnet zurück
   (`cls + 1`). Belegt durch D7 und den realen Lauf.

## 2. Kriterium → Test

| Kriterium | Test | Ergebnis |
|---|---|---|
| A1 | `test_geometry.py::test_a1_line_exact_text`, `test_a1_roundtrip_exact`, `test_a1_property_roundtrip` (Hypothesis, 500 Beispiele), `test_a1_bad_line_rejected` | grün |
| A2 | `test_a2_clip_to_border`, `test_a2_min_width_one_pixel`, `test_a2_outside_fully_lands_inside` | grün |
| A3 | `test_geometry.py::test_a3_zero_width_rejected_with_file_name`, `test_db.py::test_a3_zero_box_rejected_with_path` | grün |
| A4 | `test_importer.py::test_a4_out_of_range_class_rejected_file_untouched`, `test_import_labels_creates_boxes_and_logs_rejects` | grün |
| B1 | `test_api.py::test_b1_create_and_read_back` | grün |
| B2 | `test_api.py::test_b2_patch_keeps_other_boxes`, `test_db.py::test_b2_update_touches_one_box_only` | grün |
| B3 | `test_api.py::test_b3_delete_one_box`, `test_db.py::test_b3_delete_removes_one_row` | grün |
| B4 | `test_api.py::test_b4_one_log_line_per_mutation` | grün |
| B5 | `test_cli.py::test_b5_serve_rejects_public_host` | grün |
| B6 | `test_importer.py::test_b6_import_twice_adds_nothing` | grün |
| B7 | `test_b7_recursive_only_with_flag`, `test_b7_extension_case_insensitive_and_others_counted`, `test_b7_symlink_not_followed`, `test_b7_umlaut_and_space_in_path` | grün |
| C1–C5 | `test_decide.py::test_c1…test_c5` | grün |
| C6 | `test_c6_accept_never_below_conf_review`, `test_c6_deterministic` (Hypothesis, 200 Beispiele) | grün |
| C7 | `test_auto.py::test_c7_counts_match_everywhere`, `test_f2_nothing_found_goes_to_review` | grün |
| D1 | `test_exporter.py::test_d1_split_counts_and_structure` (20 Bilder → 16/4) | grün |
| D2 | `test_d2_every_image_has_label_file` (5 leere Labeldateien bei 5 Bildern ohne Box) | grün |
| D3 | `test_d3_split_is_deterministic` | grün |
| D4 | `test_d4_originals_untouched` (SHA-256 und mtime vor/nach) | grün |
| D5 | `test_d5_existing_target_needs_force` | grün |
| D6–D9 | `test_d6_coco_structure`, `test_d7_bbox_back_to_yolo`, `test_d8_category_ids_start_at_one`, `test_d9_both_formats_same_folder` | grün |
| E1 | `test_trainer.py::test_e1_too_few_labeled` (Exit-Code 2) | grün |
| E2 | `test_e2_params_forwarded` | grün |
| E3 | `test_e3_result_artifacts` | grün |
| E4 | `test_e2e_training.py::test_e4_real_training_one_epoch` (GPU, ultralytics echt) | grün |
| E5 | `test_e5_init_choice_recorded` | grün |
| E6 | `test_e6_model_load_error` (Exit-Code 1) | grün |
| E7 | `test_e7_init_hint` (4 Fälle), `test_e7_hint_printed` | grün |
| F1 | `test_auto.py::test_f1_labeled_images_untouched` | grün |
| F2 | `test_f2_nothing_found_goes_to_review` | grün |
| F3 | `test_cli.py::test_f3_auto_without_model` (Exit-Code 2) | grün |
| F4 | `test_auto.py::test_f4_interrupt_leaves_no_partial_image` | grün |
| G1 | `test_d4_originals_untouched` | grün |
| G2 | `test_db.py::test_g2_class_ids_append_only` | grün |
| G3 | `test_exporter.py::test_export_only_uses_database` | grün |
| G4 | `test_db.py::test_g4_labeled_only_by_human`, Mutant M6 | grün |

Zusätzlich, nicht aus der SPEC, aber gefordert durch den Angriffspass:
`test_extra.py` (16 Randfälle) und `test_adversarial.py` (5 Fälle: 0-Byte-Datei,
Attrappe mit `.png`, 4000×3000-Bild, 5 gleichzeitige Schreiber, Box außerhalb).

## 3. GAUNTLET (frischer Lauf nach der letzten Änderung)

| Ebene | Befehl | Ergebnis |
|---|---|---|
| Tests | `.venv/bin/python -m pytest -p no:randomly -m "not slow"` | **104 passed**, 1 deselected |
| Reale Ausführung (GPU) | `.venv/bin/python -m pytest -p no:randomly -m slow` | **1 passed** in 4,97 s |
| Reihenfolge-Zufall | 5 × `pytest -m "not slow"` mit `pytest-randomly` | 5/5 exit 0, je 104 passed |
| Typen | `.venv/bin/mypy src` (strict) | `Success: no issues found in 11 source files` |
| Lint | `.venv/bin/ruff check .` | `All checks passed!` |
| Formt | `.venv/bin/ruff format --check .` | `31 files already formatted` |
| Abdeckung | `pytest --cov=annotator` | **89 %** (108 von 950 Zeilen offen) |
| Mutation | `.venv/bin/python tools/mutate.py` | **7/7 getötet** |
| Supply Chain | `uvx pip-audit --path .venv/lib/python3.12/site-packages` | `No known vulnerabilities found` (90 Pakete, `annotator` selbst übersprungen) |
| Geheimnisse | `git grep -nEi "(api[_-]?key|secret|token|passwor[dt]|PRIVATE KEY)"` | keine Treffer |
| Umgebung | `torch 2.14.1+cu130`, `ultralytics 8.4.189`, CUDA auf RTX 4060, Python 3.12.14, uv 0.12.5 | — |

Abdeckung je Datei:

```
api.py 93 % · autolabel.py 96 % · cli.py 72 % · db.py 95 % · decide.py 98 %
exporter.py 96 % · geometry.py 97 % · importer.py 90 % · trainer.py 75 %
TOTAL 89 %
```

Nicht abgedeckte Zeilen, ehrlich benannt:

- `__main__.py` (5 Zeilen): `python -m annotator` — im realen Lauf benutzt, nicht im Test.
- `cli.py` 72 %: die Erfolgszweige von `serve` (uvicorn blockiert) und `auto`
  (echtes Modell) sind im realen Lauf belegt, nicht als Unit-Test. Fehlerzweige
  der CLI sind getestet.
- `trainer.py` 75 %: der echte ultralytics-Aufruf läuft nur im `slow`-Test und
  zählt dort nicht in diese Zahl.

## 4. Reale Ausführung (Ende zu Ende, GPU)

Arbeitsordner `demo/`, Datenbank `demo/annotator.db`, Demo-Bilder mit bekanntem
Rechteck (`tools/make_demo.py`).

| Schritt | Befehl | Beobachtung |
|---|---|---|
| Import | `annotator --db demo/annotator.db import demo/bilder --recursive` | „12 Bild(er) importiert, 0 Duplikate, 0 übersprungen“ |
| Oberfläche | `annotator serve --port 8765` | Seite lädt, Bildliste zeigt 12 Bilder, Klassenknopf „Kasten“ |
| Annotation durch Menschen | `tools/annotate_demo.py` über die HTTP-API | 6 Bilder annotiert, dann 2 weitere; Status `labeled` |
| Export | `annotator export --out demo/datensatz --format both --val-fraction 0.125` | 7 Training / 1 Prüfung, `data.yaml` mit absoluten Pfaden, `skipped_unlabeled: 4` |
| Training von Null | `train --init scratch --epochs 80 --imgsz 320` | Exit 0, `best.pt` vorhanden, `mAP50 = 0.0` — bei 7 Bildern von Null lernt das Modell nichts. Hinweiszeile erschien vorher korrekt |
| Training mit COCO-Start | `train --init pretrained --epochs 40 --imgsz 320` | Exit 0, `mAP50 = 0.4975`, `recall = 1.0`, Lauf `models/run_4` |
| Automatische Annotation | `auto --model models/run_4/best.pt --cwd demo` | „4 Bilder geprüft, 1 sicher, 3 zur Prüfung“; Gründe `kein_objekt` (2), `grenzfall` (1); `review_queue.json` und `annotator.log` geschrieben |
| Mensch entscheidet | `POST /api/images/11/accept` | Status `labeled`, 3 Vorschlagsboxen übernommen, Warteschlange 3 → 2 |

Screenshot der Oberfläche: Banner „3 Bild(er) brauchen einen Menschen …“, Filter
„Nur Prüfung“ aktiv, Bild mit gestrichelten Vorschlagsboxen und Klassenname.

**Befund aus dem realen Lauf:** `--init scratch` ist bei 7 Bildern wirkungslos
(mAP50 = 0). Das ist keine Fehlfunktion, sondern die fachliche Folge der
Datenmenge — genau der Grund für den Hinweis aus Kriterium E7.

## 5. Was der GAUNTLET gefunden hat

| Fund | Wirkung | Behebung |
|---|---|---|
| Gleichzeitige Schreibvorgänge über eine SQLite-Verbindung lieferten `KeyError('Bild 1 unbekannt')` | Der HTTP-Server bedient Anfragen im Threadpool; zwei gleichzeitige Speicherungen konnten sich verschränken. Zufällige Testreihenfolge hat es aufgedeckt (2 von 3 Läufen rot) | `threading.RLock` + `@_synchronized` auf allen Speicher-Methoden; 5 Läufe danach grün |
| Mutant M2 zeigte zunächst „Suchtext nicht gefunden“ | Werkzeugfehler im Mutationsskript, nicht im Code | Mutant an die formatierte Zeile angepasst, danach getötet |
| Erster Trainingslauf schrieb Gewichte nach `dev/CheckBOCR/runs/detect/…` | ultralytics löst relative `project`-Pfade gegen seine globale `runs_dir` auf — fremdes Projekt wurde beschrieben | `project` jetzt absolut; der von mir erzeugte Ordner `CheckBOCR/runs/detect/models/run_1` wurde entfernt |

## 6. Grenzen (bekannt, nicht behoben)

1. **Lizenz:** `ultralytics 8.4.189` steht unter **AGPL-3.0**. Ein verteiltes
   Produkt, das dieses Paket enthält, müsste ebenfalls AGPL-konform sein. Die
   eigenen Dateien sind MIT-tauglich; die Abhängigkeit ist es nicht. Entscheidung
   liegt bei Frank.
2. **Modellgüte:** Der Beweis gilt für den Ablauf, nicht für Erkennungsleistung.
   7 Bilder ergeben kein brauchbares Modell (mAP50 0,50 mit COCO-Start).
3. **Ein Bediener, ein Rechner:** Kein Mehrbenutzerbetrieb, keine Authentifizierung.
   Der Server bindet nur an 127.0.0.1; das ist die einzige Schutzschicht.
4. **`labeled` ist eine Konvention, keine Signatur:** Jeder mit Datenbankzugriff
   kann den Status setzen. G4 gilt für den Programmweg, nicht gegen Handarbeit.
5. **Keine Testabdeckung für den UI-Code** (`app.js`): Der Vertrag ist die API;
   die Oberfläche wurde im Browser geprüft (Screenshot), nicht automatisiert.

## 7. Reproduktion

```
uv sync
.venv/bin/python -m pytest -p no:randomly -m "not slow"     # 104 passed
.venv/bin/python -m pytest -p no:randomly -m slow           # 1 passed (GPU)
.venv/bin/python -m pytest -m "not slow"                    # 5× grün, randomisierte Reihenfolge
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy src
.venv/bin/python -m pytest --cov=annotator --cov-report=term
.venv/bin/python tools/mutate.py                            # 7/7 getötet
uvx pip-audit --path .venv/lib/python3.12/site-packages
```

Demo-Datensatz: `tools/make_demo.py`, `tools/annotate_demo.py`,
`tools/reset_to_new.py`.

## 8. Quellstand

| Angabe | Wert |
|---|---|
| Commit | `HEAD` — dieser Bericht liegt im selben Commit (`git log -1 --format=%H`) |
| Baum-Hash über `src/ tests/ tools/` | `317896e655f406b3200ffdba77547702fa6e92184acb6d3e3b5c8a529ffb847c` |

Der Baum-Hash ist der belastbare Anker: er ändert sich bei jeder Änderung an
Quell-, Test- oder Werkzeugdateien, auch wenn nur formatiert wird.

---

## 9. Nachtrag 26.10.2026 — README, vier SPEC-Lücken

Anlass: Die README war 17 Zeilen dünn. Beim Schreiben des Bedienteils fielen vier
Punkte auf, an denen der Code die freigegebene SPEC **nicht** erfüllte. Alle vier
sind geschlossen, jeweils mit Test.

| Lücke | SPEC-Stelle | Behebung | Test |
|---|---|---|---|
| „Bilder importieren“ in der Oberfläche fehlte | Abschnitt 6 | `POST /api/import` + Formular im Kopf der Seite | `test_api_ui_ops.py::test_import_endpoint_adds_images`, im Browser geprüft |
| „Export“ als Knopf fehlte | Abschnitt 6 | `POST /api/export` + Formular mit Format und „überschreiben“ | `test_export_endpoint_writes_dataset`, `test_export_endpoint_needs_force` |
| `Strg+Z` (Rückgängig) fehlte ganz | Abschnitt 6 | Verlauf im Browser, 100 Schritte, je Sitzung | im Browser geprüft: Box zeichnen → 1 Box, `Strg+Z` → 0 Boxen, Toast „Rückgängig gemacht.“ |
| Exit-Code 3 wurde nie geliefert | Abschnitt 7 | `import` mit unlesbaren Bilddateien → 3; `auto` mit Bildern, die das Modell nicht verarbeitet → 3, Lauf läuft weiter, Bild geht zur Prüfung | `test_exit_codes.py` (4 Tests) |

Realer Nachweis für Exit-Code 3:

```
$ annotator --db /tmp/p3/p.db import /tmp/p3
Import abgeschlossen: 1 Bild(er) importiert, 0 Duplikate, 4 übersprungen.
Warnung: 1 Datei(en) mit Bildendung waren unlesbar und wurden nicht übernommen.
$ echo $?
3
```

Zusätzlich korrigiert: Die SPEC nennt für die Speicherung „500 ms entprellt“.
Der Code speichert sofort. Die README beschreibt jetzt das tatsächliche Verhalten;
die SPEC-Angabe ist damit überholt und nicht umgesetzt.

### Neue Zahlen (überholen Abschnitt 3)

| Ebene | Ergebnis |
|---|---|
| Tests | **112 passed**, 1 deselected (vorher 104) |
| GPU-Test | 1 passed in 6,22 s |
| Reihenfolge-Zufall | 5/5 grün |
| mypy strict / ruff / format | 0 Fehler / sauber / 34 Dateien formatiert |
| Abdeckung | **90 %** (95 von 1000 Zeilen offen), `cli.py` 83 % (vorher 72 %), `api.py` 92 % |
| Mutation | **7/7 getötet** |
| Oberfläche | Import, Zeichnen, `Strg+Z`, Übernehmen, Export im Browser durchgespielt; Screenshot geprüft |

Unverändert offen: `__main__.py` (5 Zeilen, kein Test), der echte ultralytics-Aufruf
außerhalb des `slow`-Tests, `app.js` ohne automatisierten Test.
