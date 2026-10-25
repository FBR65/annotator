"""Kommandozeile: import, serve, review, auto, train, export."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .autolabel import run_auto
from .db import Store
from .decide import DecideConfig, Detection
from .exporter import export_dataset
from .importer import import_folder, import_labels
from .trainer import TrainingError, train

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class _YoloPredictor:
    """Lädt ultralytics erst beim ersten Bild."""

    def __init__(self, model_path: Path, device: str, conf: float) -> None:
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self._model: object | None = None

    def _ensure(self) -> object:
        if self._model is None:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._model = YOLO(str(self.model_path))
        return self._model

    def __call__(self, path: str) -> list[Detection]:
        model = self._ensure()
        results = model.predict(path, conf=self.conf, device=self.device, verbose=False)  # type: ignore[attr-defined]
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for row in boxes:
                x0, y0, x1, y1 = (float(value) for value in row.xyxy[0])
                width = float(result.orig_shape[1])
                height = float(result.orig_shape[0])
                detections.append(
                    Detection(
                        class_id=int(row.cls[0]) + 1,
                        cx=(x0 + x1) / 2 / width,
                        cy=(y0 + y1) / 2 / height,
                        w=(x1 - x0) / width,
                        h=(y1 - y0) / height,
                        confidence=float(row.conf[0]),
                    )
                )
        return detections


def _find_model(store: Store, cwd: Path) -> Path | None:
    stored = store.latest_model()
    if stored and Path(stored).is_file():
        return Path(stored)
    candidates = sorted(cwd.glob("models/run_*/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotator", description="Bildannotator mit YOLO")
    parser.add_argument("--db", default="annotator.db", help="Pfad zur SQLite-Datei")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="Bildordner einlesen")
    p_import.add_argument("folder")
    p_import.add_argument("--recursive", action="store_true", help="Unterordner mitlesen")
    p_import.add_argument("--labels", action="store_true", help="vorhandene YOLO-Labels lesen")

    p_serve = sub.add_parser("serve", help="Browser-Oberfläche starten")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    sub.add_parser("review", help="Warteschlange auf der Konsole zeigen")

    p_auto = sub.add_parser("auto", help="automatisch vorannotieren")
    p_auto.add_argument("--model", default=None)
    p_auto.add_argument("--cwd", default=".")
    p_auto.add_argument("--conf-accept", type=float, default=0.50)
    p_auto.add_argument("--conf-review", type=float, default=0.25)
    p_auto.add_argument("--device", default="cuda")

    p_train = sub.add_parser("train", help="YOLO trainieren")
    p_train.add_argument("--cwd", default=".")
    p_train.add_argument("--dataset", default=None)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--init", choices=["scratch", "pretrained"], default="scratch")
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=4)
    p_train.add_argument("--device", default="cuda")
    p_train.add_argument("--dry-run", action="store_true")

    p_export = sub.add_parser("export", help="Datensatz schreiben")
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--format", choices=["coco", "yolo", "both"], default="coco")
    p_export.add_argument("--val-fraction", type=float, default=0.2)
    p_export.add_argument("--seed", type=int, default=42)
    p_export.add_argument("--force", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    db_path = Path(args.db)

    if args.command == "import":
        store = Store(db_path)
        try:
            import_result = import_folder(store, Path(args.folder), recursive=args.recursive)
            print(
                f"Import abgeschlossen: {import_result.added} Bild(er) importiert, "
                f"{import_result.duplicates} Duplikate, {import_result.skipped} übersprungen."
            )
            if import_result.unreadable:
                print(
                    f"Warnung: {import_result.unreadable} Datei(en) mit Bildendung waren "
                    "unlesbar und wurden nicht übernommen."
                )
            if args.labels:
                label_result = import_labels(store, Path(args.folder), recursive=args.recursive)
                print(
                    f"Labels: {label_result.boxes} Boxen aus {label_result.files} Dateien, "
                    f"{label_result.rejected} abgelehnt."
                )
        finally:
            store.close()
        return 3 if import_result.unreadable else 0

    if args.command == "serve":
        if args.host not in LOOPBACK:
            print(
                f"Fehler: Nur 127.0.0.1 ist erlaubt (übergeben: {args.host}). "
                "Die Bilder bleiben lokal."
            )
            return 1
        import uvicorn

        from .api import create_app

        store = Store(db_path)
        app = create_app(store, Path("annotator.log"))
        print(f"Oberfläche bereit: http://{args.host}:{args.port}/")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        store.close()
        return 0

    if args.command == "review":
        store = Store(db_path)
        try:
            queue = store.images("needs_review")
            counts = store.counts()
            if not queue:
                print("Warteschlange: keine Bilder zur Prüfung.")
            else:
                print(f"Warteschlange: {len(queue)} Bilder zur Prüfung.")
                for image in queue:
                    print(f"  #{image.id}  {image.path}")
            print("Bestand: " + ", ".join(f"{status}={count}" for status, count in counts.items()))
        finally:
            store.close()
        return 0

    if args.command == "auto":
        cwd = Path(args.cwd)
        store = Store(db_path)
        try:
            model = Path(args.model) if args.model else _find_model(store, cwd)
            if model is None or not model.is_file():
                print("Fehler: Kein Modell gefunden. Bitte zuerst trainieren oder --model angeben.")
                return 2
            cfg = DecideConfig(conf_accept=args.conf_accept, conf_review=args.conf_review)
            predictor = _YoloPredictor(model, args.device, cfg.conf_review)
            print(f"Vorannotation mit Modell {model} ...")
            auto_result = run_auto(
                store,
                predictor,
                cfg,
                queue_path=cwd / "review_queue.json",
                log_path=cwd / "annotator.log",
            )
            print(
                f"Fertig: {auto_result.processed} Bilder geprüft, {auto_result.accepted} sicher, "
                f"{auto_result.review} zur Prüfung."
            )
            for reason, count in sorted(auto_result.reasons.items()):
                print(f"  {reason}: {count}")
            if auto_result.review:
                print(f"Warteschlange geschrieben: {cwd / 'review_queue.json'}")
                print("Bitte diese Bilder im Browser prüfen: annotator serve")
            if auto_result.failed:
                print(
                    f"Warnung: {auto_result.failed} Bild(er) konnten nicht vorannotiert werden "
                    "und stehen zur Prüfung."
                )
        finally:
            store.close()
        return 3 if auto_result.failed else 0

    if args.command == "train":
        cwd = Path(args.cwd)
        dataset = Path(args.dataset) if args.dataset else cwd / "dataset"
        store = Store(db_path)
        try:
            train_result = train(
                store,
                cwd / "models",
                dataset=dataset,
                init=args.init,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                dry_run=args.dry_run,
            )
        except TrainingError as exc:
            print(f"Fehler: {exc.message}")
            return exc.exit_code
        finally:
            store.close()
        if train_result.run_id:
            metrics = ", ".join(f"{key}={value:.4f}" for key, value in train_result.metrics.items())
            print(f"Metriken: {metrics or 'keine'}")
        return 0

    if args.command == "export":
        store = Store(db_path)
        try:
            export_result = export_dataset(
                store,
                Path(args.out),
                fmt=args.format,
                val_fraction=args.val_fraction,
                seed=args.seed,
                force=args.force,
            )
        except FileExistsError as exc:
            print(f"Fehler: {exc.args[0] if exc.args else exc}")
            return 1
        except ValueError as exc:
            print(f"Fehler: {exc}")
            return 1
        finally:
            store.close()
        print(
            f"Export fertig: {export_result.train} Bilder Training, "
            f"{export_result.val} Bilder Prüfung, Format {args.format} → {export_result.out_dir}"
        )
        return 0

    print(f"Fehler: unbekannter Befehl {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
