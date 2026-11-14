"""HTTP-API für die Browser-Oberfläche."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .db import BoxRow, ImageRow, Store
from .exporter import export_dataset
from .geometry import Box, BoxError
from .importer import import_folder, import_labels

STATIC_DIR = Path(__file__).parent / "static"


class BoxIn(BaseModel):
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


class BoxPatch(BaseModel):
    class_id: int | None = None
    cx: float | None = None
    cy: float | None = None
    w: float | None = None
    h: float | None = None


class ClassIn(BaseModel):
    name: str
    color: str = "#ff0000"


class ImportIn(BaseModel):
    folder: str
    recursive: bool = False
    labels: bool = False


class ExportIn(BaseModel):
    out: str
    format: str = "coco"
    val_fraction: float = 0.2
    seed: int = 42
    force: bool = False


def _box_dict(row: BoxRow) -> dict[str, object]:
    return {
        "id": row.id,
        "image_id": row.image_id,
        "class_id": row.class_id,
        "cx": row.cx,
        "cy": row.cy,
        "w": row.w,
        "h": row.h,
        "source": row.source,
        "confidence": row.confidence,
        "state": row.state,
    }


def _image_dict(store: Store, image: ImageRow) -> dict[str, object]:
    return {
        "id": image.id,
        "path": image.path,
        "name": Path(image.path).name,
        "width": image.width,
        "height": image.height,
        "status": image.status,
        "boxes": len(store.boxes(image.id)),
    }


def create_app(store: Store, log_path: Path) -> FastAPI:
    app = FastAPI(title="Annotator", version=__version__)
    log_file = Path(log_path)

    def log(event: str, image_id: int | None = None, **detail: object) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
            "event": event,
            "image_id": image_id,
            **detail,
        }
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def require_image(image_id: int) -> ImageRow:
        try:
            return store.image(image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def require_box(box_id: int) -> BoxRow:
        try:
            return store.box(box_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/classes")
    def list_classes() -> list[dict[str, object]]:
        return [{"id": c.id, "name": c.name, "color": c.color} for c in store.classes()]

    @app.post("/api/classes", status_code=201)
    def create_class(payload: ClassIn) -> dict[str, object]:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Klassenname ist leer")
        try:
            class_id = store.add_class(name, payload.color)
        except Exception as exc:  # sqlite3.IntegrityError: Name doppelt
            raise HTTPException(status_code=400, detail=f"Klasse nicht angelegt: {exc}") from exc
        return {"id": class_id, "name": name, "color": payload.color}

    @app.get("/api/images")
    def list_images(status: str | None = None) -> list[dict[str, object]]:
        return [_image_dict(store, image) for image in store.images(status)]

    @app.get("/api/images/{image_id}")
    def get_image(image_id: int) -> dict[str, object]:
        image = require_image(image_id)
        return {
            **_image_dict(store, image),
            "boxes": [_box_dict(row) for row in store.boxes(image_id)],
            "classes": [{"id": c.id, "name": c.name, "color": c.color} for c in store.classes()],
        }

    @app.get("/api/images/{image_id}/file")
    def get_image_file(image_id: int) -> FileResponse:
        image = require_image(image_id)
        return FileResponse(image.path)

    @app.post("/api/images/{image_id}/boxes", status_code=201)
    def create_box(image_id: int, payload: BoxIn) -> dict[str, object]:
        require_image(image_id)
        box = Box(payload.class_id, payload.cx, payload.cy, payload.w, payload.h)
        try:
            box_id = store.add_box(image_id, box)
        except BoxError as exc:
            raise HTTPException(status_code=400, detail=f"Ungültige Box: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        log("box_create", image_id, box_id=box_id, **payload.model_dump())
        return _box_dict(store.box(box_id))

    @app.patch("/api/boxes/{box_id}")
    def patch_box(box_id: int, payload: BoxPatch) -> dict[str, object]:
        require_box(box_id)
        fields = {key: value for key, value in payload.model_dump().items() if value is not None}
        if not fields:
            raise HTTPException(status_code=400, detail="Keine Änderung übergeben")
        try:
            store.update_box(box_id, **fields)
        except BoxError as exc:
            raise HTTPException(status_code=400, detail=f"Ungültige Box: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        row = store.box(box_id)
        log("box_update", row.image_id, box_id=box_id, fields=fields)
        return _box_dict(row)

    @app.delete("/api/boxes/{box_id}")
    def remove_box(box_id: int) -> dict[str, object]:
        row = require_box(box_id)
        store.delete_box(box_id)
        log("box_delete", row.image_id, box_id=box_id)
        return {"deleted": box_id}

    @app.post("/api/images/{image_id}/accept")
    def accept_proposals(image_id: int) -> dict[str, object]:
        image = require_image(image_id)
        for row in store.boxes(image_id):
            if row.state != "accepted":
                store.update_box(row.id, state="accepted")
        store.set_status(image_id, "labeled", by="human")
        log("image_accept", image_id, box_count=len(store.boxes(image_id)))
        return {
            "id": image_id,
            "status": store.image(image_id).status,
            "boxes": len(store.boxes(image_id)),
            "path": image.path,
        }

    @app.post("/api/import")
    def import_images(payload: ImportIn) -> dict[str, object]:
        ordner = Path(payload.folder).expanduser()
        if not ordner.is_dir():
            raise HTTPException(status_code=400, detail=f"Ordner nicht gefunden: {ordner}")
        try:
            result = import_folder(store, ordner, recursive=payload.recursive)
            labels = (
                import_labels(store, ordner, recursive=payload.recursive)
                if payload.labels
                else None
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Import fehlgeschlagen: {exc}") from exc
        log(
            "import",
            None,
            folder=str(ordner),
            added=result.added,
            duplicates=result.duplicates,
            skipped=result.skipped,
        )
        return {
            "added": result.added,
            "duplicates": result.duplicates,
            "skipped": result.skipped,
            "unreadable": result.unreadable,
            "labels": None
            if labels is None
            else {"boxes": labels.boxes, "files": labels.files, "rejected": labels.rejected},
        }

    @app.post("/api/export")
    def export(payload: ExportIn) -> dict[str, object]:
        ziel = Path(payload.out).expanduser()
        try:
            result = export_dataset(
                store,
                ziel,
                fmt=payload.format,
                val_fraction=payload.val_fraction,
                seed=payload.seed,
                force=payload.force,
            )
        except FileExistsError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc.args[0] if exc.args else exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        log(
            "export",
            None,
            out=str(result.out_dir),
            train=result.train,
            val=result.val,
            format=payload.format,
        )
        return {
            "out_dir": str(result.out_dir),
            "train": result.train,
            "val": result.val,
            "files": result.files,
        }

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        queue = [
            {"id": image.id, "path": image.path, "name": Path(image.path).name}
            for image in store.images("needs_review")
        ]
        return {
            "counts": store.counts(),
            "review_queue": queue,
            "classes": len(store.classes()),
        }

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
