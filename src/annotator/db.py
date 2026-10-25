"""SQLite-Speicher: Bilder, Klassen, Boxen, Läufe."""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .geometry import Box, validate_box

STATUSES = ("new", "auto_ok", "needs_review", "labeled")
BOX_FIELDS = frozenset({"class_id", "cx", "cy", "w", "h", "source", "confidence", "state"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    color TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS boxes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    w REAL NOT NULL,
    h REAL NOT NULL,
    source TEXT NOT NULL,
    confidence REAL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    params_json TEXT NOT NULL,
    metrics_json TEXT,
    model_path TEXT,
    log_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_boxes_image ON boxes(image_id);
"""


@dataclass(frozen=True)
class ClassRow:
    id: int
    name: str
    color: str


@dataclass(frozen=True)
class ImageRow:
    id: int
    path: str
    sha256: str
    width: int
    height: int
    status: str
    updated_at: str


@dataclass(frozen=True)
class BoxRow:
    id: int
    image_id: int
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    source: str
    confidence: float | None
    state: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _synchronized[F: Callable[..., Any]](method: F) -> F:
    """Hält das Verbindungsschloss für die Dauer des Aufrufs."""

    @functools.wraps(method)
    def wrapper(self: Store, *args: object, **kwargs: object) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class Store:
    """Ein Schreiber, eine Datei. Jeder Aufruf schreibt sofort (Autocommit)."""

    def __init__(self, db_path: Path | str) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    @_synchronized
    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Mehrere Schreibvorgänge als eine Einheit. Bei Abbruch: Rollback."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # --- Klassen -----------------------------------------------------------
    @_synchronized
    def add_class(self, name: str, color: str = "#ff0000") -> int:
        cur = self._conn.execute("INSERT INTO classes (name, color) VALUES (?, ?)", (name, color))
        return int(cur.lastrowid or 0)

    @_synchronized
    def classes(self) -> list[ClassRow]:
        rows = self._conn.execute("SELECT id, name, color FROM classes ORDER BY id").fetchall()
        return [ClassRow(r["id"], r["name"], r["color"]) for r in rows]

    # --- Bilder ------------------------------------------------------------
    @_synchronized
    def add_image(self, path: str, sha256: str, width: int, height: int) -> int | None:
        try:
            cur = self._conn.execute(
                "INSERT INTO images (path, sha256, width, height, status, updated_at)"
                " VALUES (?, ?, ?, ?, 'new', ?)",
                (path, sha256, width, height, _now()),
            )
        except sqlite3.IntegrityError:
            return None
        return int(cur.lastrowid or 0)

    @_synchronized
    def images(self, status: str | None = None) -> list[ImageRow]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM images ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM images WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        return [self._image_row(row) for row in rows]

    @staticmethod
    def _image_row(row: sqlite3.Row) -> ImageRow:
        return ImageRow(
            row["id"],
            row["path"],
            row["sha256"],
            row["width"],
            row["height"],
            row["status"],
            row["updated_at"],
        )

    @_synchronized
    def image(self, image_id: int) -> ImageRow:
        row = self._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(f"Bild {image_id} unbekannt")
        return self._image_row(row)

    @_synchronized
    def image_by_path(self, path: str) -> ImageRow | None:
        row = self._conn.execute("SELECT * FROM images WHERE path = ?", (path,)).fetchone()
        return None if row is None else self._image_row(row)

    @_synchronized
    def set_status(self, image_id: int, status: str, by: str = "human") -> None:
        if status not in STATUSES:
            raise ValueError(f"Unbekannter Status: {status}")
        if status == "labeled" and by != "human":
            raise PermissionError("Status 'labeled' nur nach menschlicher Handlung")
        self._conn.execute(
            "UPDATE images SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), image_id),
        )

    # --- Boxen -------------------------------------------------------------
    @_synchronized
    def add_box(
        self,
        image_id: int,
        box: Box,
        source: str = "human",
        confidence: float | None = None,
        state: str = "accepted",
    ) -> int:
        image = self.image(image_id)
        validate_box(box, image.path)
        if not any(item.id == box.class_id for item in self.classes()):
            raise ValueError(f"Klasse {box.class_id} unbekannt")
        stamp = _now()
        cur = self._conn.execute(
            "INSERT INTO boxes (image_id, class_id, cx, cy, w, h, source, confidence, state,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                box.class_id,
                box.cx,
                box.cy,
                box.w,
                box.h,
                source,
                confidence,
                state,
                stamp,
            ),
        )
        self._conn.execute("UPDATE images SET updated_at = ? WHERE id = ?", (stamp, image_id))
        return int(cur.lastrowid or 0)

    @_synchronized
    def box(self, box_id: int) -> BoxRow:
        row = self._conn.execute("SELECT * FROM boxes WHERE id = ?", (box_id,)).fetchone()
        if row is None:
            raise KeyError(f"Box {box_id} unbekannt")
        return self._box_row(row)

    @staticmethod
    def _box_row(row: sqlite3.Row) -> BoxRow:
        return BoxRow(
            row["id"],
            row["image_id"],
            row["class_id"],
            row["cx"],
            row["cy"],
            row["w"],
            row["h"],
            row["source"],
            row["confidence"],
            row["state"],
        )

    @_synchronized
    def update_box(self, box_id: int, **fields: object) -> None:
        unknown = set(fields) - BOX_FIELDS
        if unknown:
            raise ValueError(f"Unbekannte Felder: {sorted(unknown)}")
        current = self.box(box_id)
        merged = Box(
            class_id=int(str(fields.get("class_id", current.class_id))),
            cx=float(str(fields.get("cx", current.cx))),
            cy=float(str(fields.get("cy", current.cy))),
            w=float(str(fields.get("w", current.w))),
            h=float(str(fields.get("h", current.h))),
        )
        validate_box(merged, self.image(current.image_id).path)
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), _now(), box_id]
        self._conn.execute(f"UPDATE boxes SET {assignments}, updated_at = ? WHERE id = ?", values)
        self._conn.execute(
            "UPDATE images SET updated_at = ? WHERE id = ?", (_now(), current.image_id)
        )

    @_synchronized
    def delete_box(self, box_id: int) -> None:
        current = self.box(box_id)
        self._conn.execute("DELETE FROM boxes WHERE id = ?", (box_id,))
        self._conn.execute(
            "UPDATE images SET updated_at = ? WHERE id = ?", (_now(), current.image_id)
        )

    @_synchronized
    def boxes(self, image_id: int) -> list[BoxRow]:
        rows = self._conn.execute(
            "SELECT * FROM boxes WHERE image_id = ? ORDER BY id", (image_id,)
        ).fetchall()
        return [self._box_row(row) for row in rows]

    @_synchronized
    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(STATUSES, 0)
        for row in self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
        ).fetchall():
            counts[str(row["status"])] = int(row["n"])
        return counts

    # --- Läufe -------------------------------------------------------------
    @_synchronized
    def add_run(self, kind: str, params: dict[str, object]) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (kind, started_at, params_json) VALUES (?, ?, ?)",
            (kind, _now(), json.dumps(params, ensure_ascii=False, sort_keys=True)),
        )
        return int(cur.lastrowid or 0)

    @_synchronized
    def finish_run(
        self,
        run_id: int,
        metrics: dict[str, float],
        model_path: str | None,
        log_path: str,
    ) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at = ?, metrics_json = ?, model_path = ?, log_path = ?"
            " WHERE id = ?",
            (_now(), json.dumps(metrics, ensure_ascii=False), model_path, log_path, run_id),
        )

    @_synchronized
    def run(self, run_id: int) -> dict[str, object]:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Lauf {run_id} unbekannt")
        return dict(row)

    @_synchronized
    def latest_model(self) -> str | None:
        row = self._conn.execute(
            "SELECT model_path FROM runs WHERE model_path IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else str(row["model_path"])
