"""Kommandozeile: Exit-Codes, deutsche Meldungen, Host-Schutz (B5, F3, E1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotator.cli import main
from annotator.db import Store
from annotator.geometry import Box
from annotator.importer import sha256_file
from tests.conftest import make_image


def test_import_command(store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_image(tmp_path / "bilder" / "a.png")
    db = tmp_path / "cmd.db"
    code = main(["--db", str(db), "import", str(tmp_path / "bilder"), "--recursive"])
    assert code == 0
    out = capsys.readouterr().out
    assert "importiert" in out.lower()


def test_review_command_lists_queue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    s = Store(db)
    image_id = s.add_image(str(tmp_path / "a.png"), "h1", 10, 10)
    assert image_id is not None
    s.set_status(image_id, "needs_review", by="auto")
    s.close()
    assert main(["--db", str(db), "review"]) == 0
    out = capsys.readouterr().out
    assert "a.png" in out
    assert "1" in out


def test_review_command_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    Store(db).close()
    assert main(["--db", str(db), "review"]) == 0
    assert "keine" in capsys.readouterr().out.lower()


def test_f3_auto_without_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    Store(db).close()
    code = main(["--db", str(db), "auto", "--cwd", str(tmp_path)])
    assert code == 2
    assert "modell" in capsys.readouterr().out.lower()


def test_e1_train_without_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    Store(db).close()
    code = main(["--db", str(db), "train", "--cwd", str(tmp_path), "--epochs", "1"])
    assert code == 2
    assert "bilder" in capsys.readouterr().out.lower()


def test_b5_serve_rejects_public_host(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    Store(db).close()
    code = main(["--db", str(db), "serve", "--host", "0.0.0.0"])
    assert code == 1
    assert "127.0.0.1" in capsys.readouterr().out


def test_export_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    s = Store(db)
    s.add_class("Messer")
    path = make_image(tmp_path / "bilder" / "a.png")
    image_id = s.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    s.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    s.set_status(image_id, "labeled", by="human")
    s.close()
    code = main(["--db", str(db), "export", "--out", str(tmp_path / "ds"), "--format", "both"])
    assert code == 0
    out = capsys.readouterr().out
    assert (tmp_path / "ds" / "train" / "_annotations.coco.json").is_file()
    assert "export" in out.lower()


def test_export_without_force_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "cmd.db"
    Store(db).close()
    target = tmp_path / "ds"
    target.mkdir()
    code = main(["--db", str(db), "export", "--out", str(target)])
    assert code == 1
    assert "force" in capsys.readouterr().out.lower()


def test_unknown_command_exits() -> None:
    with pytest.raises(SystemExit):
        main(["gibtsnicht"])


def test_defaults_are_coco_and_scratch(tmp_path: Path) -> None:
    db = tmp_path / "cmd.db"
    s = Store(db)
    s.add_class("Messer")
    path = make_image(tmp_path / "bilder" / "a.png")
    image_id = s.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    s.add_box(image_id, Box(1, 0.5, 0.5, 0.2, 0.2))
    s.set_status(image_id, "labeled", by="human")
    s.close()
    assert main(["--db", str(db), "export", "--out", str(tmp_path / "ds")]) == 0
    assert (tmp_path / "ds" / "train" / "_annotations.coco.json").is_file()
    params = json.loads((tmp_path / "ds" / "export_meta.json").read_text(encoding="utf-8"))
    assert params["format"] == "coco"
