"""Exit-Code 3: teilweise fehlgeschlagen (SPEC Abschnitt 7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from annotator.autolabel import run_auto
from annotator.cli import main
from annotator.db import Store
from annotator.decide import Detection
from annotator.importer import import_folder, sha256_file
from tests.conftest import make_image


def test_import_counts_unreadable_png(tmp_path: Path, store: Store) -> None:
    (tmp_path / "kaputt.png").write_bytes(b"")
    make_image(tmp_path / "gut.png")
    result = import_folder(store, tmp_path)
    assert (result.added, result.skipped, result.unreadable) == (1, 1, 1)


def test_cli_import_returns_three_on_broken_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "cmd.db"
    Store(db).close()
    ordner = tmp_path / "eingang"
    make_image(ordner / "gut.png")
    (ordner / "kaputt.png").write_bytes(b"")
    code = main(["--db", str(db), "import", str(ordner)])
    assert code == 3
    ausgabe = capsys.readouterr().out
    assert "kaputt" in ausgabe.lower() or "unlesbar" in ausgabe.lower()


def test_auto_continues_after_predictor_error(tmp_path: Path, store: Store) -> None:
    ids = []
    for nummer in range(3):
        path = make_image(tmp_path / "bilder" / f"f_{nummer:02d}.png")
        image_id = store.add_image(str(path), sha256_file(path), 64, 48)
        assert image_id is not None
        ids.append(image_id)

    def predictor(pfad: str) -> list[Detection]:
        if pfad.endswith("f_01.png"):
            raise RuntimeError("Modellfehler")
        return [Detection(1, 0.5, 0.5, 0.2, 0.2, 0.95)]

    result = run_auto(
        store,
        predictor,
        queue_path=tmp_path / "queue.json",
        log_path=tmp_path / "log.txt",
    )
    assert result.processed == 3
    assert result.failed == 1
    assert store.image(ids[1]).status == "needs_review"
    assert store.image(ids[0]).status == "auto_ok"


def test_cli_auto_returns_three_on_partial_failure(
    tmp_path: Path, store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    path = make_image(tmp_path / "bilder" / "a.png")
    image_id = store.add_image(str(path), sha256_file(path), 64, 48)
    assert image_id is not None
    modell = tmp_path / "modelle" / "best.pt"
    modell.parent.mkdir(parents=True)
    modell.write_bytes(b"kein echtes Modell")
    db = store.path
    store.close()
    code = main(["--db", str(db), "auto", "--model", str(modell), "--cwd", str(tmp_path)])
    assert code == 3
    assert "fehl" in capsys.readouterr().out.lower()
