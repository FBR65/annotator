"""Demo-Hilfe: setzt Bilder eines Status zurück auf 'new', damit 'auto' sie erneut prüft.

Aufruf: .venv/bin/python tools/reset_to_new.py --db demo/annotator.db --status needs_review
"""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--status", default="needs_review")
    args = parser.parse_args()
    verbindung = sqlite3.connect(args.db, isolation_level=None)
    betroffen = verbindung.execute(
        "SELECT COUNT(*) FROM images WHERE status = ?", (args.status,)
    ).fetchone()[0]
    verbindung.execute("UPDATE images SET status = 'new' WHERE status = ?", (args.status,))
    verbindung.execute(
        "DELETE FROM boxes WHERE state = 'proposed' AND image_id IN"
        " (SELECT id FROM images WHERE status = 'new')"
    )
    print(f"{betroffen} Bild(er) von '{args.status}' auf 'new' zurückgesetzt.")
    verbindung.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
