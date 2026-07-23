"""Client de synchronisation du miroir local depuis la passerelle XRayGateway.

La passerelle (plugin Paper/Spigot) expose la base CoreProtect en lecture seule
via HTTP. Ce client reconstruit et maintient un miroir SQLite local
(mirror.db, schema CoreProtect sans les BLOBs) que le pipeline d'analyse lit
ensuite comme une base CoreProtect classique (`load_breaks(mirror.db)`).

Synchro incrementale : co_block est un journal append-only dont le rowid croit
avec le temps. Le client ne redemande que les lignes `rowid > dernier_vu`, donc
l'historique ne transite qu'une fois. Les trois tables de correspondance
(co_user, co_material_map, co_world) sont petites et rechargees entierement.

Aucune dependance hors stdlib : urllib pour le HTTP, gzip pour la decompression,
csv pour le parsing.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sqlite3
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Colonnes scalaires du miroir co_block (jamais les BLOBs meta/blockdata).
_BLOCK_COLUMNS = ["cp_rowid", "time", "user", "wid", "x", "y", "z", "type", "action"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS co_block (
    cp_rowid INTEGER PRIMARY KEY,
    time     INTEGER,
    user     INTEGER,
    wid      INTEGER,
    x        INTEGER,
    y        INTEGER,
    z        INTEGER,
    type     INTEGER,
    action   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_block_coords ON co_block(wid, x, y, z);
CREATE TABLE IF NOT EXISTS co_user (
    id   INTEGER PRIMARY KEY,
    uuid TEXT,
    user TEXT
);
CREATE TABLE IF NOT EXISTS co_material_map (
    id       INTEGER PRIMARY KEY,
    material TEXT
);
CREATE TABLE IF NOT EXISTS co_world (
    id    INTEGER PRIMARY KEY,
    world TEXT
);
"""


def open_mirror(mirror_path: Path) -> sqlite3.Connection:
    """Ouvre (et cree si besoin) le miroir local et son schema."""
    mirror_path = Path(mirror_path)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(mirror_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def current_cursor(conn: sqlite3.Connection) -> int:
    """Plus grand cp_rowid deja present dans le miroir (0 si vide)."""
    row = conn.execute("SELECT COALESCE(MAX(cp_rowid), 0) FROM co_block").fetchone()
    return int(row[0])


def _read_csv(text: str) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def apply_blocks(conn: sqlite3.Connection, csv_text: str) -> tuple[int, int]:
    """Insere un lot de co_block (CSV decompresse) dans le miroir.

    INSERT OR REPLACE par cp_rowid : la re-synchro est idempotente (un lot rejoue
    ne cree pas de doublon). Retourne (lignes appliquees, cp_rowid max du lot).
    """
    header, data = _read_csv(csv_text)
    if not data:
        return 0, 0

    idx = {name: i for i, name in enumerate(header)}
    missing = [c for c in _BLOCK_COLUMNS if c not in idx]
    if missing:
        raise ValueError(f"Colonnes manquantes dans la reponse /blocks : {missing}")

    payload = []
    max_rowid = 0
    for row in data:
        values = [int(row[idx[c]]) for c in _BLOCK_COLUMNS]
        payload.append(values)
        if values[0] > max_rowid:
            max_rowid = values[0]

    placeholders = ",".join("?" for _ in _BLOCK_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO co_block ({','.join(_BLOCK_COLUMNS)}) VALUES ({placeholders})",
        payload,
    )
    conn.commit()
    return len(payload), max_rowid


def replace_users(conn: sqlite3.Connection, csv_text: str) -> int:
    """Recharge co_user. uuid vide (cause environnementale : #lava...) -> NULL,
    pour que le filtre `uuid IS NOT NULL` de load_breaks fonctionne."""
    header, data = _read_csv(csv_text)
    idx = {name: i for i, name in enumerate(header)}
    payload = [
        (int(r[idx["id"]]), (r[idx["uuid"]] or None), r[idx["user"]])
        for r in data
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO co_user (id, uuid, user) VALUES (?, ?, ?)", payload
    )
    conn.commit()
    return len(payload)


def _replace_simple_map(
    conn: sqlite3.Connection, table: str, value_col: str, csv_text: str
) -> int:
    header, data = _read_csv(csv_text)
    idx = {name: i for i, name in enumerate(header)}
    payload = [(int(r[idx["id"]]), r[idx[value_col]]) for r in data]
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} (id, {value_col}) VALUES (?, ?)", payload
    )
    conn.commit()
    return len(payload)


def _get(url: str, path: str, token: str, params: dict | None = None,
         timeout: float = 120.0) -> bytes:
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    request = urllib.request.Request(
        url.rstrip("/") + path + query,
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def head_rowid(url: str, token: str) -> int:
    """Tete courante de co_block cote serveur (via /health), sans rien telecharger."""
    payload = json.loads(_get(url, "/health", token).decode("utf-8"))
    return int(payload.get("max_rowid", 0))


def sync(
    url: str,
    token: str,
    mirror_path: Path,
    page_size: int = 50000,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Synchronise le miroir local avec la passerelle. Reprenable et idempotent.

    Retourne un resume : lignes de blocs ajoutees, curseurs, tailles des maps.
    """
    conn = open_mirror(mirror_path)
    try:
        start_cursor = current_cursor(conn)
        cursor = start_cursor
        total_blocks = 0

        while True:
            csv_text = _get(
                url, "/blocks", token,
                {"since": cursor, "limit": page_size},
            ).decode("utf-8")
            rows, max_rowid = apply_blocks(conn, csv_text)
            if rows == 0:
                break
            total_blocks += rows
            # Garde-fou : le curseur doit progresser strictement, sinon on stoppe
            # pour ne jamais boucler indefiniment sur le meme lot.
            if max_rowid <= cursor:
                break
            cursor = max_rowid
            if progress is not None:
                progress(total_blocks, cursor)

        users = replace_users(conn, _get(url, "/users", token).decode("utf-8"))
        materials = _replace_simple_map(
            conn, "co_material_map", "material",
            _get(url, "/materials", token).decode("utf-8"),
        )
        worlds = _replace_simple_map(
            conn, "co_world", "world",
            _get(url, "/worlds", token).decode("utf-8"),
        )

        return {
            "blocks_added": total_blocks,
            "cursor_before": start_cursor,
            "cursor_after": cursor,
            "users": users,
            "materials": materials,
            "worlds": worlds,
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Synchronise un miroir local depuis la passerelle XRayGateway."
    )
    parser.add_argument("--url", required=True, help="URL de la passerelle, ex http://127.0.0.1:8787")
    parser.add_argument("--token", required=True, help="Jeton Bearer (gateway.token du plugin)")
    parser.add_argument("--mirror", required=True, type=Path, help="Chemin du miroir SQLite local")
    parser.add_argument("--page-size", type=int, default=50000, help="Lignes co_block par appel")
    args = parser.parse_args(argv)

    def show(total: int, cursor: int) -> None:
        print(f"  ... {total} blocs synchronises (rowid {cursor})")

    print(f"Synchronisation depuis {args.url} vers {args.mirror} ...")
    summary = sync(args.url, args.token, args.mirror, page_size=args.page_size, progress=show)
    print(
        "Termine : "
        f"{summary['blocks_added']} blocs ajoutes "
        f"(rowid {summary['cursor_before']} -> {summary['cursor_after']}), "
        f"{summary['users']} users, {summary['materials']} materials, "
        f"{summary['worlds']} worlds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
