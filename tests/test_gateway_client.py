"""Tests du client de synchro miroir (xray_detector.gateway_client).

Deux niveaux :
- unitaires : apply_blocks est idempotent, deduplique par cp_rowid, suit le
  curseur ; uuid vide -> NULL.
- integration : un miroir reconstruit a partir des memes lignes que la base de
  test donne un resultat load_breaks strictement identique a la base d'origine.
  C'est la garantie que le schema du miroir reste compatible avec le pipeline.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from xray_detector.gateway_client import (
    _replace_simple_map,
    apply_blocks,
    current_cursor,
    open_mirror,
    replace_users,
)
from xray_detector.mining import load_breaks

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "database_testserv.db"


def _csv(header: list[str], rows: list[tuple]) -> str:
    """Reproduit le CSV que la passerelle produirait (None -> champ vide)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    return buffer.getvalue()


def test_apply_blocks_dedup_and_cursor():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE co_block (cp_rowid INTEGER PRIMARY KEY, time INTEGER, user INTEGER,"
        " wid INTEGER, x INTEGER, y INTEGER, z INTEGER, type INTEGER, action INTEGER);"
    )
    header = ["cp_rowid", "time", "user", "wid", "x", "y", "z", "type", "action"]
    batch = _csv(header, [
        (1, 100, 5, 0, 10, 60, 20, 3, 0),
        (2, 101, 5, 0, 11, 60, 20, 3, 0),
    ])

    rows, max_rowid = apply_blocks(conn, batch)
    assert (rows, max_rowid) == (2, 2)
    assert current_cursor(conn) == 2

    # Rejouer le meme lot ne cree pas de doublon (INSERT OR REPLACE par cp_rowid).
    apply_blocks(conn, batch)
    assert conn.execute("SELECT COUNT(*) FROM co_block").fetchone()[0] == 2


def test_apply_blocks_empty_batch():
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE co_block (cp_rowid INTEGER PRIMARY KEY, time INTEGER,"
                        " user INTEGER, wid INTEGER, x INTEGER, y INTEGER, z INTEGER,"
                        " type INTEGER, action INTEGER);")
    header = ["cp_rowid", "time", "user", "wid", "x", "y", "z", "type", "action"]
    rows, max_rowid = apply_blocks(conn, _csv(header, []))
    assert (rows, max_rowid) == (0, 0)


def test_replace_users_empty_uuid_becomes_null():
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE co_user (id INTEGER PRIMARY KEY, uuid TEXT, user TEXT);")
    text = _csv(["id", "uuid", "user"], [
        (1, "abcd-uuid", "Utruna"),
        (2, None, "#lava"),  # cause environnementale : pas d'uuid
    ])
    assert replace_users(conn, text) == 2
    rows = dict(conn.execute("SELECT user, uuid FROM co_user").fetchall())
    assert rows["Utruna"] == "abcd-uuid"
    assert rows["#lava"] is None


@pytest.mark.skipif(not DB_PATH.exists(), reason="base de test absente")
def test_mirror_roundtrip_matches_source(tmp_path):
    """Miroir reconstruit depuis la base de test == base de test pour load_breaks."""
    src = sqlite3.connect(DB_PATH)

    block_rows = src.execute(
        "SELECT rowid, time, user, wid, x, y, z, type, action FROM co_block"
    ).fetchall()
    user_rows = src.execute("SELECT id, uuid, user FROM co_user").fetchall()
    material_rows = src.execute("SELECT id, material FROM co_material_map").fetchall()
    world_rows = src.execute("SELECT id, world FROM co_world").fetchall()
    src.close()

    mirror_path = tmp_path / "mirror.db"
    conn = open_mirror(mirror_path)
    block_header = ["cp_rowid", "time", "user", "wid", "x", "y", "z", "type", "action"]
    apply_blocks(conn, _csv(block_header, block_rows))
    replace_users(conn, _csv(["id", "uuid", "user"], user_rows))
    _replace_simple_map(conn, "co_material_map", "material",
                        _csv(["id", "material"], material_rows))
    _replace_simple_map(conn, "co_world", "world", _csv(["id", "world"], world_rows))
    conn.close()

    df_src, worlds_src = load_breaks(DB_PATH)
    df_mirror, worlds_mirror = load_breaks(mirror_path)

    assert worlds_src == worlds_mirror
    pd.testing.assert_frame_equal(
        df_src.reset_index(drop=True), df_mirror.reset_index(drop=True)
    )
