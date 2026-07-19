"""Acces partage aux evenements de minage d'une base CoreProtect SQLite.

Regles d'extraction identiques a scripts/extract_mining_events.py (version Postgres),
mais sans filtre minerai : on garde tous les blocs casses pour reconstituer les
trajectoires (la roche est le chemin, les minerais sont la cible) :
1. action = 0 (bloc casse) uniquement.
2. uuid non nul sur co_user (exclut les pseudo-causes environnementales : #lava, ...).
3. exclusion des blocs poses puis recasses (stockage compresse, decorations).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

EXTRACTION_SQL_BASE = """
    SELECT
        b.time,
        u.user AS pseudo,
        b.x,
        b.y,
        b.z,
        m.material,
        b.wid
    FROM co_block b
    JOIN co_user u ON u.id = b.user
    JOIN co_material_map m ON m.id = b.type
    WHERE b.action = 0
      AND u.uuid IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM co_block prior
          WHERE prior.wid = b.wid
            AND prior.x = b.x
            AND prior.y = b.y
            AND prior.z = b.z
            AND prior.action = 1
            AND prior.time < b.time
      )
"""

WORLD_NAMES_SQL = "SELECT id, world FROM co_world"

# Familles de minerais (libelles) ; les variantes deepslate/nether sont rabattues dessus.
ORE_FAMILIES: dict[str, str] = {
    "diamond": "Diamant",
    "emerald": "Emeraude",
    "gold": "Or",
    "redstone": "Redstone",
    "lapis": "Lapis",
    "copper": "Cuivre",
    "iron": "Fer",
    "coal": "Charbon",
    "quartz": "Quartz",
    "ancient_debris": "Debris antiques",
}


def ore_family(material: str) -> str | None:
    """Rabat un materiau sur sa famille de minerai, ou None si ce n'est pas un minerai."""
    name = material.removeprefix("minecraft:")
    if name == "ancient_debris":
        return "ancient_debris"
    if not name.endswith("_ore"):
        return None
    name = name.removesuffix("_ore")
    for prefix in ("deepslate_", "nether_"):
        name = name.removeprefix(prefix)
    return name if name in ORE_FAMILIES else None


def _build_extraction_sql(start_ts: int | None = None, end_ts: int | None = None) -> tuple[str, dict[str, int]]:
    conditions = []
    params: dict[str, int] = {}
    if start_ts is not None:
        conditions.append("b.time >= :start_ts")
        params["start_ts"] = start_ts
    if end_ts is not None:
        conditions.append("b.time <= :end_ts")
        params["end_ts"] = end_ts

    sql = EXTRACTION_SQL_BASE
    if conditions:
        sql += "\n      AND " + "\n      AND ".join(conditions)
    sql += "\n    ORDER BY u.user, b.time"
    return sql, params


def load_breaks(
    db_path: Path, start_ts: int | None = None, end_ts: int | None = None
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Charge les blocs casses par les vrais joueurs, avec la colonne `ore` (famille ou NaN)."""
    conn = sqlite3.connect(db_path)
    try:
        sql, params = _build_extraction_sql(start_ts=start_ts, end_ts=end_ts)
        df = pd.read_sql(sql, conn, params=params)
        worlds = dict(conn.execute(WORLD_NAMES_SQL).fetchall())
    finally:
        conn.close()
    df["ore"] = df["material"].map(ore_family)
    return df, worlds


# Pseudos inventes pour l'anonymisation (aucun vrai joueur). Attribution deterministe
# par ordre alphabetique des vrais pseudos ; au-dela de la liste : JoueurN.
ANON_NAMES = [
    "Silexis", "Cobaltin", "Grimval", "Ondelune", "Ferbrune", "Vertigan",
    "Rubisco", "Palissandre", "Quartzelle", "Brumaille", "Solastre", "Molvane",
    "Tessonier", "Viperine", "Ambrelin", "Corindon",
]


def anonymize_players(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remplace les pseudos reels par des pseudos inventes (mapping deterministe).

    Retourne (df anonymise, mapping reel -> invente). Le mapping ne doit servir
    qu'en console : il ne doit jamais etre embarque dans un livrable partage.
    """
    real_names = sorted(df["pseudo"].unique())
    mapping = {
        real: ANON_NAMES[i] if i < len(ANON_NAMES) else f"Joueur{i + 1}"
        for i, real in enumerate(real_names)
    }
    df = df.copy()
    df["pseudo"] = df["pseudo"].map(mapping)
    return df, mapping


def segment_sessions(
    df: pd.DataFrame, gap_seconds: int, min_blocks: int, gap_blocks: float = 30.0
) -> tuple[pd.DataFrame, int]:
    """Ajoute une colonne session_id par joueur+monde, coupee sur les trous > gap_seconds
    ou quand temps et distance spatiale convergent assez pour suggérer un vrai changement de zone.

    Retourne (sessions gardees, nombre de sessions ecartees car < min_blocks).
    """
    df = df.sort_values(["pseudo", "wid", "time"]).copy()
    grp = df.groupby(["pseudo", "wid"])
    time_gap = grp["time"].diff()
    spatial_gap = (
        grp["x"].diff().pow(2)
        + grp["y"].diff().pow(2)
        + grp["z"].diff().pow(2)
    ).pow(0.5)
    combined_gap = (time_gap / gap_seconds) + (spatial_gap / gap_blocks)
    new_session = (time_gap > gap_seconds) | (combined_gap >= 1.5)
    df["session_id"] = new_session.groupby([df["pseudo"], df["wid"]]).cumsum().astype(int)

    counts = df.groupby(["pseudo", "wid", "session_id"])["time"].transform("size")
    kept = df[counts >= min_blocks].copy()
    dropped = df.loc[counts < min_blocks, ["pseudo", "wid", "session_id"]].drop_duplicates()
    return kept, len(dropped)
