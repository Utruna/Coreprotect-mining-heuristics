
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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_utc_datetime(value: str) -> datetime:
    """Parse une date ISO en UTC. Accepte un suffixe Z, un offset explicite,
    une date seule (2026-06-01) ou des champs non zero-padded (2026-6-1)."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    date_part, sep, time_part = raw.partition("T")
    pieces = date_part.split("-")
    if len(pieces) == 3 and all(piece.isdigit() for piece in pieces):
        year, month, day = pieces
        raw = f"{year}-{int(month):02d}-{int(day):02d}{sep}{time_part}"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

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

# Dimensions ou chaque famille de minerai apparait naturellement. Sert a ne pas
# scorer une session pour un minerai impossible dans son monde (un score diamant
# au Nether n'a aucun sens et gonfle les comptes). L'or existe dans les deux
# (nether_gold_ore est rabattu sur gold) ; l'End n'a aucun minerai.
ORE_DIMENSIONS: dict[str, frozenset[str]] = {
    "diamond": frozenset({"overworld"}),
    "emerald": frozenset({"overworld"}),
    "gold": frozenset({"overworld", "nether"}),
    "redstone": frozenset({"overworld"}),
    "lapis": frozenset({"overworld"}),
    "copper": frozenset({"overworld"}),
    "iron": frozenset({"overworld"}),
    "coal": frozenset({"overworld"}),
    "quartz": frozenset({"nether"}),
    "ancient_debris": frozenset({"nether"}),
}


def world_dimension(world_name: str) -> str:
    """Devine la dimension d'un monde a partir de son nom CoreProtect.

    Convention des noms usuels : "world", "world_nether", "world_the_end"
    (et leurs variantes prefixees par le nom du serveur).
    """
    name = world_name.lower()
    if "nether" in name:
        return "nether"
    if "the_end" in name or name.endswith("_end") or name == "end":
        return "end"
    return "overworld"


# Blocs typiques des cavernes / geodes / cavites naturelles, peu compatibles avec
# l'hypothese d'un strip-mining en galerie reguliere. On s'en sert pour ecarter
# les sessions qui ne sont pas dans le domaine d'analyse du score x-ray V1.
CAVE_SIGNATURE_MATERIALS = {
    "minecraft:calcite",
    "minecraft:smooth_basalt",
    "minecraft:moss_block",
    "minecraft:clay",
    "minecraft:sculk",
    "minecraft:rooted_dirt",
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:moss_carpet",
    "minecraft:glow_lichen",
    "minecraft:dripstone_block",
    "minecraft:pointed_dripstone",
    "minecraft:amethyst_block",
    "minecraft:budding_amethyst",
    "minecraft:amethyst_cluster",
    "minecraft:spore_blossom",
    "minecraft:cave_vines",
    "minecraft:cave_vines_plant",
    "minecraft:big_dripleaf",
    "minecraft:small_dripleaf",
    "minecraft:weeping_vines",
    "minecraft:twisting_vines",
}

# Seuils choisis pour filtrer les sessions qui ressemblent fortement a de la
# cavitation naturelle. C'est le ratio qui decide : un seuil absolu qui declenche
# seul ecarte a tort les longues sessions (10 blocs de mousse sur 8 000 casses ne
# font pas une grotte — verifie sur un x-rayeur confirme dont les deux plus
# grosses sessions etaient exclues a 0,3 % de signatures). Le seuil absolu ne
# sert plus que de plancher d'evidence pour les sessions minuscules.
CAVE_SIGNATURE_MIN_BLOCKS = 3
CAVE_SIGNATURE_MIN_RATIO = 0.015

# Signature « forme » d'une session de grotte, complementaire des materiaux :
# enormement de minerais visibles ET (de la marche OU un rythme lent sur une
# petite session). Un x-rayeur a aussi un rendement eleve, mais il creuse vite
# et marche peu : les deux criteres de droite l'ecartent. Seuils calibres sur
# les 54 sessions taguees « grotte » par la moderation (fev-avr 2026) :
# 44/54 detectees, zero session etiquetee triche ou suspect exclue.
CAVE_SHAPE_JUMP_DISTANCE = 4.0  # meme seuil de pas que features.JUMP_DISTANCE
CAVE_SHAPE_MIN_ORE_PER_100 = 12.0
CAVE_SHAPE_MIN_WALK_RATIO = 0.06
CAVE_SHAPE_MAX_BLOCKS_PER_MIN = 35.0
CAVE_SHAPE_MAX_BLOCKS = 300


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


def cave_signature_count(seg: pd.DataFrame) -> int:
    """Compte les blocs signatures d'un environnement naturel (grotte, geode, etc.)."""
    return int(seg["material"].isin(CAVE_SIGNATURE_MATERIALS).sum())


def is_cave_shaped_session(seg: pd.DataFrame) -> bool:
    """Detecte une session de grotte par sa forme (rendement + marche/rythme).

    Attrape les cavites sans materiaux signature (grottes de pierre nue) que la
    regle des materiaux rate. Sans les colonnes ore/x/y/z/time (tests, appels
    partiels), la regle est simplement inactive.
    """
    n = len(seg)
    if n < 2 or not {"ore", "x", "y", "z", "time"} <= set(seg.columns):
        return False
    if 100.0 * seg["ore"].notna().sum() / n < CAVE_SHAPE_MIN_ORE_PER_100:
        return False
    seg = seg.sort_values("time")
    dists = seg[["x", "y", "z"]].astype(float).diff().iloc[1:].pow(2).sum(axis=1).pow(0.5)
    if float((dists > CAVE_SHAPE_JUMP_DISTANCE).mean()) >= CAVE_SHAPE_MIN_WALK_RATIO:
        return True
    duration_min = (seg["time"].iloc[-1] - seg["time"].iloc[0]) / 60
    blocks_per_min = n / duration_min if duration_min else float("inf")
    return blocks_per_min <= CAVE_SHAPE_MAX_BLOCKS_PER_MIN and n <= CAVE_SHAPE_MAX_BLOCKS


def is_cave_like_session(
    seg: pd.DataFrame,
    min_signature_blocks: int = CAVE_SIGNATURE_MIN_BLOCKS,
    min_signature_ratio: float = CAVE_SIGNATURE_MIN_RATIO,
) -> bool:
    """Detecte les sessions qui sortent du cadre strip-mining de l'analyse x-ray."""
    if seg.empty:
        return False
    signature_blocks = cave_signature_count(seg)
    if signature_blocks >= max(min_signature_blocks, min_signature_ratio * len(seg)):
        return True
    return is_cave_shaped_session(seg)


def filter_cave_like_sessions(
    df: pd.DataFrame,
    min_signature_blocks: int = CAVE_SIGNATURE_MIN_BLOCKS,
    min_signature_ratio: float = CAVE_SIGNATURE_MIN_RATIO,
) -> tuple[pd.DataFrame, int]:
    """Ecarte les sessions qui ressemblent a des cavernes / geodes naturelles."""
    if df.empty:
        return df.copy(), 0

    keep = pd.Series(True, index=df.index)
    excluded = 0
    for _, seg in df.groupby(["pseudo", "wid", "session_id"], sort=False):
        if is_cave_like_session(
            seg,
            min_signature_blocks=min_signature_blocks,
            min_signature_ratio=min_signature_ratio,
        ):
            keep.loc[seg.index] = False
            excluded += 1

    return df.loc[keep].copy(), excluded


def filter_end_world_sessions(
    df: pd.DataFrame, worlds: dict[int, str]
) -> tuple[pd.DataFrame, int]:
    """Ecarte les sessions minees dans l'End : aucun minerai n'y apparait
    (casser de l'endstone n'est jamais du x-ray), il n'y a rien a scorer."""
    if df.empty:
        return df.copy(), 0
    end_wids = {wid for wid, name in worlds.items() if world_dimension(name) == "end"}
    if not end_wids:
        return df.copy(), 0
    mask = df["wid"].isin(end_wids)
    excluded = int(df.loc[mask].groupby(["pseudo", "wid", "session_id"]).ngroups)
    return df.loc[~mask].copy(), excluded


# Blocs de recolte de surface : bucheronnage (troncs, bois, feuilles) et
# sable / gres. Une session composee principalement de ces blocs n'est pas du
# minage : le score x-ray n'y a aucun sens, on l'ecarte de l'analyse.
SURFACE_GATHER_MIN_RATIO = 0.5


def surface_gather_count(seg: pd.DataFrame) -> int:
    """Compte les blocs de recolte de surface (bois, sable, gres)."""
    names = seg["material"].str.removeprefix("minecraft:")
    return int(
        (
            names.str.endswith(("_log", "_wood", "_leaves"))
            | names.isin(("sand", "red_sand"))
            | names.str.contains("sandstone", regex=False)
        ).sum()
    )


def is_surface_gathering_session(
    seg: pd.DataFrame, min_ratio: float = SURFACE_GATHER_MIN_RATIO
) -> bool:
    """Detecte une session dominee par la recolte de surface (pas du minage)."""
    if seg.empty:
        return False
    return surface_gather_count(seg) / len(seg) >= min_ratio


def filter_surface_gathering_sessions(
    df: pd.DataFrame, min_ratio: float = SURFACE_GATHER_MIN_RATIO
) -> tuple[pd.DataFrame, int]:
    """Ecarte les sessions dominees par la recolte de surface (bois, sable, gres)."""
    if df.empty:
        return df.copy(), 0

    keep = pd.Series(True, index=df.index)
    excluded = 0
    for _, seg in df.groupby(["pseudo", "wid", "session_id"], sort=False):
        if is_surface_gathering_session(seg, min_ratio=min_ratio):
            keep.loc[seg.index] = False
            excluded += 1

    return df.loc[keep].copy(), excluded


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
