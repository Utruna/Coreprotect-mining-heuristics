"""Extraction V1 : isoler les evenements de minage reel depuis CoreProtect.

Applique les 4 regles validees empiriquement sur la base (voir README du projet) :
1. action = 0 (bloc casse) uniquement.
2. uuid non nul sur co_user (exclut les pseudo-causes environnementales : #piston, #lava, ...).
3. exclusion des blocs poses puis recasses (stockage compresse, decorations).
4. filtre sur une liste de materiaux de minerai validee manuellement (pas de LIKE approximatif).

Usage:
    python scripts/extract_mining_events.py                  # historique complet
    python scripts/extract_mining_events.py --sample-days 7   # fenetre de validation
    python scripts/extract_mining_events.py --force           # saute la confirmation materiaux
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mining_events_raw.parquet"

# Liste validee manuellement le 2026-07-13 a partir de :
#   SELECT DISTINCT material FROM co_material_map WHERE material LIKE '%ore%' OR material LIKE '%debris%';
# Exclus volontairement (faux positifs du LIKE, pas des minerais) :
#   minecraft:explorer_pottery_sherd, minecraft:heavy_core, minecraft:spore_blossom
CURATED_ORE_MATERIALS = sorted(
    {
        "minecraft:ancient_debris",
        "minecraft:coal_ore",
        "minecraft:copper_ore",
        "minecraft:deepslate_coal_ore",
        "minecraft:deepslate_copper_ore",
        "minecraft:deepslate_diamond_ore",
        "minecraft:deepslate_emerald_ore",
        "minecraft:deepslate_gold_ore",
        "minecraft:deepslate_iron_ore",
        "minecraft:deepslate_lapis_ore",
        "minecraft:deepslate_redstone_ore",
        "minecraft:diamond_ore",
        "minecraft:emerald_ore",
        "minecraft:gold_ore",
        "minecraft:iron_ore",
        "minecraft:lapis_ore",
        "minecraft:nether_gold_ore",
        "minecraft:nether_quartz_ore",
        "minecraft:redstone_ore",
    }
)

DISCOVER_ORE_MATERIALS_SQL = """
    SELECT DISTINCT material
    FROM co_material_map
    WHERE material LIKE '%ore%' OR material LIKE '%debris%'
    ORDER BY material;
"""

EXTRACTION_SQL = """
    SELECT
        b.time,
        u.user AS pseudo,
        u.uuid,
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
      AND m.material = ANY(%(ore_materials)s)
      {time_filter}
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


def get_connection():
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5433"),
            dbname=os.getenv("POSTGRES_DB", "coreprotect"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )
    except psycopg2.OperationalError as exc:
        print(
            "Impossible de se connecter a Postgres. "
            "Verifie que le conteneur Docker est demarre : `docker compose up -d`.\n"
            f"Detail : {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def discover_and_validate_materials(conn, force: bool) -> list[str]:
    discovered = pd.read_sql(DISCOVER_ORE_MATERIALS_SQL, conn)["material"].tolist()

    print("Materiaux detectes (LIKE '%ore%' OR LIKE '%debris%') :")
    for material in discovered:
        print(f"  - {material}")

    extra = sorted(set(discovered) - set(CURATED_ORE_MATERIALS))
    missing = sorted(set(CURATED_ORE_MATERIALS) - set(discovered))

    if extra or missing:
        print("\nATTENTION : la liste detectee differe de la liste validee (CURATED_ORE_MATERIALS).")
        if extra:
            print(f"  Nouveaux materiaux non valides manuellement : {extra}")
        if missing:
            print(f"  Materiaux valides absents de la base actuelle : {missing}")
        if not force:
            print(
                "\nRelis la liste, mets a jour CURATED_ORE_MATERIALS dans "
                "scripts/extract_mining_events.py si necessaire, puis relance avec --force."
            )
            raise SystemExit(1)
        print("--force actif : poursuite avec CURATED_ORE_MATERIALS malgre la divergence.")
    else:
        print("\nListe detectee identique a la liste validee. Poursuite de l'extraction.")

    return CURATED_ORE_MATERIALS


def run_extraction(conn, ore_materials: list[str], sample_days: int | None) -> pd.DataFrame:
    time_filter = ""
    params: dict = {"ore_materials": ore_materials}

    if sample_days is not None:
        # co_block fait ~93 Go via sqlite_fdw : un MIN(time) dessus scanne toute la table
        # (mesure : le pushdown ORDER BY/LIMIT echoue aussi, >60s sans resultat).
        # co_session couvre la meme periode et ne fait que quelques dizaines de milliers
        # de lignes : on l'utilise comme proxy bon marche pour borner la fenetre d'echantillon.
        min_time = pd.read_sql("SELECT MIN(time) AS min_time FROM co_session;", conn)["min_time"].iloc[0]
        window_end = min_time + sample_days * 86400
        time_filter = "AND b.time BETWEEN %(window_start)s AND %(window_end)s"
        params["window_start"] = min_time
        params["window_end"] = window_end
        print(
            f"\nMode --sample-days {sample_days} : fenetre "
            f"[{datetime.fromtimestamp(min_time, tz=timezone.utc)} -> "
            f"{datetime.fromtimestamp(window_end, tz=timezone.utc)}]"
        )

    query = EXTRACTION_SQL.format(time_filter=time_filter)
    print("\nExecution de la requete d'extraction (peut prendre du temps sur l'historique complet)...")
    return pd.read_sql(query, conn, params=params)


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    total_rows = len(df)
    distinct_players = df["uuid"].nunique()

    print("\n--- Resume de l'extraction ---")
    print(f"Lignes extraites       : {total_rows}")
    print(f"Joueurs distincts       : {distinct_players}")

    if total_rows:
        min_time = datetime.fromtimestamp(df["time"].min(), tz=timezone.utc)
        max_time = datetime.fromtimestamp(df["time"].max(), tz=timezone.utc)
        print(f"Periode couverte        : {min_time} -> {max_time}")
    else:
        print("Periode couverte        : aucune ligne, periode indeterminee")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Fichier Parquet         : {output_path} ({size_mb:.2f} Mo)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-days",
        type=int,
        default=None,
        help="Limite l'extraction aux N premiers jours de l'historique (mode de validation).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore une divergence entre la liste de materiaux detectee et la liste validee.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Chemin du fichier Parquet de sortie (defaut : {OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    conn = get_connection()
    try:
        ore_materials = discover_and_validate_materials(conn, force=args.force)
        df = run_extraction(conn, ore_materials, sample_days=args.sample_days)
    finally:
        conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print_summary(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
