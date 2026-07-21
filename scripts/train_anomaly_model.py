"""Entrainement du modele de detection d'anomalies (Isolation Forest) sur des sessions.

Deux sources possibles pour le corpus d'entrainement :
- un CSV de features deja produit par scripts/analyze_mining_sessions.py
  (--from-csv, instantane -- c'est la voie normale pour la grosse base) ;
- une base CoreProtect SQLite (--db, avec fenetre --start/--end poussee dans le
  SQL), en rejouant le meme pipeline que l'analyse : segmentation, filtre des
  sessions de grotte, features.

Le modele est sauvegarde en joblib (defaut : data/models/anomaly_iforest_<ore>.joblib)
et scripts/analyze_mining_sessions.py le charge automatiquement pour ajouter la
colonne anomaly_score a cote du score heuristique V1.

Usage:
    python scripts/train_anomaly_model.py --from-csv data/processed/session_features_diamond_anon.csv
    python scripts/train_anomaly_model.py --db data/raw/CoreProtect/database.db --start 2026-06-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from xray_detector.anomaly_model import (
    DEFAULT_CONTAMINATION,
    train_anomaly_model,
    save_model,
    score_anomalies,
)
from xray_detector.features import compute_session_features, score_session
from xray_detector.mining import (
    ORE_FAMILIES,
    filter_cave_like_sessions,
    load_breaks,
    parse_utc_datetime,
    segment_sessions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "session_labels.csv"


def build_feature_table_from_db(args: argparse.Namespace) -> pd.DataFrame:
    """Rejoue le pipeline d'analyse (memes filtres que analyze_mining_sessions)."""
    start_ts = int(parse_utc_datetime(args.start).timestamp()) if args.start else None
    end_ts = int(parse_utc_datetime(args.end).timestamp()) if args.end else None
    df, worlds = load_breaks(args.db, start_ts=start_ts, end_ts=end_ts)
    if df.empty:
        raise SystemExit("Aucun bloc casse dans la fenetre demandee.")
    df, _ = segment_sessions(df, gap_seconds=args.gap, min_blocks=args.min_blocks)
    df, cave_dropped = filter_cave_like_sessions(df)
    if cave_dropped:
        print(f"Sessions exclues car ressemblant a des grottes/geodes : {cave_dropped}")
    iso = lambda ts: pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    rows = []
    for (pseudo, wid, _sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        features = compute_session_features(seg, target=args.ore)
        rows.append({"pseudo": pseudo, "world": worlds.get(wid, f"monde {wid}"),
                     "start_utc": iso(seg["time"].min()),
                     **features, **score_session(features, target=args.ore)})
    return pd.DataFrame(rows)


def exclude_labeled_sessions(
    table: pd.DataFrame, labels_path: Path, exclude: list[str]
) -> tuple[pd.DataFrame, int]:
    """Retire du corpus les sessions verifiees a la main (data/labels/).

    L'Isolation Forest apprend « la session typique » : laisser les sessions de
    triche confirmees dans le corpus lui apprendrait que le x-ray est typique.
    Appariement sur (pseudo, world, start_utc), la seule cle stable entre runs.
    """
    if not labels_path.exists():
        return table, 0
    labels = pd.read_csv(labels_path)
    bad = labels[labels["label"].isin(exclude)]
    if bad.empty:
        return table, 0
    keys = {"pseudo", "world", "start_utc"}
    if not keys <= set(table.columns):
        print(f"Attention : le corpus n'a pas les colonnes {sorted(keys)} — "
              f"exclusion des sessions etiquetees impossible (CSV trop ancien ? "
              f"regenerer avec analyze_mining_sessions.py).")
        return table, 0
    merged = table.merge(bad[sorted(keys)].drop_duplicates(), on=sorted(keys),
                         how="left", indicator=True)
    kept = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    return kept.reset_index(drop=True), len(table) - len(kept)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-csv", type=Path, default=None,
                        help="CSV de features produit par analyze_mining_sessions.py.")
    source.add_argument("--db", type=Path, default=None,
                        help="Base CoreProtect SQLite (pipeline complet rejoue).")
    parser.add_argument("--start", default=None, help="Debut de fenetre (ISO UTC, avec --db).")
    parser.add_argument("--end", default=None, help="Fin de fenetre incluse (ISO UTC, avec --db).")
    parser.add_argument("--gap", type=int, default=300,
                        help="Trou temporel en secondes qui coupe une session (defaut : 300).")
    parser.add_argument("--min-blocks", type=int, default=50,
                        help="Nombre minimal de blocs par session (defaut : 50).")
    parser.add_argument("--ore", default="diamond", choices=sorted(ORE_FAMILIES),
                        help="Minerai cible des features (defaut : diamond).")
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION,
                        help="Part de sessions supposees atypiques dans le corpus "
                             f"(hyperparametre documente, defaut : {DEFAULT_CONTAMINATION}).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Fichier joblib de sortie "
                             "(defaut : data/models/anomaly_iforest_<ore>.joblib).")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="CSV d'annotations manuelles dont les sessions triche/suspect "
                             f"sont exclues du corpus (defaut : {DEFAULT_LABELS} ; "
                             "ignore s'il n'existe pas).")
    parser.add_argument("--exclude-labels", default="triche,suspect",
                        help="Etiquettes a exclure du corpus, separees par des virgules "
                             "(defaut : triche,suspect).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Ne pas exclure les sessions etiquetees.")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = MODELS_DIR / f"anomaly_iforest_{args.ore}.joblib"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    if args.from_csv is not None:
        if not args.from_csv.exists():
            raise SystemExit(f"CSV introuvable : {args.from_csv}")
        table = pd.read_csv(args.from_csv)
        # Le CSV d'analyse peut deja porter les colonnes anomaly_* d'un modele
        # precedent : on les ecarte pour ne pas les dupliquer au re-scoring.
        table = table.drop(columns=[c for c in table.columns if c.startswith("anomaly_")])
        if "target" in table.columns and not (table["target"] == args.ore).all():
            raise SystemExit(
                f"Le CSV contient des features calculees pour "
                f"{sorted(table['target'].unique())}, pas pour --ore {args.ore}."
            )
        source_label = str(args.from_csv)
    else:
        if not args.db.exists():
            raise SystemExit(f"Base introuvable : {args.db}")
        table = build_feature_table_from_db(args)
        source_label = f"{args.db} [{args.start or 'debut'} -> {args.end or 'fin'}]"

    n_excluded = 0
    if not args.no_labels:
        exclude = [lab.strip() for lab in args.exclude_labels.split(",") if lab.strip()]
        table, n_excluded = exclude_labeled_sessions(table, args.labels, exclude)
        if n_excluded:
            print(f"Sessions etiquetees {exclude} exclues du corpus : {n_excluded} "
                  f"({args.labels})")

    print(f"Corpus d'entrainement : {len(table)} sessions ({source_label})")
    model = train_anomaly_model(table, target=args.ore, contamination=args.contamination)
    model.metadata = {"source": source_label, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "labeled_sessions_excluded": n_excluded}
    save_model(model, args.output)
    print(f"Modele ecrit : {args.output}")

    # Controle de coherence avec le score heuristique V1 sur le corpus lui-meme.
    scored = pd.concat([table, score_anomalies(model, table)], axis=1)
    if "score" in scored.columns:
        both = scored[["score", "anomaly_score"]].dropna()
        rho = both["score"].corr(both["anomaly_score"], method="spearman")
        print(f"Correlation de rang (Spearman) anomaly_score vs score V1 : {rho:.2f} "
              f"(sur {len(both)} sessions)")
    cols = [c for c in ("pseudo", "score", "verdict", "anomaly_score",
                        "anomaly_top_feature") if c in scored.columns]
    top = scored.sort_values("anomaly_score", ascending=False).head(10)
    print("\n--- 10 sessions les plus atypiques du corpus ---")
    print(top[cols].to_string(index=False))
    print(f"\nTemps total : {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
