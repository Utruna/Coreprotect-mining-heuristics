"""Analyse statistique des sessions de minage : features de trajectoire + score x-ray V1.

Charge les blocs casses d'une base CoreProtect SQLite, segmente en sessions de
minage (xray_detector.mining), calcule les features de trajectoire et le score
de suspicion heuristique (xray_detector.features), puis :
- affiche le tableau par session (trie par score decroissant),
- ecrit le tableau complet en CSV dans data/processed/,
- genere une figure comparative (bar panels par feature) dans reports/figures/.

Usage:
    python scripts/analyze_mining_sessions.py
    python scripts/analyze_mining_sessions.py --db data/raw/database_testserv.db --gap 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xray_detector.features import compute_session_features, score_session
from xray_detector.mining import (
    ORE_FAMILIES,
    anonymize_players,
    load_breaks,
    segment_sessions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "database_testserv.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Couleurs categoriellles par joueur (slots dark de la palette de reference dataviz,
# validees sur le fond #14171c : bande de luminance, chroma, daltonisme, contraste).
PLAYER_SLOTS = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767"]

# Statuts (reserves au verdict, jamais reutilises pour une serie).
VERDICT_COLORS = {
    "fortement suspect": "#d03b3b",
    "a surveiller": "#fab219",
    "RAS": "#0ca30c",
}

SURFACE = "#14171c"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2a2f38"

def feature_panels(target_label: str) -> list[tuple[str, str, float | None]]:
    # (colonne, titre du panneau, ligne de reference facultative)
    return [
        ("target_per_100", f"{target_label} / 100 blocs", None),
        ("detour_factor", "Facteur de detour entre filons\n(pointille : 1 = ligne droite)", 1.0),
        ("turn_toward_ore_rate", "Virages orientes vers le prochain\nfilon (pointille : 0.5 = hasard)", 0.5),
        ("mean_blocks_between_veins", "Blocs mines entre deux filons", None),
        ("changes_per_100", "Changements de direction\n/ 100 blocs", None),
        ("mean_run_h", "Longueur moyenne d'un segment\ndroit horizontal (blocs)", None),
    ]


def analyze(df: pd.DataFrame, worlds: dict[int, str], target: str) -> pd.DataFrame:
    rows = []
    for (pseudo, wid, sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        features = compute_session_features(seg, target=target)
        rows.append(
            {
                "pseudo": pseudo,
                "world": worlds.get(wid, f"monde {wid}"),
                "session": sid,
                "target": target,
                **features,
                **score_session(features, target=target),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def print_report(table: pd.DataFrame) -> None:
    display_cols = [
        "pseudo", "n_blocks", "duration_min", "target_per_100", "n_target_veins",
        "mean_blocks_between_veins", "detour_factor", "turn_toward_ore_rate",
        "changes_per_100", "mean_run_h", "mean_run_v", "score", "verdict",
    ]
    print("\n--- Features et score par session (trie par score decroissant) ---")
    print(table[display_cols].to_string(index=False))


def build_figure(table: pd.DataFrame, output: Path, target_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    players = sorted(table["pseudo"].unique())
    colors = {p: PLAYER_SLOTS[i % len(PLAYER_SLOTS)] for i, p in enumerate(players)}
    table = table.sort_values(["pseudo", "session"]).reset_index(drop=True)
    bar_colors = [colors[p] for p in table["pseudo"]]

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK_SECONDARY, "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    })

    fig = plt.figure(figsize=(12.5, 8.5), dpi=140)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.65], hspace=0.55, wspace=0.28,
                            left=0.06, right=0.97, top=0.86, bottom=0.06)

    xs = range(len(table))
    for idx, (col, title, refline) in enumerate(feature_panels(target_label)):
        ax = fig.add_subplot(grid[idx // 3, idx % 3])
        values = table[col].fillna(0.0)
        ax.bar(xs, values, width=0.55, color=bar_colors, zorder=3)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:g}", (x, v), ha="center", va="bottom", fontsize=9,
                        color=INK_PRIMARY, xytext=(0, 2), textcoords="offset points")
        if refline is not None:
            ax.axhline(refline, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=4)
        ax.set_title(title, fontsize=10, color=INK_SECONDARY, pad=8)
        ax.set_xticks([])
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.margins(y=0.18)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, length=0)

    # Panneau score : barres horizontales, couleur de statut selon le verdict.
    ax = fig.add_subplot(grid[2, :])
    ys = range(len(table))
    ax.barh(ys, table["score"], height=0.5,
            color=[VERDICT_COLORS[v] for v in table["verdict"]], zorder=3)
    for y, (score, verdict, pseudo) in enumerate(
        zip(table["score"], table["verdict"], table["pseudo"])
    ):
        ax.annotate(f"{score:g} - {verdict}", (score, y), va="center", fontsize=9,
                    color=INK_PRIMARY, xytext=(6, 0), textcoords="offset points")
    ax.set_yticks(list(ys), table["pseudo"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_title("Score de suspicion x-ray (heuristique V1, 0-100)",
                 fontsize=10, color=INK_SECONDARY, pad=8, loc="left")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0, labelsize=8)

    fig.suptitle("Features de trajectoire par session de minage", x=0.06, y=0.97,
                 ha="left", fontsize=14, color=INK_PRIMARY, fontweight="bold")
    fig.text(0.06, 0.925,
             f"Minerai surveille : {target_label.lower()} - une barre par session, "
             "couleur par joueur",
             fontsize=10, color=INK_MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in players]
    fig.legend(handles, players, loc="upper right", bbox_to_anchor=(0.97, 0.99),
               ncol=len(players), frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)
    print(f"Figure ecrite : {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"Base CoreProtect SQLite (defaut : {DEFAULT_DB}).")
    parser.add_argument("--gap", type=int, default=300,
                        help="Trou temporel en secondes qui coupe une session (defaut : 300).")
    parser.add_argument("--min-blocks", type=int, default=50,
                        help="Nombre minimal de blocs pour garder une session (defaut : 50).")
    parser.add_argument("--ore", default="diamond", choices=sorted(ORE_FAMILIES),
                        help="Minerai surveille pour les filons et le score (defaut : diamond).")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV de sortie (defaut : data/processed/session_features_<ore>.csv).")
    parser.add_argument("--figure", type=Path, default=None,
                        help="Figure PNG (defaut : reports/figures/session_features_<ore>.png).")
    parser.add_argument("--no-figure", action="store_true",
                        help="Ne pas generer la figure comparative.")
    parser.add_argument("--anonymize", action="store_true",
                        help="Remplace les pseudos par des pseudos inventes (partage public).")
    args = parser.parse_args(argv)
    suffix = "_anon" if args.anonymize else ""
    if args.output is None:
        args.output = PROCESSED_DIR / f"session_features_{args.ore}{suffix}.csv"
    if args.figure is None:
        args.figure = FIGURES_DIR / f"session_features_{args.ore}{suffix}.png"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.db.exists():
        raise SystemExit(f"Base introuvable : {args.db}")

    df, worlds = load_breaks(args.db)
    print(f"{len(df)} blocs casses par {df['pseudo'].nunique()} joueurs charges depuis {args.db.name}")

    if args.anonymize:
        df, mapping = anonymize_players(df)
        print("Anonymisation (mapping console uniquement, absent des sorties) :")
        for real, anon in mapping.items():
            print(f"  {real} -> {anon}")

    df, dropped = segment_sessions(df, gap_seconds=args.gap, min_blocks=args.min_blocks)
    if dropped:
        print(f"Sessions ignorees (< {args.min_blocks} blocs) : {dropped}")
    if df.empty:
        raise SystemExit("Aucune session retenue avec ces seuils.")

    target_label = ORE_FAMILIES[args.ore]
    print(f"Minerai surveille : {target_label.lower()} ({args.ore})")

    table = analyze(df, worlds, target=args.ore)
    print_report(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\nTableau complet ecrit : {args.output}")

    if not args.no_figure:
        build_figure(table, args.figure, target_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
