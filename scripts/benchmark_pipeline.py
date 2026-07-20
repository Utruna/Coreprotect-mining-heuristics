"""Benchmark reproductible du pipeline d'analyse (temps, CPU, RAM) sur une fenetre fixe.

Rejoue les etapes de scripts/analyze_mining_sessions.py (extraction SQL,
sessionization, features + score, modele d'anomalie) sur une fenetre temporelle
FIXE de 30 jours, mesure chaque etape, puis ajoute une ligne par run dans
reports/benchmarks/benchmark_history.csv. La fenetre et la base ne changent pas
d'une mesure a l'autre : la charge de travail est identique, donc les ecarts
entre deux lignes de l'historique refletent l'evolution du code (commit stocke
avec chaque mesure), pas celle des donnees.

Metriques par run :
- temps mur par etape et total (time.perf_counter),
- temps CPU du process, user + system (psutil),
- pic de RAM (RSS echantillonne a 50 ms + peak working set Windows).

La figure reports/figures/benchmark_evolution.png est regeneree apres chaque
mesure et montre l'evolution sur l'ensemble de l'historique.

Usage:
    python scripts/benchmark_pipeline.py                      # base reelle, juin 2026
    python scripts/benchmark_pipeline.py --runs 3 --label "avant index time"
    python scripts/benchmark_pipeline.py --db data/raw/database_testserv.db \
        --start 2026-07-14 --end 2026-07-15                   # essai rapide
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xray_detector.anomaly_model import load_model, score_anomalies  # noqa: E402
from xray_detector.features import compute_session_features, score_session  # noqa: E402
from xray_detector.mining import (  # noqa: E402
    filter_cave_like_sessions,
    load_breaks,
    parse_utc_datetime,
    segment_sessions,
)

DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "CoreProtect" / "database.db"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
HISTORY_CSV = PROJECT_ROOT / "reports" / "benchmarks" / "benchmark_history.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "benchmark_evolution.png"

# Fenetre de reference : 30 jours pleins, fixes une fois pour toutes. Ne pas la
# deplacer entre deux mesures, sinon la charge change et l'historique ne compare
# plus le code mais les donnees.
DEFAULT_START = "2026-06-01"
DEFAULT_END = "2026-07-01"

STAGES = ["extraction", "sessionization", "features", "anomalie"]

CSV_COLUMNS = [
    "run_at_utc", "commit", "branch", "dirty", "label", "run_index", "runs_total",
    "db", "start", "end", "ore", "gap_s", "min_blocks",
    "n_blocks", "n_players", "n_sessions",
    "wall_extraction_s", "wall_sessionization_s", "wall_features_s", "wall_anomalie_s",
    "wall_total_s", "cpu_total_s", "peak_ram_mb",
    "python", "machine",
]

SURFACE = "#14171c"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2a2f38"
# Palette categorielle des etapes, validee (dataviz) sur le fond #14171c.
STAGE_COLORS = {
    "extraction": "#3987e5",
    "sessionization": "#199e70",
    "features": "#c98500",
    "anomalie": "#9085e9",
}
STAGE_LABELS = {
    "extraction": "Extraction SQL",
    "sessionization": "Sessionization",
    "features": "Features + score",
    "anomalie": "Modele d'anomalie",
}


class RssSampler:
    """Echantillonne le RSS du process dans un thread pour capturer le pic memoire."""

    def __init__(self, interval: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        info = self.process.memory_info()
        self.peak_rss = max(self.peak_rss, info.rss, getattr(info, "peak_wset", 0))


def git_context() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "yes" if run("status", "--porcelain") else "no",
    }


def run_pipeline_once(args: argparse.Namespace, start_ts: int, end_ts: int) -> dict[str, float]:
    """Un run complet du pipeline analytique, chronometre etape par etape."""
    process = psutil.Process()
    walls: dict[str, float] = {}
    cpu_before = process.cpu_times()

    with RssSampler() as sampler:
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        df, worlds = load_breaks(args.db, start_ts=start_ts, end_ts=end_ts)
        walls["extraction"] = time.perf_counter() - t0
        if df.empty:
            raise SystemExit("Aucun bloc casse dans la fenetre demandee.")
        n_blocks, n_players = len(df), df["pseudo"].nunique()

        t0 = time.perf_counter()
        df, _ = segment_sessions(df, gap_seconds=args.gap, min_blocks=args.min_blocks)
        df, _ = filter_cave_like_sessions(df)
        walls["sessionization"] = time.perf_counter() - t0
        if df.empty:
            raise SystemExit("Aucune session retenue avec ces seuils.")

        t0 = time.perf_counter()
        rows = []
        for (pseudo, wid, sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
            features = compute_session_features(seg, target=args.ore)
            rows.append({"pseudo": pseudo, "wid": wid, "session": sid, **features,
                         **score_session(features, target=args.ore)})
        import pandas as pd

        table = pd.DataFrame(rows)
        walls["features"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        model_path = MODELS_DIR / f"anomaly_iforest_{args.ore}.joblib"
        if model_path.exists():
            model = load_model(model_path)
            if model.target == args.ore:
                score_anomalies(model, table)
        walls["anomalie"] = time.perf_counter() - t0

        wall_total = time.perf_counter() - t_total

    cpu_after = process.cpu_times()
    cpu_total = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)

    return {
        **{f"wall_{stage}_s": round(walls[stage], 3) for stage in STAGES},
        "wall_total_s": round(wall_total, 3),
        "cpu_total_s": round(cpu_total, 3),
        "peak_ram_mb": round(sampler.peak_rss / 1024 / 1024, 1),
        "n_blocks": n_blocks,
        "n_players": n_players,
        "n_sessions": int(table.shape[0]),
    }


def append_history(rows: list[dict], history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def build_evolution_figure(
    history_path: Path, output: Path, workload: tuple[str, str, str] | None = None
) -> None:
    """Une barre par mesure (mediane des runs) : temps par etape, RAM, CPU.

    Seuls les runs de la meme charge (base + fenetre) sont traces : comparer des
    bases ou des fenetres differentes ne dirait rien de l'evolution du code. Par
    defaut la charge de reference est celle du dernier run de l'historique.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.read_csv(history_path)
    if history.empty:
        return
    if workload is None:
        last = history.iloc[-1]
        workload = (last["db"], last["start"], last["end"])
    db, start, end = workload
    mask = (history["db"] == db) & (history["start"] == start) & (history["end"] == end)
    excluded = int(len(history) - mask.sum())
    if excluded:
        print(f"Figure : {excluded} run(s) ecartes (autre base ou fenetre que "
              f"{db} {start} -> {end}) ; ils restent dans le CSV.")
    history = history[mask]
    if history.empty:
        return

    wall_cols = [f"wall_{stage}_s" for stage in STAGES]
    grouped = (
        history.groupby(["run_at_utc", "commit", "label"], dropna=False, sort=False)[
            wall_cols + ["wall_total_s", "cpu_total_s", "peak_ram_mb"]
        ]
        .median()
        .reset_index()
    )
    labels = [
        f"{str(row.run_at_utc)[5:10]}\n{row.commit}" for row in grouped.itertuples()
    ]
    xs = range(len(grouped))

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK_SECONDARY, "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    })
    fig = plt.figure(figsize=(12.5, 7.5), dpi=140)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1.5, 1],
                            hspace=0.5, wspace=0.24,
                            left=0.07, right=0.97, top=0.86, bottom=0.09)

    def style(ax) -> None:
        # Avec peu de mesures, borne l'axe pour eviter des barres demesurees.
        ax.set_xlim(-0.6, max(len(grouped) - 0.4, 5.6))
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0, labelsize=8)
        ax.margins(y=0.2)

    def stacked_panel(ax, stages: list[str], totals, title: str) -> None:
        bottom = [0.0] * len(grouped)
        for stage in stages:
            values = grouped[f"wall_{stage}_s"]
            ax.bar(xs, values, width=0.5, bottom=bottom, color=STAGE_COLORS[stage],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=STAGE_LABELS[stage])
            bottom = [b + v for b, v in zip(bottom, values)]
        for x, total in zip(xs, totals):
            ax.annotate(f"{total:.3g} s", (x, bottom[x]), ha="center", va="bottom",
                        fontsize=9, color=INK_PRIMARY, xytext=(0, 3),
                        textcoords="offset points")
        ax.set_xticks(list(xs), labels)
        ax.set_title(title, fontsize=10, color=INK_SECONDARY, loc="left", pad=8)
        style(ax)

    # Panneau 1 : temps total empile ; panneau 2 : zoom sur les etapes ecrasees
    # par l'extraction, a leur propre echelle (meme code couleur).
    stacked_panel(fig.add_subplot(grid[0, 0]), STAGES, grouped["wall_total_s"],
                  "Temps mur par etape (s, mediane des runs)")
    small_stages = STAGES[1:]
    small_totals = sum(grouped[f"wall_{stage}_s"] for stage in small_stages)
    stacked_panel(fig.add_subplot(grid[0, 1]), small_stages, small_totals,
                  "Zoom hors extraction (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[s]) for s in STAGES]
    fig.legend(handles, [STAGE_LABELS[s] for s in STAGES], loc="upper right",
               bbox_to_anchor=(0.97, 0.995), ncol=len(STAGES), frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY)

    # Panneaux 2 et 3 : pic RAM et temps CPU (une seule serie chacun).
    for col, slot, title, unit in [
        ("peak_ram_mb", grid[1, 0], "Pic de RAM (Mo)", "Mo"),
        ("cpu_total_s", grid[1, 1], "Temps CPU user+system (s)", "s"),
    ]:
        ax = fig.add_subplot(slot)
        values = grouped[col]
        ax.bar(xs, values, width=0.5, color="#3987e5", edgecolor=SURFACE,
               linewidth=1.5, zorder=3)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.3g}" if unit == "s" else f"{v:.0f}", (x, v),
                        ha="center", va="bottom", fontsize=9,
                        color=INK_PRIMARY, xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(list(xs), labels)
        ax.set_title(title, fontsize=10, color=INK_SECONDARY, loc="left", pad=8)
        style(ax)

    fig.suptitle("Evolution des performances du pipeline", x=0.07, y=0.97,
                 ha="left", fontsize=14, color=INK_PRIMARY, fontweight="bold")
    fig.text(0.07, 0.925,
             f"Charge identique a chaque mesure : {db}, fenetre {start} -> {end} - "
             "une barre par mesure (date + commit)",
             fontsize=10, color=INK_MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)
    print(f"Figure ecrite : {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"Base CoreProtect SQLite (defaut : {DEFAULT_DB}).")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Debut de la fenetre, ISO UTC (defaut : {DEFAULT_START}).")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"Fin de la fenetre, ISO UTC (defaut : {DEFAULT_END}).")
    parser.add_argument("--ore", default="diamond", help="Minerai surveille (defaut : diamond).")
    parser.add_argument("--gap", type=int, default=300,
                        help="Trou temporel en secondes qui coupe une session (defaut : 300).")
    parser.add_argument("--min-blocks", type=int, default=50,
                        help="Nombre minimal de blocs par session (defaut : 50).")
    parser.add_argument("--runs", type=int, default=1,
                        help="Nombre de runs mesures (defaut : 1 ; la figure prend la mediane).")
    parser.add_argument("--label", default="",
                        help="Note libre stockee avec la mesure (ex : 'avant index time').")
    parser.add_argument("--history", type=Path, default=HISTORY_CSV,
                        help=f"CSV d'historique (defaut : {HISTORY_CSV}).")
    parser.add_argument("--no-figure", action="store_true",
                        help="Ne pas regenerer la figure d'evolution.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.db.exists():
        raise SystemExit(f"Base introuvable : {args.db}")

    start_ts = int(parse_utc_datetime(args.start).timestamp())
    end_ts = int(parse_utc_datetime(args.end).timestamp())
    if end_ts < start_ts:
        raise SystemExit("--end est anterieur a --start.")

    context = git_context()
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Benchmark : {args.db.name}, fenetre {args.start} -> {args.end}, "
          f"{args.runs} run(s), commit {context['commit'] or '?'}"
          f"{' (dirty)' if context['dirty'] == 'yes' else ''}")
    if context["dirty"] == "yes":
        print("Attention : arbre de travail modifie, la mesure ne correspond pas "
              "exactement au commit stocke.")

    rows = []
    for run_index in range(1, args.runs + 1):
        metrics = run_pipeline_once(args, start_ts, end_ts)
        print(f"  run {run_index}/{args.runs} : total {metrics['wall_total_s']:.1f} s "
              f"(extraction {metrics['wall_extraction_s']:.1f} s), "
              f"CPU {metrics['cpu_total_s']:.1f} s, pic RAM {metrics['peak_ram_mb']:.0f} Mo, "
              f"{metrics['n_blocks']} blocs / {metrics['n_sessions']} sessions")
        rows.append({
            "run_at_utc": run_at, **context, "label": args.label,
            "run_index": run_index, "runs_total": args.runs,
            "db": args.db.name, "start": args.start, "end": args.end, "ore": args.ore,
            "gap_s": args.gap, "min_blocks": args.min_blocks,
            **metrics,
            "python": platform.python_version(), "machine": platform.node(),
        })

    append_history(rows, args.history)
    print(f"Historique mis a jour : {args.history} "
          f"({sum(1 for _ in args.history.open()) - 1} runs au total)")

    if not args.no_figure:
        build_evolution_figure(args.history, FIGURE_PATH,
                               (args.db.name, args.start, args.end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
