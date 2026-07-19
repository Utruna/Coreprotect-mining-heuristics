"""Genere les figures pedagogiques de readmeArbre.md (foret d'isolation).

Quatre PNG dans reports/figures/ :
- anomaly_isolation_concept.png : pourquoi un point atypique s'isole en peu de
  coupes aleatoires (donnees synthetiques 2D) ;
- anomaly_clipping.png : l'ecretage directionnel sur mean_blocks_between_veins
  (distribution reelle du corpus) ;
- anomaly_score_mapping.png : decision_function -> anomaly_score 0-100, ancres
  reelles du modele committe ;
- anomaly_scores_overview.png : distribution du score sur le corpus + verite
  terrain (chiffres de readmeAnalyse.md).

Usage : .venv\\Scripts\\python.exe scripts\\make_anomaly_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xray_detector.anomaly_model import (  # noqa: E402
    SUSPICIOUS_DIRECTION,
    load_model,
    score_anomalies,
)

FIGURES = ROOT / "reports" / "figures"
MODEL_PATH = ROOT / "data" / "models" / "anomaly_iforest_diamond.joblib"
CORPUS_CSV = ROOT / "data" / "processed" / "session_features_diamond_anon.csv"

# Palette (validee CVD, cf. skill dataviz) : les paires rouge/vert portent
# toujours une etiquette texte, jamais la couleur seule.
SURFACE = "#ffffff"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # corpus / normal
GREEN = "#008300"  # legitime
RED = "#e34948"    # anomalie / x-ray
ORANGE = "#eb6834"  # coupes de l'arbre (figure concept)

# Fond derriere les annotations posees sur des barres ou une courbe.
NOTE_BBOX = {"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "font.size": 9.5,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Figure 1 : le principe de l'isolation (synthetique)
# ---------------------------------------------------------------------------

def _random_cuts(points: np.ndarray, target: int, seed: int, max_cuts: int):
    """Coupes aleatoires successives dans la cellule contenant `target`.

    Reproduit le principe d'un arbre d'isolation : a chaque etape, une feature
    (axe) et un seuil tires au hasard dans l'etendue des points restants ; on
    garde le cote contenant la cible. Retourne les segments a tracer
    [(dim, seuil, cellule avant coupe)] et le nombre de points restants.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(points))
    cell = [
        points[:, 0].min() - 0.6, points[:, 0].max() + 0.6,
        points[:, 1].min() - 0.6, points[:, 1].max() + 0.6,
    ]
    cuts = []
    while len(idx) > 1 and len(cuts) < max_cuts:
        dim = int(rng.integers(2))
        lo, hi = points[idx, dim].min(), points[idx, dim].max()
        if hi - lo < 1e-9:
            break
        thr = float(rng.uniform(lo, hi))
        cuts.append((dim, thr, list(cell)))
        keep_low = points[target, dim] <= thr
        mask = (points[idx, dim] <= thr) if keep_low else (points[idx, dim] > thr)
        idx = idx[mask]
        if dim == 0:
            cell[1 if keep_low else 0] = thr
        else:
            cell[3 if keep_low else 2] = thr
    return cuts, len(idx)


def fig_isolation_concept() -> None:
    rng = np.random.default_rng(4)
    normal = rng.normal(0.0, 1.0, size=(60, 2))
    points = np.vstack([normal, [3.7, 3.3]])
    anom_idx = len(points) - 1
    center_idx = int(np.argmin(np.linalg.norm(normal - normal.mean(0), axis=1)))

    # Seeds deterministes : premiere graine qui isole l'anomalie en <= 3 coupes,
    # premiere qui laisse le point central non isole apres 8.
    seed_a = next(
        s for s in range(500)
        if (c := _random_cuts(points, anom_idx, s, 12))[1] == 1 and len(c[0]) <= 3
    )
    seed_n = next(
        s for s in range(500) if _random_cuts(points, center_idx, s, 8)[1] > 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharex=True, sharey=True)
    for ax, target, seed, max_cuts, title in (
        (axes[0], anom_idx, seed_a, 12, "Session atypique : isolée en "),
        (axes[1], center_idx, seed_n, 8, "Session typique : toujours pas isolée après "),
    ):
        cuts, remaining = _random_cuts(points, target, seed, max_cuts)
        ax.scatter(points[:, 0], points[:, 1], s=30, color=BLUE, alpha=0.85,
                   linewidths=0, zorder=3)
        tx, ty = points[target]
        col = RED if target == anom_idx else GREEN
        ax.scatter([tx], [ty], s=100, color=col, zorder=4,
                   edgecolors=SURFACE, linewidths=1.5)
        for k, (dim, thr, cell) in enumerate(cuts, start=1):
            if dim == 0:
                ax.plot([thr, thr], cell[2:], color=ORANGE, lw=1.5, zorder=2)
                ax.annotate(str(k), (thr, cell[3]), textcoords="offset points",
                            xytext=(3, -11), color=ORANGE, fontsize=9,
                            fontweight="bold")
            else:
                ax.plot(cell[:2], [thr, thr], color=ORANGE, lw=1.5, zorder=2)
                ax.annotate(str(k), (cell[1], thr), textcoords="offset points",
                            xytext=(-11, 4), color=ORANGE, fontsize=9,
                            fontweight="bold")
        ax.set_title(title + f"{len(cuts)} coupes", fontsize=10.5)
        ax.annotate("x-ray ?" if target == anom_idx else "typique",
                    (tx, ty), textcoords="offset points", xytext=(8, 6),
                    color=col, fontsize=9, fontweight="bold")
        ax.set_xlabel("feature 1 (ex. target_per_100_dig)")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("feature 2 (ex. detour_factor)")
    fig.suptitle("Isoler un point par des coupes aléatoires : l'atypique tombe seul très vite",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "anomaly_isolation_concept.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 : ecretage directionnel (donnees reelles)
# ---------------------------------------------------------------------------

def fig_clipping(corpus: pd.DataFrame) -> None:
    feat = "mean_blocks_between_veins"
    values = corpus[feat].astype(float).dropna()
    med = values.median()
    assert SUSPICIOUS_DIRECTION[feat] == -1  # suspect quand c'est bas
    clipped = values.clip(upper=med)
    bins = np.histogram_bin_edges(values, bins=40)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True)
    axes[0].hist(values, bins=bins, color=BLUE, alpha=0.85)
    axes[0].set_title("Avant : la malchance (beaucoup de blocs entre deux filons) est « extrême » aussi",
                      fontsize=10.5)
    axes[1].hist(clipped, bins=bins, color=BLUE, alpha=0.85)
    axes[1].set_title("Après écrêtage : tout ce qui dépasse la médiane est ramené à la médiane",
                      fontsize=10.5)
    span = values.max() - med
    for ax in axes:
        ax.axvline(med, color=INK, lw=1.2, ls="--")
        ax.set_ylabel("sessions")
    axes[0].annotate(f"médiane = {med:.0f}", (med, axes[0].get_ylim()[1] * 0.9),
                     textcoords="offset points", xytext=(6, 0), color=INK, fontsize=9,
                     bbox=NOTE_BBOX)
    axes[0].annotate("côté « légitime »\n(malchance, quadrillage patient)",
                     (med + span * 0.45, axes[0].get_ylim()[1] * 0.55),
                     color=MUTED, fontsize=9, ha="center", bbox=NOTE_BBOX)
    axes[0].annotate("côté suspect :\ntrouver vite", (med * 0.45, axes[0].get_ylim()[1] * 0.55),
                     color=RED, fontsize=9, ha="center", bbox=NOTE_BBOX)
    axes[1].annotate("plus rien à isoler de ce côté →",
                     (med + span * 0.45, axes[1].get_ylim()[1] * 0.5),
                     color=MUTED, fontsize=9, ha="center", bbox=NOTE_BBOX)
    axes[1].set_xlabel(f"{feat} (corpus réel, {len(values)} sessions mesurées)")
    fig.suptitle("Écrêtage directionnel : l'extrême côté légitime ne compte plus comme anomalie",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "anomaly_clipping.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 : normalisation decision_function -> 0-100 (ancres reelles)
# ---------------------------------------------------------------------------

def fig_score_mapping(model, scored: pd.DataFrame) -> None:
    raw_min, raw_max = model.raw_min, model.raw_max
    xs = np.array([raw_min, 0.0, raw_max])
    ys = np.array([100.0, 50.0, 0.0])

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axvspan(raw_min * 1.05, 0, color=RED, alpha=0.06)
    ax.axvspan(0, raw_max * 1.05, color=BLUE, alpha=0.06)
    ax.plot(xs, ys, color=INK, lw=2)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.axhline(50, color=BASELINE, lw=1, ls="--")
    ax.scatter(scored["anomaly_raw"], scored["anomaly_score"], s=18, color=BLUE,
               alpha=0.5, linewidths=0, zorder=3, label="sessions du corpus")
    ax.annotate("decision_function = 0\n= seuil de contamination\n→ score 50",
                (0, 50), textcoords="offset points", xytext=(10, 26),
                color=INK, fontsize=9, bbox=NOTE_BBOX)
    ax.annotate(f"session la plus atypique\ndu corpus ({raw_min:.3f} → 100)",
                (raw_min, 100), textcoords="offset points", xytext=(6, -24),
                color=RED, fontsize=9, bbox=NOTE_BBOX)
    ax.annotate(f"session la plus « normale »\ndu corpus ({raw_max:.3f} → 0)",
                (raw_max, 0), textcoords="offset points", xytext=(-130, 14),
                color=MUTED, fontsize=9, bbox=NOTE_BBOX)
    ax.set_xlabel("anomaly_raw (decision_function sklearn — élevé = normal)")
    ax.set_ylabel("anomaly_score (0-100)")
    ax.set_title("Deux segments de droite, ancrés sur les extrêmes du corpus d'entraînement",
                 fontsize=10.5)
    fig.suptitle("Du score brut de la forêt au score 0-100", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "anomaly_score_mapping.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 : distribution corpus + verite terrain
# ---------------------------------------------------------------------------

# Chiffres verite terrain de readmeAnalyse.md (« Ce que ca donne ») : sessions de
# la base de test, jamais vues, scorees par le modele committe.
GROUND_TRUTH = [
    ("Joueur 1 — session A (x-ray simulé)", 66.1, RED),
    ("Joueur 1 — session B (x-ray simulé)", 60.9, RED),
    ("Joueur 2 (x-ray simulé)", 49.5, RED),
    ("Joueur 3 (strip-mining légitime)", 42.0, GREEN),
]


def fig_scores_overview(scored: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(8.6, 6.0), height_ratios=[1.15, 1.0]
    )
    ax = axes[0]
    ax.hist(scored["anomaly_score"], bins=40, color=BLUE, alpha=0.85)
    ax.axvline(50, color=INK, lw=1.2, ls="--")
    share = float((scored["anomaly_score"] >= 50).mean()) * 100
    ax.annotate(f"score ≥ 50 : {share:.0f} % du corpus\n(≈ contamination 5 %)",
                (50, ax.get_ylim()[1] * 0.8), textcoords="offset points",
                xytext=(8, 0), color=INK, fontsize=9)
    ax.set_title(f"Corpus d'entraînement ({len(scored)} sessions de la vraie base)",
                 fontsize=10.5)
    ax.set_ylabel("sessions")
    ax.set_xlim(0, 100)

    ax = axes[1]
    labels = [g[0] for g in GROUND_TRUTH][::-1]
    vals = [g[1] for g in GROUND_TRUTH][::-1]
    cols = [g[2] for g in GROUND_TRUTH][::-1]
    bars = ax.barh(labels, vals, color=cols, height=0.55, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.1f}", (val, bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points", xytext=(5, -3), fontsize=9, color=INK)
    ax.axvline(50, color=INK, lw=1.2, ls="--")
    ax.annotate("50", (50, ax.get_ylim()[1]), textcoords="offset points",
                xytext=(3, -12), color=INK, fontsize=9)
    ax.set_title("Vérité terrain (base de test, sessions jamais vues par le modèle)",
                 fontsize=10.5)
    ax.set_xlabel("anomaly_score")
    ax.set_xlim(0, 100)
    fig.suptitle("Ce que le score donne en pratique", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "anomaly_scores_overview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(CORPUS_CSV)
    model = load_model(MODEL_PATH)
    scored = score_anomalies(model, corpus)

    fig_isolation_concept()
    fig_clipping(corpus)
    fig_score_mapping(model, scored)
    fig_scores_overview(scored)
    for name in (
        "anomaly_isolation_concept.png",
        "anomaly_clipping.png",
        "anomaly_score_mapping.png",
        "anomaly_scores_overview.png",
    ):
        print(f"OK  {FIGURES / name}")


if __name__ == "__main__":
    main()
