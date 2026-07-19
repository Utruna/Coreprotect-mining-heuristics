"""Features de trajectoire de minage et score de suspicion x-ray (V1 heuristique).

Une session est une suite de blocs casses (x, y, z, time, ore) triee par temps.
La trajectoire est reconstituee de bloc casse en bloc casse ; un pas de plus de
JUMP_DISTANCE blocs est un deplacement sans minage (marche en grotte, chute,
teleportation) et coupe la continuite directionnelle.

Trois familles de features, calculees pour un minerai cible (target, par defaut
le diamant -- gold, iron, copper... sont aussi valides) :
- forme du chemin : longueur des segments droits (horizontaux / verticaux),
  frequence des changements de direction ;
- rendement : minerais pour 100 blocs (toutes familles et cible), blocs mines
  entre deux filons de la cible ;
- intentionnalite : facteur de detour entre filons successifs (chemin mine /
  distance a vol d'oiseau) et taux de virages orientes vers le prochain filon
  decouvert -- un joueur legitime ne peut pas viser un filon qu'il ne voit pas.

Le score V1 est une combinaison ponderee de trois indicateurs bornes, calibree
sur la connaissance du jeu (voir SCORE_RAMPS et TARGET_RATE_RAMPS). Il est concu
pour etre remplace par un modele entraine des qu'on aura un corpus etiquete.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Pas de plus de JUMP_DISTANCE blocs : deplacement sans minage, coupe la continuite.
JUMP_DISTANCE = 4.0
# Pas de plus de TELEPORT_DISTANCE blocs : probable /tp, exclut la paire de filons du detour.
TELEPORT_DISTANCE = 30.0
# Deux diamants a distance de Chebyshev <= 2 appartiennent au meme filon.
VEIN_CHEBYSHEV = 2
# Paires de filons a moins de MIN_VEIN_SPACING blocs : detour instable, ignorees.
MIN_VEIN_SPACING = 3.0

VERTICAL_AXIS = 1  # colonnes (x, y, z) -> y

# Rampes lineaires bornees (bas -> 0, haut -> 1) des indicateurs du score V1.
# Calibrees sur le comportement attendu en jeu :
# - target_per_100 : rendement en minerai cible / 100 blocs, borne par famille
#   (TARGET_RATE_RAMPS) car un minerai commun se trouve bien plus souvent ;
# - detour_factor : 1.0 = tunnel parfaitement droit de filon en filon ; un joueur
#   legitime quadrille (>= 3 fois la distance a vol d'oiseau).
# - turn_toward_ore_rate : un virage aleatoire rapproche du prochain filon ~1 fois
#   sur 2 ; viser juste a chaque virage trahit une information invisible.
SCORE_RAMPS: dict[str, tuple[float, float, float]] = {
    # indicateur: (borne basse, borne haute, poids)
    "target_per_100": (0.8, 3.0, 0.4),  # bornes remplacees par TARGET_RATE_RAMPS
    "detour_factor": (3.0, 1.4, 0.3),  # bornes inversees : petit detour = suspect
    "turn_toward_ore_rate": (0.5, 0.85, 0.3),
}

# Bornes de rendement (bas, haut) par minerai cible : au-dela du haut, le rendement
# n'est plus explicable par la chance. Diamant calibre sur le strip-mining a Y-59
# (~0.3-0.8 / 100 blocs) ; les autres sont des estimations d'apres l'abondance
# relative en jeu, a recalibrer sur donnees reelles.
TARGET_RATE_RAMPS: dict[str, tuple[float, float]] = {
    "diamond": (0.8, 3.0),
    "emerald": (0.4, 1.5),
    "ancient_debris": (0.3, 1.2),
    "gold": (1.2, 4.0),
    "lapis": (1.2, 4.0),
    "redstone": (2.0, 6.0),
    "iron": (3.0, 9.0),
    "copper": (3.0, 9.0),
    "coal": (4.0, 12.0),
    "quartz": (4.0, 12.0),
}
DEFAULT_RATE_RAMP = (1.0, 4.0)

VERDICT_THRESHOLDS = [(60.0, "fortement suspect"), (30.0, "a surveiller"), (0.0, "RAS")]


def _steps(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deltas entre blocs consecutifs et leur norme euclidienne."""
    deltas = np.diff(pos, axis=0)
    return deltas, np.linalg.norm(deltas, axis=1)


def _directions(deltas: np.ndarray, dists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direction dominante (axe 0..2, signe -1/+1) de chaque pas, et validite.

    Un pas est invalide (ne porte pas de direction) s'il est nul ou si c'est un
    saut (> JUMP_DISTANCE).
    """
    valid = (dists > 0) & (dists <= JUMP_DISTANCE)
    axes = np.abs(deltas).argmax(axis=1)
    signs = np.sign(deltas[np.arange(len(deltas)), axes]).astype(int)
    return np.stack([axes, signs], axis=1), valid


def _runs(directions: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Segments droits : liste de (axe, longueur en pas). Coupes aux sauts."""
    runs: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None  # (axe, signe)
    length = 0
    for i in range(len(directions)):
        if not valid[i]:
            if current is not None:
                runs.append((current[0], length))
            current, length = None, 0
            continue
        d = (int(directions[i, 0]), int(directions[i, 1]))
        if d == current:
            length += 1
        else:
            if current is not None:
                runs.append((current[0], length))
            current, length = d, 1
    if current is not None:
        runs.append((current[0], length))
    return runs


def _direction_changes(directions: np.ndarray, valid: np.ndarray) -> list[int]:
    """Indices de pas ou la direction change (sans compter les reprises apres saut)."""
    changes = []
    prev: tuple[int, int] | None = None
    prev_was_valid = False
    for i in range(len(directions)):
        if not valid[i]:
            prev_was_valid = False
            continue
        d = (int(directions[i, 0]), int(directions[i, 1]))
        if prev_was_valid and d != prev:
            changes.append(i)
        prev, prev_was_valid = d, True
    return changes


def ore_veins(seg: pd.DataFrame, family: str = "diamond") -> list[dict]:
    """Regroupe les casses du minerai cible en filons (Chebyshev <= VEIN_CHEBYSHEV).

    Retourne, en ordre chronologique : first/last (indice de ligne dans la session,
    0-based) et centroid (np.ndarray de taille 3).
    """
    veins: list[dict] = []
    positions: list[np.ndarray] = []
    is_target = (seg["ore"] == family).to_numpy()
    coords = seg[["x", "y", "z"]].to_numpy(float)

    for i in np.flatnonzero(is_target):
        p = coords[i]
        if positions and np.max(np.abs(p - positions[-1])) <= VEIN_CHEBYSHEV:
            positions.append(p)
            veins[-1]["last"] = int(i)
        else:
            positions = [p]
            veins.append({"first": int(i), "last": int(i)})
        veins[-1].setdefault("blocks", []).append(p)

    for vein in veins:
        vein["centroid"] = np.mean(vein.pop("blocks"), axis=0)
    return veins


def compute_session_features(seg: pd.DataFrame, target: str = "diamond") -> dict[str, float]:
    """Calcule les features d'une session (DataFrame trie par temps) pour un minerai cible."""
    seg = seg.sort_values("time")
    pos = seg[["x", "y", "z"]].to_numpy(float)
    n_blocks = len(seg)
    duration_min = (seg["time"].iloc[-1] - seg["time"].iloc[0]) / 60 if n_blocks > 1 else 0.0

    deltas, dists = _steps(pos)
    directions, valid = _directions(deltas, dists)
    runs = _runs(directions, valid)
    changes = _direction_changes(directions, valid)
    n_valid = int(valid.sum())

    h_runs = [ln for axis, ln in runs if axis != VERTICAL_AXIS]
    v_runs = [ln for axis, ln in runs if axis == VERTICAL_AXIS]
    vertical_steps = int(((directions[:, 0] == VERTICAL_AXIS) & valid).sum())

    n_ores = int(seg["ore"].notna().sum())
    n_target = int((seg["ore"] == target).sum())
    veins = ore_veins(seg, target)

    # Blocs mines entre la fin d'un filon et le debut du suivant.
    between = [
        veins[k + 1]["first"] - veins[k]["last"] - 1
        for k in range(len(veins) - 1)
    ]

    # Facteur de detour : chemin parcouru / distance a vol d'oiseau entre filons.
    detours = []
    for k in range(len(veins) - 1):
        a, b = veins[k]["last"], veins[k + 1]["first"]
        leg = dists[a:b]
        straight = float(np.linalg.norm(pos[b] - pos[a]))
        if straight < MIN_VEIN_SPACING or (leg > TELEPORT_DISTANCE).any():
            continue
        detours.append(float(leg.sum()) / straight)

    # Taux de virages orientes vers le prochain filon (pas encore decouvert).
    toward, evaluated = 0, 0
    vein_firsts = [v["first"] for v in veins]
    for s in changes:
        nxt = next((v for f, v in zip(vein_firsts, veins) if f > s), None)
        if nxt is None:
            continue
        axis, sign = int(directions[s, 0]), int(directions[s, 1])
        to_vein = nxt["centroid"] - pos[s]
        evaluated += 1
        if sign * to_vein[axis] > 0:
            toward += 1

    per100 = 100.0 / n_blocks if n_blocks else math.nan
    return {
        "n_blocks": n_blocks,
        "duration_min": round(duration_min, 1),
        "blocks_per_min": round(n_blocks / duration_min, 1) if duration_min else math.nan,
        "ore_per_100": round(n_ores * per100, 2),
        "target_per_100": round(n_target * per100, 2),
        "n_target_veins": len(veins),
        "mean_blocks_between_veins": round(float(np.mean(between)), 1) if between else math.nan,
        "detour_factor": round(float(np.mean(detours)), 2) if detours else math.nan,
        "turn_toward_ore_rate": round(toward / evaluated, 3) if evaluated else math.nan,
        "changes_per_100": round(len(changes) * 100.0 / n_valid, 1) if n_valid else math.nan,
        "mean_run_h": round(float(np.mean(h_runs)), 1) if h_runs else math.nan,
        "mean_run_v": round(float(np.mean(v_runs)), 1) if v_runs else math.nan,
        "vertical_step_ratio": round(vertical_steps / n_valid, 3) if n_valid else math.nan,
    }


def _ramp(value: float, low: float, high: float) -> float:
    """Interpolation lineaire bornee entre low (0) et high (1) ; gere low > high."""
    if math.isnan(value):
        return math.nan
    span = high - low
    return min(max((value - low) / span, 0.0), 1.0)


def score_session(features: dict[str, float], target: str = "diamond") -> dict[str, float | str]:
    """Score de suspicion 0-100 (V1 heuristique) + contributions par indicateur."""
    total, weight_sum = 0.0, 0.0
    contributions: dict[str, float | str] = {}
    for name, (low, high, weight) in SCORE_RAMPS.items():
        if name == "target_per_100":
            low, high = TARGET_RATE_RAMPS.get(target, DEFAULT_RATE_RAMP)
        indicator = _ramp(features[name], low, high)
        contributions[f"ind_{name}"] = round(indicator, 3) if not math.isnan(indicator) else math.nan
        if not math.isnan(indicator):
            total += weight * indicator
            weight_sum += weight

    score = round(100.0 * total / weight_sum, 1) if weight_sum else math.nan
    contributions["score"] = score
    contributions["verdict"] = next(
        label for threshold, label in VERDICT_THRESHOLDS
        if not math.isnan(score) and score >= threshold
    ) if not math.isnan(score) else "indeterminable"
    return contributions
