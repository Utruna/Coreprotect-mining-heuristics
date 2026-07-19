"""Features de trajectoire de minage et score de suspicion x-ray (V1 heuristique).

Une session est une suite de blocs casses (x, y, z, time, ore) triee par temps.
La trajectoire est reconstituee de bloc casse en bloc casse ; un pas de plus de
JUMP_DISTANCE blocs est un deplacement sans minage (marche en grotte, chute,
teleportation) et coupe la continuite directionnelle.

Chaque casse est de plus classee en phase de creusage ou de marche : en
strip-mining deux casses consecutives sont a ~1 bloc d'ecart (le joueur creuse
son chemin), alors qu'en grotte le joueur marche dans l'air entre les casses.
Le rendement du score n'est calcule que sur les blocs creuses, et un filon ne
compte pour l'intentionnalite que s'il a ete atteint en creusant : marcher vers
un minerai visible en grotte n'est pas une fuite d'information.

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
# Pas de creusage : <= DIG_STEP_DISTANCE blocs (couvre les diagonales sqrt(3) ~ 1.73
# des tunnels 2 de haut). Au-dela, le joueur s'est deplace sans casser son chemin.
DIG_STEP_DISTANCE = 2.0
# Une phase de creusage doit compter au moins DIG_PHASE_MIN_BLOCKS casses contigues :
# casser un minerai expose + un bloc adjacent en grotte ne constitue pas un creusage.
DIG_PHASE_MIN_BLOCKS = 4
# Un filon est "atteint en creusant" si les APPROACH_DIG_STEPS pas qui le precedent
# sont des pas de creusage.
APPROACH_DIG_STEPS = 3
# En dessous de MIN_DIG_BLOCKS_FOR_RATE blocs creuses, target_per_100_dig est
# statistiquement instable -> NaN.
MIN_DIG_BLOCKS_FOR_RATE = 30
# Rectitude macro : vol d'oiseau / chemin de la trajectoire simplifiee (un point
# d'ancrage tous les STRAIGHTNESS_STRIDE blocs), ce qui gomme le zigzag bloc a
# bloc du geste de minage (tunnel 2 de haut, escalier) pour ne garder que la
# forme de la galerie. Au-dela de CORRIDOR_STRAIGHTNESS, la session est un
# couloir : le joueur n'a fait aucun choix de navigation, detour_factor et
# turn_toward_ore_rate n'y portent aucun signal (couloir droit ou en escalier
# ~0.95+, diagonal x/z ~0.71 ; quadrillage et x-ray slaloment bien plus bas).
STRAIGHTNESS_STRIDE = 8
CORRIDOR_STRAIGHTNESS = 0.65
# Deux diamants a distance de Chebyshev <= 2 appartiennent au meme filon.
VEIN_CHEBYSHEV = 2
# Paires de filons a moins de MIN_VEIN_SPACING blocs : detour instable, ignorees.
MIN_VEIN_SPACING = 3.0

VERTICAL_AXIS = 1  # colonnes (x, y, z) -> y

# Rampes lineaires bornees (bas -> 0, haut -> 1) des indicateurs du score V1.
# Calibrees sur le comportement attendu en jeu :
# - target_per_100_dig : rendement en minerai cible / 100 blocs creuses, borne par famille
#   (TARGET_RATE_RAMPS) car un minerai commun se trouve bien plus souvent ;
# - detour_factor : 1.0 = tunnel parfaitement droit de filon en filon ; un joueur
#   legitime quadrille (>= 3 fois la distance a vol d'oiseau).
# - turn_toward_ore_rate : un virage aleatoire rapproche du prochain filon ~1 fois
#   sur 2 ; viser juste a chaque virage trahit une information invisible.
SCORE_RAMPS: dict[str, tuple[float, float, float]] = {
    # indicateur: (borne basse, borne haute, poids)
    "target_per_100_dig": (0.8, 3.0, 0.4),  # bornes remplacees par TARGET_RATE_RAMPS
    "detour_factor": (3.0, 1.4, 0.3),  # bornes inversees : petit detour = suspect
    "turn_toward_ore_rate": (0.5, 0.85, 0.3),
}

# Preuve minimale par indicateur : (colonne de comptage, minimum requis). Sous le
# minimum, l'indicateur est ecarte du score (trop peu d'evenements evalues).
MIN_DETOUR_PAIRS = 2
MIN_TURNS_EVALUATED = 5
EVIDENCE_REQUIREMENTS: dict[str, tuple[str, int]] = {
    "detour_factor": ("n_detour_pairs", MIN_DETOUR_PAIRS),
    "turn_toward_ore_rate": ("n_turns_evaluated", MIN_TURNS_EVALUATED),
}

# Sous MIN_WEIGHT_SUM de poids d'indicateurs calculables, le score est plafonne :
# le rendement seul (poids 0.4) ne peut jamais produire "fortement suspect".
MIN_WEIGHT_SUM = 0.6
PARTIAL_EVIDENCE_SCORE_CAP = 59.9

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


def dig_block_mask(dists: np.ndarray, min_phase: int = DIG_PHASE_MIN_BLOCKS) -> np.ndarray:
    """Masque booleen des blocs casses en phase de creusage.

    Un pas est un pas de creusage s'il fait au plus DIG_STEP_DISTANCE blocs ; un
    bloc est en creusage s'il borde au moins un pas de creusage. Les phases de
    moins de `min_phase` blocs contigus sont retrogradees en marche (un minerai
    pioche au passage dans une grotte n'est pas un creusage).
    """
    n_steps = len(dists)
    mask = np.zeros(n_steps + 1, dtype=bool)
    dig_step = dists <= DIG_STEP_DISTANCE

    # Une phase = une suite de pas de creusage consecutifs ; elle couvre les
    # blocs a ses deux extremites. Les phases trop courtes sont ignorees.
    start = None
    for i in range(n_steps + 1):
        in_dig = i < n_steps and dig_step[i]
        if in_dig and start is None:
            start = i
        elif not in_dig and start is not None:
            if i - start + 1 >= min_phase:
                mask[start:i + 1] = True
            start = None
    return mask


def _is_dig_reached(vein_first: int, dists: np.ndarray, dig_mask: np.ndarray) -> bool:
    """Vrai si le filon a ete atteint en creusant (approche + bloc en phase de creusage)."""
    if not dig_mask[vein_first]:
        return False
    start = max(0, vein_first - APPROACH_DIG_STEPS)
    return bool(np.all(dists[start:vein_first] <= DIG_STEP_DISTANCE))


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

    dig_mask = dig_block_mask(dists)
    n_dig_blocks = int(dig_mask.sum())
    n_steps = len(dists)
    walk_step_ratio = float((dists > JUMP_DISTANCE).sum()) / n_steps if n_steps else math.nan
    anchors = np.concatenate([pos[::STRAIGHTNESS_STRIDE], pos[-1:]])
    macro_path = float(np.linalg.norm(np.diff(anchors, axis=0), axis=1).sum())
    path_straightness = (
        float(np.linalg.norm(pos[-1] - pos[0])) / macro_path if macro_path > 0 else math.nan
    )

    h_runs = [ln for axis, ln in runs if axis != VERTICAL_AXIS]
    v_runs = [ln for axis, ln in runs if axis == VERTICAL_AXIS]
    vertical_steps = int(((directions[:, 0] == VERTICAL_AXIS) & valid).sum())

    n_ores = int(seg["ore"].notna().sum())
    n_target = int((seg["ore"] == target).sum())
    n_target_dig = int(((seg["ore"] == target).to_numpy() & dig_mask).sum())
    veins = ore_veins(seg, target)

    # Blocs mines entre la fin d'un filon et le debut du suivant.
    between = [
        veins[k + 1]["first"] - veins[k]["last"] - 1
        for k in range(len(veins) - 1)
    ]

    # Seuls les filons atteints en creusant portent le signal d'intentionnalite :
    # marcher vers un minerai visible en grotte n'est pas suspect.
    dig_veins = [v for v in veins if _is_dig_reached(v["first"], dists, dig_mask)]

    # Facteur de detour : chemin mine / distance a vol d'oiseau entre filons
    # successifs atteints en creusant. Une paire traversee par un pas de marche
    # (> JUMP_DISTANCE) est ignoree : la longueur du chemin n'y a pas de sens.
    detours = []
    for k in range(len(dig_veins) - 1):
        a, b = dig_veins[k]["last"], dig_veins[k + 1]["first"]
        leg = dists[a:b]
        straight = float(np.linalg.norm(pos[b] - pos[a]))
        if straight < MIN_VEIN_SPACING or (leg > JUMP_DISTANCE).any():
            continue
        detours.append(float(leg.sum()) / straight)

    # Taux de virages orientes vers le prochain filon creuse (pas encore decouvert).
    toward, evaluated = 0, 0
    vein_firsts = [v["first"] for v in dig_veins]
    for s in changes:
        nxt = next((v for f, v in zip(vein_firsts, dig_veins) if f > s), None)
        if nxt is None:
            continue
        axis, sign = int(directions[s, 0]), int(directions[s, 1])
        to_vein = nxt["centroid"] - pos[s]
        evaluated += 1
        if sign * to_vein[axis] > 0:
            toward += 1

    per100 = 100.0 / n_blocks if n_blocks else math.nan
    per100_dig = 100.0 / n_dig_blocks if n_dig_blocks >= MIN_DIG_BLOCKS_FOR_RATE else math.nan
    return {
        "n_blocks": n_blocks,
        "duration_min": round(duration_min, 1),
        "blocks_per_min": round(n_blocks / duration_min, 1) if duration_min else math.nan,
        "n_dig_blocks": n_dig_blocks,
        "dig_ratio": round(n_dig_blocks / n_blocks, 3) if n_blocks else math.nan,
        "walk_step_ratio": round(walk_step_ratio, 3) if not math.isnan(walk_step_ratio) else math.nan,
        "path_straightness": round(path_straightness, 3) if not math.isnan(path_straightness) else math.nan,
        "ore_per_100": round(n_ores * per100, 2),
        "target_per_100": round(n_target * per100, 2),
        "target_per_100_dig": round(n_target_dig * per100_dig, 2) if not math.isnan(per100_dig) else math.nan,
        "n_target_veins": len(veins),
        "n_dig_veins": len(dig_veins),
        "n_detour_pairs": len(detours),
        "n_turns_evaluated": evaluated,
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
    """Score de suspicion 0-100 (V1 heuristique) + contributions par indicateur.

    Un indicateur sans preuve suffisante (EVIDENCE_REQUIREMENTS) est ecarte, de
    meme que les indicateurs d'intentionnalite d'une session en couloir
    (path_straightness >= CORRIDOR_STRAIGHTNESS : creuser tout droit n'est pas
    viser). Si le poids total des indicateurs restants est sous MIN_WEIGHT_SUM,
    le score est plafonne a PARTIAL_EVIDENCE_SCORE_CAP (jamais "fortement
    suspect" sur un seul indicateur). `evidence_weight` expose le poids utilise.
    """
    total, weight_sum = 0.0, 0.0
    contributions: dict[str, float | str] = {}
    straightness = features.get("path_straightness", math.nan)
    is_corridor = not math.isnan(straightness) and straightness >= CORRIDOR_STRAIGHTNESS
    for name, (low, high, weight) in SCORE_RAMPS.items():
        if name == "target_per_100_dig":
            low, high = TARGET_RATE_RAMPS.get(target, DEFAULT_RATE_RAMP)
        value = features[name]
        evidence = EVIDENCE_REQUIREMENTS.get(name)
        if evidence is not None:
            count = features.get(evidence[0])
            if count is not None and count < evidence[1]:
                value = math.nan
            # Session en couloir : aucun choix de navigation, l'intentionnalite
            # ne porte aucun signal (les filons se trouvaient sur la ligne).
            if is_corridor:
                value = math.nan
        indicator = _ramp(value, low, high)
        contributions[f"ind_{name}"] = round(indicator, 3) if not math.isnan(indicator) else math.nan
        if not math.isnan(indicator):
            total += weight * indicator
            weight_sum += weight

    score = round(100.0 * total / weight_sum, 1) if weight_sum else math.nan
    if not math.isnan(score) and weight_sum < MIN_WEIGHT_SUM:
        score = min(score, PARTIAL_EVIDENCE_SCORE_CAP)
    contributions["score"] = score
    contributions["evidence_weight"] = round(weight_sum, 2)
    contributions["verdict"] = next(
        label for threshold, label in VERDICT_THRESHOLDS
        if not math.isnan(score) and score >= threshold
    ) if not math.isnan(score) else "indeterminable"
    return contributions
