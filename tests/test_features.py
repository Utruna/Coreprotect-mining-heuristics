"""Tests des features de trajectoire sur des chemins synthetiques."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import numpy as np

from xray_detector.features import (
    compute_session_features,
    dig_block_mask,
    ore_veins,
    score_session,
)


def make_session(
    coords: list[tuple[int, int, int]],
    diamonds: set[int] = frozenset(),
    family: str = "diamond",
) -> pd.DataFrame:
    """Session synthetique : un bloc casse par seconde le long de `coords`.

    `diamonds` : indices de coords qui sont des casses du minerai `family`.
    """
    return pd.DataFrame(
        {
            "time": range(len(coords)),
            "x": [c[0] for c in coords],
            "y": [c[1] for c in coords],
            "z": [c[2] for c in coords],
            "material": [
                f"minecraft:{family}_ore" if i in diamonds else "minecraft:stone"
                for i in range(len(coords))
            ],
            "ore": [family if i in diamonds else None for i in range(len(coords))],
        }
    )


def east(n: int, start=(0, 0, 0)) -> list[tuple[int, int, int]]:
    x, y, z = start
    return [(x + i, y, z) for i in range(n)]


def test_straight_tunnel_has_no_direction_change():
    feats = compute_session_features(make_session(east(11)))
    assert feats["changes_per_100"] == 0.0
    assert feats["mean_run_h"] == 10.0
    assert feats["vertical_step_ratio"] == 0.0
    assert math.isnan(feats["mean_run_v"])


def test_l_turn_counts_one_change():
    coords = east(6) + [(5, 0, z) for z in range(1, 5)]  # 5 pas est, virage, 4 pas sud
    feats = compute_session_features(make_session(coords))
    assert feats["changes_per_100"] == pytest.approx(100 / 9, abs=0.1)
    assert feats["mean_run_h"] == pytest.approx((5 + 4) / 2)


def test_vertical_staircase():
    coords = [(0, -i, 0) for i in range(6)]
    feats = compute_session_features(make_session(coords))
    assert feats["vertical_step_ratio"] == 1.0
    assert feats["mean_run_v"] == 5.0


def test_jump_breaks_continuity():
    # Deux tunnels droits separes d'un saut de 20 blocs : aucun virage compte.
    coords = east(5) + east(5, start=(30, 0, 0))
    feats = compute_session_features(make_session(coords))
    assert feats["changes_per_100"] == 0.0
    assert feats["mean_run_h"] == 4.0  # deux runs de 4 pas


def test_ore_veins_grouping():
    coords = east(20)
    seg = make_session(coords, diamonds={0, 1, 12})  # blocs 0-1 adjacents, 12 isole
    veins = ore_veins(seg, "diamond")
    assert len(veins) == 2
    assert veins[0]["first"] == 0 and veins[0]["last"] == 1
    assert veins[1]["first"] == 12


def test_target_family_gold():
    seg = make_session(east(21), diamonds={0, 20}, family="gold")
    feats = compute_session_features(seg, target="gold")
    assert feats["n_target_veins"] == 2
    assert feats["detour_factor"] == pytest.approx(1.0)
    # La meme session analysee pour le diamant ne voit aucun filon.
    assert compute_session_features(seg, target="diamond")["n_target_veins"] == 0


def test_straight_dig_between_veins_has_detour_one():
    seg = make_session(east(21), diamonds={0, 20})
    feats = compute_session_features(seg)
    assert feats["detour_factor"] == pytest.approx(1.0)
    assert feats["mean_blocks_between_veins"] == 19.0


def test_turn_toward_hidden_vein():
    # Est, puis virage plein sud pile vers un filon 5 blocs plus loin.
    coords = east(6) + [(5, 0, z) for z in range(1, 6)]
    seg = make_session(coords, diamonds={len(coords) - 1})
    feats = compute_session_features(seg)
    assert feats["turn_toward_ore_rate"] == 1.0


def test_score_separates_profiles():
    xray = {"target_per_100_dig": 5.8, "detour_factor": 1.1, "turn_toward_ore_rate": 0.9}
    legit = {"target_per_100_dig": 0.5, "detour_factor": 4.0, "turn_toward_ore_rate": 0.45}
    assert score_session(xray)["score"] > 90
    assert score_session(legit)["score"] < 10
    assert score_session(xray)["verdict"] == "fortement suspect"
    assert score_session(legit)["verdict"] == "RAS"


def test_score_uses_target_rate_ramp():
    feats = {"target_per_100_dig": 5.0, "detour_factor": math.nan, "turn_toward_ore_rate": math.nan}
    # 5 fers / 100 blocs est banal, 5 diamants / 100 blocs est sature (mais plafonne :
    # le rendement seul ne suffit plus, voir test_score_caps_partial_evidence).
    assert score_session(feats, target="iron")["score"] < 50
    assert score_session(feats, target="diamond")["score"] == 59.9


def test_score_caps_partial_evidence():
    # Rendement sature mais seul indicateur calculable (poids 0.4 < MIN_WEIGHT_SUM) :
    # score plafonne sous "fortement suspect".
    result = score_session(
        {"target_per_100_dig": 3.5, "detour_factor": math.nan, "turn_toward_ore_rate": math.nan}
    )
    assert result["score"] == 59.9
    assert result["verdict"] == "a surveiller"
    assert result["evidence_weight"] == 0.4


def test_score_discards_indicators_without_evidence():
    # detour_factor sur une seule paire et turn_toward sur 2 virages : preuves
    # insuffisantes, seuls le rendement (plafonne) reste.
    feats = {
        "target_per_100_dig": 5.0,
        "detour_factor": 1.0,
        "n_detour_pairs": 1,
        "turn_toward_ore_rate": 1.0,
        "n_turns_evaluated": 2,
    }
    result = score_session(feats)
    assert math.isnan(result["ind_detour_factor"])
    assert math.isnan(result["ind_turn_toward_ore_rate"])
    assert result["score"] == 59.9


def test_dig_block_mask_contiguous_tunnel():
    dists = np.ones(10)  # tunnel continu, pas de 1 bloc
    assert dig_block_mask(dists).all()


def test_dig_block_mask_two_tunnels_with_jump():
    dists = np.array([1.0] * 5 + [20.0] + [1.0] * 5)
    mask = dig_block_mask(dists)
    assert mask.all()  # les deux tunnels sont des phases de creusage valides


def test_dig_block_mask_demotes_short_phases():
    # Paires de casses isolees separees par des marches : ramassage en grotte.
    dists = np.array([1.0, 8.0, 1.0, 8.0, 1.0])
    assert not dig_block_mask(dists).any()


def test_cave_picker_is_not_flagged():
    # Joueur en grotte : 15 ramassages de 2 blocs (pierre + diamant expose),
    # separes par des marches de 8 blocs. Aucune phase de creusage valide.
    coords, diamonds = [], set()
    for i in range(15):
        x = i * 10
        coords += [(x, 0, 0), (x + 1, 0, 0)]
        diamonds.add(len(coords) - 1)
    feats = compute_session_features(make_session(coords, diamonds))
    assert feats["n_dig_blocks"] == 0
    assert math.isnan(feats["target_per_100_dig"])
    assert feats["n_dig_veins"] == 0
    assert math.isnan(feats["detour_factor"])
    result = score_session(feats)
    assert result["verdict"] == "indeterminable"


def test_ore_after_cave_arrival_is_suspicious_but_arrival_ore_is_not():
    # Le saut de 19 blocs represente le deplacement dans une cavite, quel que
    # soit le temps ecoule. Le minerai a l'arrivee (index 2) est visible et neutre. Le
    # bloc 3 est de la pierre ; le minerai en 3e casse (index 4) est suspect.
    seg = make_session(
        [(0, 0, 0), (1, 0, 0), (20, 0, 0), (21, 0, 0), (22, 0, 0)],
        diamonds={2, 4},
    )
    seg.loc[2:, "time"] += 3_600

    feats = compute_session_features(seg)

    assert feats["n_cave_arrivals"] == 1
    assert feats["n_cave_followup_slots"] == 1
    assert feats["n_cave_followup_target"] == 1
    assert feats["cave_followup_target_rate"] == 0.5
    assert score_session(feats)["verdict"] == "a surveiller"


def test_ore_vein_after_cave_arrival_is_not_suspicious():
    # Les minerais consecutifs forment un filon visible : aucun ne doit etre
    # compte comme une cible suivie en grotte, meme en 2e ou 3e casse.
    seg = make_session(
        [(0, 0, 0), (1, 0, 0), (20, 0, 0), (21, 0, 0), (22, 0, 0)],
        diamonds={2, 3, 4},
    )

    feats = compute_session_features(seg)

    assert feats["n_cave_arrivals"] == 1
    assert feats["n_cave_followup_slots"] == 0
    assert feats["n_cave_followup_target"] == 0
    assert math.isnan(feats["cave_followup_target_rate"])


def test_cave_followup_suspicion_decreases_with_distance():
    # A nombre de casses identique, une cible a deux blocs du minerai d'arrivee
    # pese 2,5 fois plus dans le score qu'une cible situee a cinq blocs.
    close = make_session(
        [(0, 0, 0), (1, 0, 0), (20, 0, 0), (21, 0, 0), (22, 0, 0)],
        diamonds={2, 4},
    )
    far = make_session(
        [(0, 0, 0), (1, 0, 0), (20, 0, 0), (21, 0, 0), (22, 0, 0), (23, 0, 0), (24, 0, 0), (25, 0, 0)],
        diamonds={2, 7},
    )

    close_features = compute_session_features(close)
    far_features = compute_session_features(far)

    assert close_features["cave_followup_target_rate"] == 0.5
    assert far_features["cave_followup_target_rate"] == 0.2
    assert score_session(close_features)["score"] > score_session(far_features)["score"]


def test_corridor_session_is_capped():
    # Long couloir diagonal en escalier (zigzag bloc a bloc) avec des diamants
    # exposes ramasses au passage : aucune decision de navigation, les
    # indicateurs d'intentionnalite ne doivent pas compter.
    coords = []
    x = z = 0
    for i in range(120):
        coords.append((x, 0, z))
        if i % 2 == 0:
            x += 1
        else:
            z += 1
    diamonds = {20, 21, 60, 61, 100}
    feats = compute_session_features(make_session(coords, diamonds))
    assert feats["path_straightness"] > 0.65
    result = score_session(feats)
    assert math.isnan(result["ind_detour_factor"])
    assert math.isnan(result["ind_turn_toward_ore_rate"])
    assert result["score"] <= 59.9
    assert result["verdict"] != "fortement suspect"


def test_two_high_tunnel_is_a_corridor():
    # Tunnel droit de 2 de haut : le geste alterne bas/haut (zigzag bloc a bloc)
    # mais la galerie est une ligne droite -> rectitude macro elevee.
    coords = []
    for x in range(60):
        coords += [(x, 0, 0), (x, 1, 0)]
    diamonds = {30, 31, 90}
    feats = compute_session_features(make_session(coords, diamonds))
    assert feats["path_straightness"] > 0.65
    result = score_session(feats)
    assert math.isnan(result["ind_detour_factor"])
    assert math.isnan(result["ind_turn_toward_ore_rate"])
    assert result["score"] <= 59.9


def test_quadrillage_is_not_a_corridor():
    # Quadrillage : allers-retours paralleles, rectitude faible.
    coords = []
    for row in range(6):
        xs = range(20) if row % 2 == 0 else range(19, -1, -1)
        coords += [(xv, 0, row * 3) for xv in xs]
    feats = compute_session_features(make_session(coords))
    assert feats["path_straightness"] < 0.65


def test_strip_miner_crossing_cave_keeps_dig_features():
    # Tunnel de 40 blocs (2 filons), traversee de grotte (20 blocs de marche),
    # second tunnel de 40 blocs (1 filon).
    coords = east(40) + east(40, start=(60, 0, 0))
    diamonds = {10, 30, 60}
    feats = compute_session_features(make_session(coords, diamonds))
    assert feats["n_dig_blocks"] == 80
    assert feats["dig_ratio"] == 1.0
    assert feats["target_per_100_dig"] == pytest.approx(3 / 80 * 100, abs=0.01)
    assert feats["n_dig_veins"] == 3
    # La paire de filons qui traverse la grotte est exclue du detour.
    assert feats["n_detour_pairs"] == 1
    assert feats["detour_factor"] == pytest.approx(1.0)
