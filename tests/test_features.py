"""Tests des features de trajectoire sur des chemins synthetiques."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from xray_detector.features import (
    compute_session_features,
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
    xray = {"target_per_100": 5.8, "detour_factor": 1.1, "turn_toward_ore_rate": 0.9}
    legit = {"target_per_100": 0.5, "detour_factor": 4.0, "turn_toward_ore_rate": 0.45}
    assert score_session(xray)["score"] > 90
    assert score_session(legit)["score"] < 10
    assert score_session(xray)["verdict"] == "fortement suspect"
    assert score_session(legit)["verdict"] == "RAS"


def test_score_uses_target_rate_ramp():
    feats = {"target_per_100": 5.0, "detour_factor": math.nan, "turn_toward_ore_rate": math.nan}
    # 5 fers / 100 blocs est banal, 5 diamants / 100 blocs est sature.
    assert score_session(feats, target="iron")["score"] < 50
    assert score_session(feats, target="diamond")["score"] == 100.0


def test_score_handles_missing_indicators():
    result = score_session(
        {"target_per_100": 3.5, "detour_factor": math.nan, "turn_toward_ore_rate": math.nan}
    )
    assert result["score"] == 100.0  # seul indicateur disponible, sature
