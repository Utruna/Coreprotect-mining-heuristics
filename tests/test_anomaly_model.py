"""Tests du modele de detection d'anomalies (Isolation Forest).

Deux niveaux :
- tests synthetiques (toujours executes) : mecanique du module sur un corpus
  legitime genere -- bornes du score, strategie NaN, direction du signal ;
- test d'integration sur la verite terrain (memes conventions que
  tests/test_ground_truth.py) : le modele entraine sur le corpus reel
  (data/models/anomaly_iforest_diamond.joblib, cree par
  scripts/train_anomaly_model.py) doit classer les sessions x-ray simulees de
  database_testserv.db au-dessus de la session legitime. Saute si la base de
  test ou le modele entraine sont absents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xray_detector.anomaly_model import (
    ANOMALY_FEATURES,
    MIN_TRAIN_SESSIONS,
    load_model,
    save_model,
    score_anomalies,
    train_anomaly_model,
)
from xray_detector.features import compute_session_features
from xray_detector.mining import (
    filter_cave_like_sessions,
    load_breaks,
    segment_sessions,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "raw" / "database_testserv.db"
MODEL_PATH = ROOT / "data" / "models" / "anomaly_iforest_diamond.joblib"


def legit_corpus(n: int = 80, seed: int = 7) -> pd.DataFrame:
    """Corpus synthetique de sessions de strip-mining legitimes (plages plausibles).

    Les plages reprennent les ordres de grandeur observes sur les sessions
    legitimes (readmeAnalyse.md) : rendement diamant sous ~2 / 100 blocs creuses,
    detour de quadrillage >= ~2, virages vers le filon proches du hasard.
    """
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "target_per_100_dig": rng.uniform(0.2, 2.0, n),
        "target_per_100": rng.uniform(0.2, 2.0, n),
        "ore_per_100": rng.uniform(6.0, 20.0, n),
        "mean_blocks_between_veins": rng.uniform(30.0, 120.0, n),
        "detour_factor": rng.uniform(2.0, 4.5, n),
        "turn_toward_ore_rate": rng.uniform(0.35, 0.65, n),
        "changes_per_100": rng.uniform(60.0, 95.0, n),
        "mean_run_h": rng.uniform(1.0, 2.0, n),
        "mean_run_v": rng.uniform(1.0, 1.6, n),
        "vertical_step_ratio": rng.uniform(0.2, 0.5, n),
        "dig_ratio": rng.uniform(0.85, 1.0, n),
        "walk_step_ratio": rng.uniform(0.0, 0.06, n),
        "path_straightness": rng.uniform(0.1, 0.6, n),
    })
    # Memes conventions de NaN que compute_session_features : quelques sessions
    # sans preuve suffisante pour le rendement ou l'intentionnalite.
    table.loc[table.index[:5], "target_per_100_dig"] = np.nan
    table.loc[table.index[5:10], ["detour_factor", "turn_toward_ore_rate"]] = np.nan
    return table


def xray_like_row() -> dict[str, float]:
    """Session au profil x-ray : rendement enorme, lignes droites vers les filons."""
    return {
        "target_per_100_dig": 6.0, "target_per_100": 6.0, "ore_per_100": 22.0,
        "mean_blocks_between_veins": 15.0, "detour_factor": 1.2,
        "turn_toward_ore_rate": 0.9, "changes_per_100": 85.0,
        "mean_run_h": 1.2, "mean_run_v": 1.1, "vertical_step_ratio": 0.35,
        "dig_ratio": 0.99, "walk_step_ratio": 0.0, "path_straightness": 0.3,
    }


@pytest.fixture(scope="module")
def model():
    return train_anomaly_model(legit_corpus(), target="diamond")


def test_refuses_tiny_corpus():
    with pytest.raises(ValueError):
        train_anomaly_model(legit_corpus(n=MIN_TRAIN_SESSIONS - 1))


def test_scores_bounded_and_columns_present(model):
    scored = score_anomalies(model, legit_corpus(seed=11))
    assert list(scored.columns) == [
        "anomaly_raw", "anomaly_score", "anomaly_top_feature", "anomaly_top_delta",
    ]
    assert scored["anomaly_score"].between(0.0, 100.0).all()


def test_xray_profile_more_anomalous_than_legit(model):
    corpus = legit_corpus(seed=11)
    legit = corpus.iloc[[0]]
    xray = pd.DataFrame([xray_like_row()])
    s_legit = score_anomalies(model, legit)["anomaly_score"].iloc[0]
    s_xray = score_anomalies(model, xray)["anomaly_score"].iloc[0]
    assert s_xray > 50.0  # au-dela du seuil de contamination
    assert s_xray >= s_legit + 30.0
    # L'explication pointe une feature de rendement ou d'intentionnalite.
    top = score_anomalies(model, xray)["anomaly_top_feature"].iloc[0]
    assert top in {"target_per_100_dig", "target_per_100", "ore_per_100",
                   "mean_blocks_between_veins", "detour_factor", "turn_toward_ore_rate"}


def test_nan_does_not_create_anomaly(model):
    """Une session mediane dont on retire des preuves (NaN) ne devient pas atypique."""
    median_row = pd.DataFrame([model.medians[ANOMALY_FEATURES].to_dict()])
    holed = median_row.copy()
    holed.loc[:, ["target_per_100_dig", "detour_factor", "turn_toward_ore_rate"]] = np.nan
    s_median = score_anomalies(model, median_row)["anomaly_score"].iloc[0]
    s_holed = score_anomalies(model, holed)["anomaly_score"].iloc[0]
    assert s_holed == pytest.approx(s_median, abs=1e-6)
    assert s_holed < 50.0


def test_extreme_legit_direction_not_flagged(model):
    """Tres malchanceux / tres quadrilleur = extreme cote legitime -> pas atypique.

    C'est le garde-fou directionnel (SUSPICIOUS_DIRECTION) : sans lui, la longue
    session patiente d'un joueur legitime score comme un x-ray.
    """
    row = pd.DataFrame([{
        **model.medians[ANOMALY_FEATURES].to_dict(),
        "target_per_100_dig": 0.0, "target_per_100": 0.0, "ore_per_100": 1.0,
        "mean_blocks_between_veins": 500.0, "detour_factor": 8.0,
        "turn_toward_ore_rate": 0.1,
    }])
    assert score_anomalies(model, row)["anomaly_score"].iloc[0] < 50.0


def test_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    table = legit_corpus(seed=3)
    pd.testing.assert_frame_equal(
        score_anomalies(model, table), score_anomalies(reloaded, table)
    )


# --- Verite terrain (database_testserv.db + modele entraine sur le corpus reel) ---

@pytest.fixture(scope="module")
def ground_truth_scores() -> pd.Series:
    """Meilleur anomaly_score par joueur (pipeline complet, modele reel)."""
    df, _ = load_breaks(DB_PATH)
    df, _ = segment_sessions(df, gap_seconds=300, min_blocks=50)
    df, _ = filter_cave_like_sessions(df)
    rows = []
    for (pseudo, _wid, _sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        rows.append({"pseudo": pseudo, **compute_session_features(seg, target="diamond")})
    table = pd.DataFrame(rows)
    trained = load_model(MODEL_PATH)
    table["anomaly_score"] = score_anomalies(trained, table)["anomaly_score"]
    return table.groupby("pseudo")["anomaly_score"].max()


@pytest.mark.skipif(not DB_PATH.exists(), reason="base de test absente")
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="modele entraine absent "
                    "(scripts/train_anomaly_model.py)")
def test_ground_truth_separation(ground_truth_scores):
    legit = ground_truth_scores["Utruna"]
    for cheater in ("IxLikexYoou44", "acsterix"):
        assert ground_truth_scores[cheater] > legit + 5.0
    assert ground_truth_scores["IxLikexYoou44"] >= 55.0
    assert legit < 50.0  # sous le seuil de contamination
