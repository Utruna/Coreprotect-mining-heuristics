"""Test d'integration sur la base de verite terrain (database_testserv.db).

IxLikexYoou44 et acsterix simulent du x-ray, Utruna mine legitimement. Ce test
verrouille la separation des profils : toute evolution des features ou du score
qui la casse est detectee immediatement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from xray_detector.features import compute_session_features, score_session
from xray_detector.mining import (
    filter_cave_like_sessions,
    load_breaks,
    segment_sessions,
)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "database_testserv.db"

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="base de test absente")


@pytest.fixture(scope="module")
def best_scores() -> pd.Series:
    """Meilleur score de session par joueur (pipeline complet, cible diamant)."""
    df, _ = load_breaks(DB_PATH)
    df, _ = segment_sessions(df, gap_seconds=300, min_blocks=50)
    df, _ = filter_cave_like_sessions(df)
    rows = []
    for (pseudo, _wid, _sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        features = compute_session_features(seg, target="diamond")
        rows.append({"pseudo": pseudo, **score_session(features, target="diamond")})
    table = pd.DataFrame(rows)
    return table.groupby("pseudo")["score"].max()


def test_simulated_cheaters_stay_flagged(best_scores):
    assert best_scores["IxLikexYoou44"] >= 55
    assert best_scores["acsterix"] >= 40


def test_legit_player_stays_clear(best_scores):
    assert best_scores["Utruna"] < 30
