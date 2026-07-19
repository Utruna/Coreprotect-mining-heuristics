"""Detection d'anomalies non supervisee sur les features de session (Isolation Forest).

Complement du score heuristique V1 (features.score_session), pas un remplacement :
la ou le score V1 encode une connaissance du jeu (rampes calibrees a la main),
l'Isolation Forest apprend ce qu'est une session *typique* du corpus d'entrainement
et mesure a quel point une session s'en ecarte. Le seul a priori injecte est la
direction suspecte des features de rendement / intentionnalite (voir
SUSPICIOUS_DIRECTION) : sans lui, une session extreme cote "legitime" score
aussi haut qu'un x-ray. "Atypique" ne veut pas dire "tricheur" pour autant : le
modele est un detecteur d'ecart au corpus, a lire a cote du score V1, jamais seul.

Conventions partagees avec le pipeline existant :
- meme table de features que score_session() (sortie de compute_session_features),
  memes NaN (indicateur sans preuve suffisante -> NaN, jamais 0) ;
- entrainement sur des sessions ayant deja passe filter_cave_like_sessions, pour
  ne pas reapprendre les faux positifs de grotte deja regles en amont.

Strategie NaN (explicite, voir readmeAnalyse.md) : imputation par la mediane du
corpus d'entrainement. Un NaN signifie "pas assez de preuve pour calculer
l'indicateur" ; la mediane est la valeur neutre du corpus, donc une session
incomplete ne peut pas devenir anormale *a cause de ses trous* -- elle ne peut
l'etre que par les features effectivement mesurees.

Normalisation du score : `anomaly_score` 0-100, ancre sur la decision_function
de sklearn (elevee = normal, negative = anomalie au sens du seuil de
contamination). 50 correspond exactement a ce seuil : [0, train_max] -> [50, 0]
et [train_min, 0] -> [100, 50], borne aux extremes du corpus d'entrainement.
Un score >= 50 se lit donc "plus atypique que (1 - contamination) du corpus".

Explication : `anomaly_top_feature` est la feature dont le remplacement par la
mediane du corpus rapproche le plus la session de la normale (perturbation
une-feature-a-la-fois), avec son gain brut dans `anomaly_top_delta`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Features de forme / rendement / intentionnalite, invariantes a la taille de la
# session. Exclues volontairement : n_blocks, duration_min, blocks_per_min
# (taille et vitesse ne sont pas des signaux de x-ray, et l'outillage/haste les
# fait varier), les comptages n_* (correles a la longueur de session) et les
# colonnes du score V1 (score, ind_*, verdict : le modele doit rester independant
# de l'heuristique pour que la comparaison ait un sens).
ANOMALY_FEATURES: list[str] = [
    "target_per_100_dig",
    "target_per_100",  # garde en plus de la version creusage : couvre l'evasion par la marche
    "ore_per_100",
    "mean_blocks_between_veins",
    "detour_factor",
    "turn_toward_ore_rate",
    "changes_per_100",
    "mean_run_h",
    "mean_run_v",
    "vertical_step_ratio",
    "dig_ratio",
    "walk_step_ratio",
    "path_straightness",
]

# L'Isolation Forest est non-directionnel : une session extreme cote "legitime"
# (creuser 200 blocs entre deux filons, quadriller avec un enorme detour) serait
# aussi anormale qu'un x-ray -- observe sur la verite terrain, ou la longue
# session patiente du joueur legitime sortait plus atypique qu'un tricheur.
# Pour les features dont la direction suspecte est connue, on ecrete donc le cote
# legitime a la mediane du corpus d'entrainement : +1 = suspect quand c'est haut
# (les valeurs sous la mediane sont ramenees a la mediane), -1 = suspect quand
# c'est bas (idem au-dessus). Les features de forme, sans direction evidente,
# restent bilaterales.
SUSPICIOUS_DIRECTION: dict[str, int] = {
    "target_per_100_dig": +1,
    "target_per_100": +1,
    "ore_per_100": +1,
    "mean_blocks_between_veins": -1,  # trouver vite = suspect
    "detour_factor": -1,  # ligne droite de filon en filon = suspect
    "turn_toward_ore_rate": +1,
}

# En dessous, un corpus ne definit pas de "normale" : sur une poignee de sessions,
# l'Isolation Forest isole d'abord le point seul de son cote -- sur la base de
# test, ce serait le joueur legitime. On refuse d'entrainer plutot que de scorer
# a l'envers.
MIN_TRAIN_SESSIONS = 30

# Part de sessions supposees atypiques dans le corpus d'entrainement. Sans verite
# terrain, ce n'est PAS calibrable precisement : c'est un hyperparametre documente
# qui fixe ou tombe le "50" du score normalise (voir readmeAnalyse.md).
DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 0


@dataclass
class AnomalyModel:
    """Isolation Forest entraine + tout ce qu'il faut pour scorer ailleurs."""

    forest: IsolationForest
    feature_names: list[str]
    medians: pd.Series  # medianes du corpus d'entrainement (imputation + perturbation)
    raw_min: float  # decision_function la plus basse du corpus (ancre du 100)
    raw_max: float  # decision_function la plus haute du corpus (ancre du 0)
    target: str
    contamination: float
    n_train_sessions: int
    metadata: dict = field(default_factory=dict)


def _feature_matrix(table: pd.DataFrame, feature_names: list[str],
                    medians: pd.Series) -> np.ndarray:
    """Extrait et impute (mediane du corpus d'entrainement) la matrice de features."""
    missing = [c for c in feature_names if c not in table.columns]
    if missing:
        raise ValueError(f"Colonnes de features absentes de la table : {missing}")
    matrix = table[feature_names].astype(float).copy()
    matrix = matrix.fillna(medians)
    for name, direction in SUSPICIOUS_DIRECTION.items():
        med = medians[name]
        if direction > 0:
            matrix[name] = matrix[name].clip(lower=med)
        else:
            matrix[name] = matrix[name].clip(upper=med)
    return matrix.to_numpy()


def train_anomaly_model(
    table: pd.DataFrame,
    target: str = "diamond",
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> AnomalyModel:
    """Entraine un Isolation Forest sur une table de features de sessions.

    `table` est la sortie de analyze() / compute_session_features (une ligne par
    session, NaN inclus), deja filtree des sessions de grotte. Leve ValueError
    sous MIN_TRAIN_SESSIONS ou si une feature est entierement NaN (mediane
    indefinie -> imputation impossible).
    """
    if len(table) < MIN_TRAIN_SESSIONS:
        raise ValueError(
            f"{len(table)} sessions d'entrainement : il en faut au moins "
            f"{MIN_TRAIN_SESSIONS} pour definir une normale exploitable."
        )
    features = table[
        [c for c in ANOMALY_FEATURES if c in table.columns]
    ].astype(float)
    missing = [c for c in ANOMALY_FEATURES if c not in table.columns]
    if missing:
        raise ValueError(f"Colonnes de features absentes de la table : {missing}")
    medians = features.median()
    all_nan = medians[medians.isna()].index.tolist()
    if all_nan:
        raise ValueError(f"Features entierement NaN dans le corpus : {all_nan}")

    forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    matrix = _feature_matrix(table, ANOMALY_FEATURES, medians)
    forest.fit(matrix)
    raw = forest.decision_function(matrix)
    return AnomalyModel(
        forest=forest,
        feature_names=list(ANOMALY_FEATURES),
        medians=medians,
        raw_min=float(raw.min()),
        raw_max=float(raw.max()),
        target=target,
        contamination=contamination,
        n_train_sessions=len(table),
    )


def _normalize(raw: np.ndarray, model: AnomalyModel) -> np.ndarray:
    """decision_function -> score 0-100, 50 = seuil de contamination (raw = 0).

    Cote normal [0, raw_max] -> [50, 0], cote anomalie [raw_min, 0] -> [100, 50],
    borne aux extremes observes sur le corpus d'entrainement.
    """
    pos_span = max(model.raw_max, 1e-9)
    neg_span = max(-model.raw_min, 1e-9)
    score = np.where(
        raw >= 0,
        50.0 * (1.0 - raw / pos_span),
        50.0 + 50.0 * (-raw / neg_span),
    )
    return np.clip(score, 0.0, 100.0)


def score_anomalies(model: AnomalyModel, table: pd.DataFrame) -> pd.DataFrame:
    """Score d'anomalie par session, aligne sur l'index de `table`.

    Colonnes retournees :
    - anomaly_raw : decision_function sklearn (elevee = normal, < 0 = anomalie) ;
    - anomaly_score : normalisation 0-100 (50 = seuil de contamination) ;
    - anomaly_top_feature : feature dont le remplacement par la mediane du corpus
      rapproche le plus la session de la normale ("" si rien ne tire vers
      l'anomalie) ;
    - anomaly_top_delta : gain brut de decision_function de ce remplacement.
    """
    matrix = _feature_matrix(table, model.feature_names, model.medians)
    raw = model.forest.decision_function(matrix)

    # Perturbation une-feature-a-la-fois : pour chaque session, n_features copies
    # avec une feature ramenee a la mediane, scorees en un seul appel.
    n, d = matrix.shape
    perturbed = np.repeat(matrix, d, axis=0)
    med = model.medians[model.feature_names].to_numpy()
    idx = np.tile(np.arange(d), n)
    perturbed[np.arange(n * d), idx] = med[idx]
    deltas = model.forest.decision_function(perturbed).reshape(n, d) - raw[:, None]

    top_idx = deltas.argmax(axis=1)
    top_delta = deltas[np.arange(n), top_idx]
    top_feature = np.where(
        top_delta > 0,
        np.array(model.feature_names, dtype=object)[top_idx],
        "",
    )
    return pd.DataFrame(
        {
            "anomaly_raw": np.round(raw, 4),
            "anomaly_score": np.round(_normalize(raw, model), 1),
            "anomaly_top_feature": top_feature,
            "anomaly_top_delta": np.round(top_delta, 4),
        },
        index=table.index,
    )


def save_model(model: AnomalyModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> AnomalyModel:
    model = joblib.load(path)
    if not isinstance(model, AnomalyModel):
        raise TypeError(f"{path} ne contient pas un AnomalyModel.")
    return model
