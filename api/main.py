"""API du miroir XRayGateway.

Remplace complètement la version précédente de ce fichier (upload CSV vers une
API qui recevait les blocs poussés par un plugin séparé). Ce design est
abandonné : votre vrai plugin, `xray-gateway-plugin`, expose déjà un serveur
HTTP en lecture seule (/health, /blocks, /users, /materials, /worlds, auth
Bearer, CSV gzippé), et `gateway_client.py` sait déjà synchroniser un miroir
SQLite local de façon incrémentale (reprenable, idempotent, jamais un scan
complet répété). Cette API ne fait qu'orchestrer ce qui existe déjà :
synchroniser le miroir, puis lancer votre pipeline d'analyse réel dessus.

Aucune logique d'analyse n'est dupliquée ici : compute_session_features,
score_session, segment_sessions, les trois filtres (grotte, End, récolte de
surface) et score_anomalies viennent tels quels de xray_detector, copié dans
l'image Docker (voir Dockerfile).
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from xray_detector.anomaly_model import load_model, score_anomalies
from xray_detector.features import compute_session_features, score_session
from xray_detector.gateway_client import sync as gateway_sync
from xray_detector.mining import (
    ORE_FAMILIES,
    filter_cave_like_sessions,
    filter_end_world_sessions,
    filter_surface_gathering_sessions,
    load_breaks,
    parse_utc_datetime,
    segment_sessions,
)

# Le token n'est JAMAIS baké dans l'image : uniquement passé à l'exécution
# (docker run -e / docker-compose env_file), comme tout secret.
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")
if not GATEWAY_TOKEN:
    raise RuntimeError("GATEWAY_TOKEN manquant : renseignez api/.env avant de démarrer.")
# Doit vivre sur un volume PERSISTANT (voir docker-compose.yml) : sans ça, le
# miroir repart de zéro (resync complète depuis rowid 0) à chaque redémarrage
# du container, ce qui annule l'intérêt de la synchro incrémentale.
MIRROR_PATH = Path(os.environ.get("MIRROR_PATH", "/data/mirror.db"))
MODELS_DIR = Path(__file__).resolve().parent / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("xrayindexer.api")

_anomaly_models: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # GATEWAY_TOKEN est deja garanti non vide (raise a l'import). Seul GATEWAY_URL
    # peut manquer ici : /sync et /report le signaleront (HTTP 500) le cas echeant.
    if not GATEWAY_URL:
        logger.warning("GATEWAY_URL absent de l'environnement : /sync et /report echoueront.")
    MIRROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODELS_DIR.exists():
        for path in sorted(MODELS_DIR.glob("anomaly_iforest_*.joblib")):
            ore_key = path.stem.removeprefix("anomaly_iforest_")
            try:
                _anomaly_models[ore_key] = load_model(path)
                logger.info("Modele charge : %s (%s)", ore_key, path.name)
            except Exception as exc:  # noqa: BLE001 - on log et on continue sans ce modele
                logger.warning("Modele %s non charge : %s", path.name, exc)
    yield
    _anomaly_models.clear()


app = FastAPI(title="XRayGateway Analysis API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mirror_exists": MIRROR_PATH.exists(),
        "gateway_configured": bool(GATEWAY_URL and GATEWAY_TOKEN),
        "models_loaded": sorted(_anomaly_models.keys()),
    }


@app.post("/sync")
def trigger_sync(page_size: int = Query(50000, ge=1000)) -> dict:
    """Déclenche une synchro (reprenable) sans lancer d'analyse. Utile pour
    pré-chauffer le miroir avant le premier /report, ou en tâche planifiée."""
    if not GATEWAY_URL or not GATEWAY_TOKEN:
        raise HTTPException(500, "GATEWAY_URL / GATEWAY_TOKEN non configures sur ce container.")
    try:
        return gateway_sync(GATEWAY_URL, GATEWAY_TOKEN, MIRROR_PATH, page_size=page_size)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Synchronisation vers la passerelle echouee : {exc}") from exc


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@app.get("/report")
def report(
    ore: str = Query("diamond"),
    start: str | None = Query(None, description="ISO UTC, ex 2026-07-01"),
    end: str | None = Query(None, description="ISO UTC, ex 2026-07-08"),
    gap_seconds: int = Query(300, ge=1),
    min_blocks: int = Query(50, ge=1),
    sync_first: bool = Query(True, description="Synchronise le miroir avant d'analyser."),
) -> dict:
    if ore not in ORE_FAMILIES:
        raise HTTPException(
            400, f"Minerai inconnu : {ore}. Valeurs possibles : {sorted(ORE_FAMILIES)}"
        )

    if sync_first:
        if not GATEWAY_URL or not GATEWAY_TOKEN:
            raise HTTPException(500, "GATEWAY_URL / GATEWAY_TOKEN non configures sur ce container.")
        gateway_sync(GATEWAY_URL, GATEWAY_TOKEN, MIRROR_PATH)

    if not MIRROR_PATH.exists():
        raise HTTPException(422, "Aucun miroir local : appelez /sync (ou sync_first=true) d'abord.")

    start_ts = int(parse_utc_datetime(start).timestamp()) if start else None
    end_ts = int(parse_utc_datetime(end).timestamp()) if end else None

    df, worlds = load_breaks(MIRROR_PATH, start_ts=start_ts, end_ts=end_ts)
    if df.empty:
        return {"target": ore, "sessions": [], "dropped": {}}

    df, dropped_short = segment_sessions(df, gap_seconds=gap_seconds, min_blocks=min_blocks)
    df, dropped_cave = filter_cave_like_sessions(df)
    df, dropped_end = filter_end_world_sessions(df, worlds)
    df, dropped_surface = filter_surface_gathering_sessions(df)

    if df.empty:
        return {
            "target": ore,
            "sessions": [],
            "dropped": {"short": dropped_short, "cave": dropped_cave,
                        "end_world": dropped_end, "surface_gather": dropped_surface},
        }

    model = _anomaly_models.get(ore)
    rows = []
    for (pseudo, wid, session_id), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        feats = compute_session_features(seg, target=ore)
        result = {
            "pseudo": pseudo,
            "world": worlds.get(wid, f"monde {wid}"),
            "session_id": int(session_id),
            **feats,
            **score_session(feats, target=ore),
        }
        if model is not None:
            scored = score_anomalies(model, pd.DataFrame([feats])).iloc[0]
            result["anomaly_score"] = float(scored["anomaly_score"])
            result["anomaly_top_feature"] = str(scored["anomaly_top_feature"])
        rows.append({k: _json_safe(v) for k, v in result.items()})

    rows.sort(key=lambda r: (r.get("score") or -1), reverse=True)
    return {
        "target": ore,
        "sessions": rows,
        "dropped": {"short": dropped_short, "cave": dropped_cave,
                    "end_world": dropped_end, "surface_gather": dropped_surface},
        "anomaly_model_loaded": model is not None,
    }
