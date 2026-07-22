"""Reconstruction 3D interactive des sessions de minage depuis une base CoreProtect SQLite.

Extrait les blocs casses par les vrais joueurs et segmente en sessions de minage
(via xray_detector.mining), calcule les features de trajectoire et le score de
suspicion V1 par session et par minerai cible (via xray_detector.features), puis
genere une page HTML autonome construite autour de Plotly :
- bandeau superieur : selection joueur/session, double curseur temporel avec champs
  d'heure editables, stats live de la fenetre affichee ;
- panneau d'analyse : minerai surveille, score en anneau avec verdict, jauges des
  trois indicateurs du score, tuiles de features, classement des sessions ;
- scene 3D plein ecran : roche en gris translucide, minerais colores par famille
  (cible mise en avant), trace chronologique, legende cliquable.

Usage:
    python scripts/render_mining_3d.py
    python scripts/render_mining_3d.py --db data/raw/database_testserv.db --gap 300
  python scripts/render_mining_3d.py --window last-12h
  python scripts/render_mining_3d.py --start 2026-07-18T00:00:00Z --end 2026-07-19T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from xray_detector.anomaly_model import load_model, score_anomalies
from xray_detector.features import compute_session_features, score_session
from xray_detector.mining import (
    ORE_DIMENSIONS,
    ORE_FAMILIES,
    anonymize_players,
    filter_cave_like_sessions,
    filter_end_world_sessions,
    filter_surface_gathering_sessions,
    load_breaks,
    parse_utc_datetime,
    segment_sessions,
    world_dimension,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "database_testserv.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "figures" / "mining_sessions_3d.html"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Couleur par famille de minerai, ancree sur l'identite visuelle des minerais en jeu.
# Separation daltonisme (pire paire adjacente dE 14.7 > seuil 12) et contraste fond sombre
# (>= 3:1) valides via le validateur du skill dataviz. Fer et charbon restent volontairement
# peu satures (c'est leur couleur en jeu) : ils sont differencies de la roche par la taille
# des marqueurs, le contour, la legende et le hover.
FAMILY_COLORS: dict[str, str] = {
    "diamond": "#35C7DB",
    "emerald": "#3BB25A",
    "gold": "#E3C230",
    "redstone": "#D64545",
    "lapis": "#4A6FD6",
    "copper": "#C05F20",
    "iron": "#CDB48E",
    "coal": "#878E98",
    "quartz": "#EDE7DB",
    "ancient_debris": "#8A6A55",
}

TUNNEL_COLOR = "#5c6470"
PATH_COLOR = "#8a93a6"
SURFACE_DARK = "#14171c"


def fmt_time(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def fmt_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d/%m/%Y")


def requested_time_window(args: argparse.Namespace) -> tuple[int | None, int | None]:
    """Calcule la fenêtre demandée sans regarder la base.

    Cela permet de pousser le filtre au niveau SQL avant de charger un gros dump.
    """
    if args.start or args.end:
        start = int(parse_utc_datetime(args.start).timestamp()) if args.start else None
        end = int(parse_utc_datetime(args.end).timestamp()) if args.end else None
        if start is None and end is None:
            return None, None
        return start, end

    if args.window == "all":
        return None, None

    now = datetime.now(timezone.utc)
    if args.window == "last-12h":
        return int((now - timedelta(hours=12)).timestamp()), int(now.timestamp())
    if args.window == "last-24h":
        return int((now - timedelta(hours=24)).timestamp()), int(now.timestamp())
    if args.window == "yesterday":
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        return int(yesterday_start.timestamp()), int(now.timestamp())

    raise ValueError(f"Fenetre de temps inconnue : {args.window}")


def _json_safe(values: dict) -> dict:
    """Remplace les NaN par None pour une serialisation JSON stricte."""
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in values.items()
    }


def load_anomaly_models(present: list[str]) -> dict:
    """Charge les modeles d'anomalie disponibles (un par minerai cible, optionnels).

    Le modele complete le score heuristique V1 dans le panneau (bloc "ecart au
    corpus") ; un minerai sans modele entraine affiche simplement l'etat absent.
    """
    models = {}
    for key in present:
        path = MODELS_DIR / f"anomaly_iforest_{key}.joblib"
        if path.exists():
            model = load_model(path)
            if model.target == key:
                models[key] = model
    return models


def build_payload(df: pd.DataFrame, worlds: dict[int, str]) -> dict:
    """Serialise sessions, analyse par minerai cible et palette pour le JS de la page."""
    materials = sorted(df["material"].unique())
    mat_index = {m: i for i, m in enumerate(materials)}
    families = [[key, label, FAMILY_COLORS[key]] for key, label in ORE_FAMILIES.items()]
    fam_index = {key: i for i, (key, _, _) in enumerate(families)}
    present = [key for key in ORE_FAMILIES if (df["ore"] == key).any()]

    models = load_anomaly_models(present)
    if models:
        print("Modeles d'anomalie charges : " + ", ".join(
            f"{k} ({m.n_train_sessions} sessions)" for k, m in models.items()))
    else:
        print("Aucun modele d'anomalie (data/models/) : le panneau n'affichera "
              "que le score V1. Entrainement : scripts/train_anomaly_model.py")

    sessions = []
    for (pseudo, wid, _sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        seg = seg.sort_values("time")
        t0, t1 = int(seg["time"].min()), int(seg["time"].max())
        n_ores = int(seg["ore"].notna().sum())
        dimension = world_dimension(worlds.get(int(wid), f"monde {wid}"))

        # Une session n'est scoree que pour les minerais possibles dans sa
        # dimension : pas de score diamant au Nether ni ancient_debris ailleurs.
        analysis = {}
        for key in present:
            if dimension not in ORE_DIMENSIONS[key]:
                continue
            features = compute_session_features(seg, target=key)
            entry = {**features, **score_session(features, target=key)}
            if key in models:
                scored = score_anomalies(models[key], pd.DataFrame([features])).iloc[0]
                entry["anomaly_score"] = float(scored["anomaly_score"])
                entry["anomaly_top_feature"] = str(scored["anomaly_top_feature"])
            analysis[key] = _json_safe(entry)

        sessions.append(
            {
                "player": pseudo,
                "world": worlds.get(int(wid), f"monde {wid}"),
                "date": fmt_date(t0),
                "t0": t0,
                "t1": t1,
                "label": (
                    f"{fmt_time(t0)} -> {fmt_time(t1)} - "
                    f"{len(seg)} blocs - {n_ores} minerais"
                ),
                "analysis": analysis,
                "t": seg["time"].astype(int).tolist(),
                "x": seg["x"].astype(int).tolist(),
                "y": seg["y"].astype(int).tolist(),
                "z": seg["z"].astype(int).tolist(),
                "m": [mat_index[m] for m in seg["material"]],
                "f": [-1 if pd.isna(f) else fam_index[f] for f in seg["ore"]],
            }
        )

    return {
        "sessions": sessions,
        "materials": [m.removeprefix("minecraft:") for m in materials],
        "families": families,
        "presentFamilies": present,
        "anomalyModels": {
            key: {"n": model.n_train_sessions, "contamination": model.contamination}
            for key, model in models.items()
        },
        "tunnelColor": TUNNEL_COLOR,
        "pathColor": PATH_COLOR,
        "surface": SURFACE_DARK,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Minage 3D - reconstruction et analyse</title>
<style>
  :root {
    color-scheme: dark;
    --surface: #14171c;
    --panel: #171b22;
    --raised: rgba(255, 255, 255, 0.035);
    --border: rgba(255, 255, 255, 0.07);
    --ink: #e8ebf0;
    --ink-2: #9aa3b2;
    --ink-3: #67707f;
    --accent: #35C7DB;
    --good: #0ca30c;
    --warn: #fab219;
    --crit: #d03b3b;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    display: flex; flex-direction: column; overflow: hidden;
    background: var(--surface); color: var(--ink);
    font: 13px/1.45 system-ui, "Segoe UI", sans-serif;
  }
  header {
    flex: 0 0 auto; display: flex; align-items: center; flex-wrap: wrap;
    gap: 10px 22px; padding: 10px 16px;
    background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .app-title { font-weight: 700; font-size: 14px; letter-spacing: 0.02em; }
  .app-title .dim { color: var(--ink-3); font-weight: 400; }
  .ctrl { display: flex; align-items: center; gap: 8px; }
  label { color: var(--ink-2); }
  select, input[type="time"], button {
    background: var(--raised); color: var(--ink); border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 9px; font: inherit;
  }
  /* La liste deroulante native herite du background-color du <select> : il doit
     etre opaque, sinon Windows la peint en blanc sous le texte clair. */
  select { background-color: #1e242d; }
  select option, select optgroup { background-color: #1e242d; color: var(--ink); }
  select:focus-visible, input:focus-visible, button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
  }
  input[type="time"] { width: 106px; }
  button { cursor: pointer; }
  button:hover { background: rgba(255, 255, 255, 0.08); }

  /* Double curseur : deux <input type=range> superposes sur une piste custom */
  .range-wrap { position: relative; width: 280px; height: 28px; }
  .range-wrap .track, .range-wrap .fill {
    position: absolute; top: 50%; height: 4px; border-radius: 2px;
    transform: translateY(-50%); pointer-events: none;
  }
  .range-wrap .track { left: 0; right: 0; background: #333b48; }
  .range-wrap .fill { background: var(--accent); }
  .range-wrap input[type="range"] {
    -webkit-appearance: none; appearance: none;
    position: absolute; left: 0; top: 50%; transform: translateY(-50%);
    width: 100%; margin: 0; background: none; pointer-events: none; height: 28px;
  }
  .range-wrap input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; pointer-events: auto;
    width: 15px; height: 15px; border-radius: 50%;
    background: var(--ink); border: 3px solid var(--accent); cursor: ew-resize;
  }
  .range-wrap input[type="range"]::-moz-range-thumb {
    pointer-events: auto; width: 10px; height: 10px; border-radius: 50%;
    background: var(--ink); border: 3px solid var(--accent); cursor: ew-resize;
  }
  #slider-start { z-index: 3; } #slider-end { z-index: 4; }

  #session-info { color: var(--ink-2); }
  #session-info strong { color: var(--ink); }
  #stats { color: var(--ink-2); margin-left: auto; text-align: right; }
  #stats strong { color: var(--ink); font-variant-numeric: tabular-nums; }
  #stats .diamond { color: var(--accent); }

  main { flex: 1 1 auto; display: flex; min-height: 0; }
  #plot-wrap { flex: 1 1 auto; min-width: 0; position: relative; }
  #plot { position: absolute; inset: 0; }

  #filters {
    /* haut-gauche : la modebar Plotly occupe le coin haut-droit */
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(20, 23, 28, 0.9); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px; min-width: 185px;
    display: flex; flex-direction: column; gap: 1px;
  }
  .filter-row {
    display: flex; align-items: center; gap: 8px; padding: 4px 7px;
    border-radius: 7px; cursor: pointer; user-select: none;
  }
  .filter-row:hover { background: var(--raised); }
  .filter-row.all {
    border-bottom: 1px solid var(--border); border-radius: 7px 7px 0 0;
    padding-bottom: 7px; margin-bottom: 4px;
  }
  .filter-row .box {
    width: 14px; height: 14px; border-radius: 4px; border: 1px solid var(--ink-3);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; color: var(--surface); flex: 0 0 auto;
  }
  .filter-row.on .box, .filter-row.some .box {
    background: var(--accent); border-color: var(--accent);
  }
  .filter-row.some .box { opacity: 0.55; }
  .filter-row .swatch { width: 10px; height: 10px; border-radius: 3px; flex: 0 0 auto; }
  .filter-row .lbl { flex: 1 1 auto; color: var(--ink-2); font-size: 12px; }
  .filter-row.on .lbl { color: var(--ink); }
  .filter-row .cnt {
    color: var(--ink-3); font-size: 11px; font-variant-numeric: tabular-nums;
  }

  #modal, #overview {
    position: fixed; inset: 0; z-index: 50; display: none;
    align-items: center; justify-content: center;
    background: rgba(8, 10, 13, 0.62);
  }
  #modal.open, #overview.open { display: flex; }
  .modal-card {
    width: min(780px, calc(100vw - 40px)); max-height: calc(100vh - 60px);
    overflow-y: auto; background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px 28px 28px;
  }
  .modal-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px;
  }
  .modal-head h2 { margin: 0; font-size: 17px; }
  .modal-card h3 {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--ink-3); margin: 24px 0 10px;
  }
  .modal-card p, .modal-card li {
    color: var(--ink-2); font-size: 12.5px; line-height: 1.65; margin: 6px 0;
  }
  .modal-card strong { color: var(--ink); }
  .metric-def {
    margin: 8px 0; padding: 10px 12px; background: var(--raised);
    border: 1px solid var(--border); border-radius: 10px;
  }
  .metric-def > strong { display: block; margin-bottom: 2px; font-size: 12.5px; }
  .metric-def p { margin: 2px 0 0; }

  aside {
    flex: 0 0 332px; overflow-y: auto; padding: 18px 18px 14px;
    background: var(--panel); border-left: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 18px;
  }
  aside.hidden { display: none; }
  .section-title {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--ink-3); margin-bottom: 10px;
  }
  .panel-head { display: flex; align-items: center; justify-content: space-between; }
  .panel-head select { width: 100%; }

  .score-block { display: flex; align-items: center; gap: 18px; }
  .score-ring { position: relative; width: 108px; height: 108px; flex: 0 0 auto; }
  .score-ring svg { transform: rotate(-90deg); }
  .score-ring .num {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
  }
  .score-ring .num b { font-size: 24px; font-variant-numeric: tabular-nums; }
  .score-ring .num span { font-size: 10px; color: var(--ink-3); }
  .verdict {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 11px; border-radius: 999px; font-weight: 600; font-size: 12px;
    background: var(--raised); border: 1px solid var(--border);
  }
  .verdict .dot { width: 8px; height: 8px; border-radius: 50%; }
  .score-side { display: flex; flex-direction: column; gap: 9px; }
  .score-side .who { font-size: 13px; color: var(--ink-2); }
  .score-side .who strong { color: var(--ink); }

  .meter { margin-bottom: 12px; }
  .meter .row {
    display: flex; justify-content: space-between; margin-bottom: 5px;
  }
  .meter .name { color: var(--ink-2); }
  .meter .val { color: var(--ink); font-variant-numeric: tabular-nums; }
  .meter .bar {
    height: 6px; border-radius: 3px; background: #242a35; overflow: hidden;
  }
  .meter .bar i {
    display: block; height: 100%; border-radius: 3px;
    background: var(--accent); transition: width 0.25s ease;
  }
  .meter .hint { margin-top: 4px; font-size: 10.5px; color: var(--ink-3); }

  /* Ecart au corpus (modele d'anomalie) : jauge 0-100 avec repere au seuil 50 */
  .anomaly-bar {
    position: relative; height: 6px; border-radius: 3px;
    background: #242a35; overflow: visible;
  }
  .anomaly-bar i {
    display: block; height: 100%; border-radius: 3px;
    background: var(--accent); transition: width 0.25s ease;
  }
  .anomaly-bar .tick {
    position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px;
    background: var(--ink-3); border-radius: 1px;
  }
  .anomaly-empty { font-size: 11px; color: var(--ink-3); line-height: 1.5; }

  .tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .tile {
    background: var(--raised); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 10px;
  }
  .tile b {
    display: block; font-size: 14.5px; font-variant-numeric: tabular-nums;
  }
  .tile span { font-size: 10.5px; color: var(--ink-3); }

  .loc {
    display: flex; align-items: center; gap: 10px; padding: 8px 10px;
    background: var(--raised); border: 1px solid var(--border); border-radius: 10px;
  }
  .loc-info { flex: 1 1 auto; min-width: 0; }
  .loc-info b { display: block; font-size: 12.5px; }
  .loc-info span {
    font-size: 11px; color: var(--ink-3); font-variant-numeric: tabular-nums;
  }
  #copy-tp { white-space: nowrap; }
  #copy-tp.copied, #anno-export.copied { border-color: #0ca30c; color: #7fd67f; }

  /* Mode annotation (page generee avec --annotation) */
  .anno-row { display: flex; gap: 6px; }
  .anno-row + .anno-row { margin-top: 6px; }
  .anno-row button { flex: 1 1 0; }
  .anno-row button.on[data-v="legit"] { border-color: var(--good); color: var(--good); }
  .anno-row button.on[data-v="suspect"] { border-color: var(--warn); color: var(--warn); }
  .anno-row button.on[data-v="triche"] { border-color: var(--crit); color: var(--crit); }
  #anno-grotte.on { border-color: var(--accent); color: var(--accent); }
  #anno-export { width: 100%; margin-top: 8px; }
  .rank-row .anno-mark { font-size: 12px; flex: 0 0 auto; }

  .rank-controls { display: flex; gap: 6px; margin-bottom: 8px; }
  .rank-controls input[type="search"] {
    flex: 1 1 auto; min-width: 0;
    background: var(--raised); color: var(--ink); border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 9px; font: inherit;
  }
  .rank-controls select { flex: 0 0 auto; max-width: 130px; }
  .rank { display: flex; flex-direction: column; gap: 6px; }
  .rank-row {
    display: flex; align-items: center; gap: 10px; padding: 8px 10px;
    border-radius: 10px; border: 1px solid transparent; cursor: pointer;
  }
  .rank-row:hover { background: var(--raised); }
  .rank-row.active { border-color: var(--border); background: var(--raised); }
  .rank-row .who { flex: 1 1 auto; min-width: 0; }
  .rank-row .who b { display: block; font-size: 12.5px; }
  .rank-row .who span { font-size: 10.5px; color: var(--ink-3); }
  .chip {
    padding: 3px 9px; border-radius: 999px; font-weight: 700; font-size: 11.5px;
    font-variant-numeric: tabular-nums;
  }
  .footnote { font-size: 10.5px; color: var(--ink-3); line-height: 1.5; }

  /* Vue d'ensemble : histogramme des scores + nuage V1 x anomalie sur tout le corpus */
  .modal-card.wide { width: min(1120px, calc(100vw - 40px)); }
  .overview-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 14px;
  }
  @media (max-width: 940px) { .overview-grid { grid-template-columns: 1fr; } }
  .chart-block {
    background: var(--raised); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }
  .chart-block h4 { margin: 0 0 2px; font-size: 13px; color: var(--ink); }
  .chart-block .hint { font-size: 11px; color: var(--ink-3); margin: 0 0 8px; }
  .chart-block svg, .chart-block canvas { width: 100%; height: auto; display: block; }
  #ov-canvas { cursor: pointer; }
  .ov-legend {
    display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px;
    font-size: 12px; color: var(--ink-2);
  }
  .ov-legend b { color: var(--ink); font-variant-numeric: tabular-nums; }
  .ov-legend .sw {
    display: inline-block; width: 10px; height: 10px; border-radius: 3px;
    margin-right: 6px; vertical-align: -1px;
  }
  #ov-tip {
    position: fixed; z-index: 60; display: none; pointer-events: none;
    background: #0d0f13; border: 1px solid var(--border); border-radius: 7px;
    padding: 6px 10px; font-size: 12px; color: var(--ink-2);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
  }
  #ov-tip b { color: var(--ink); }
</style>
</head>
<body>
<header>
  <div class="app-title">Minage 3D <span class="dim">- reconstruction &amp; analyse</span></div>
  <div class="ctrl" id="world-ctrl">
    <label for="world-select">Monde</label>
    <select id="world-select"></select>
  </div>
  <div class="ctrl">
    <label for="session-select">Session</label>
    <select id="session-select"></select>
    <span id="session-info"></span>
  </div>
  <div class="ctrl">
    <label for="time-start">Fenetre</label>
    <input type="time" id="time-start" step="1">
    <div class="range-wrap">
      <div class="track"></div>
      <div class="fill" id="range-fill"></div>
      <input type="range" id="slider-start">
      <input type="range" id="slider-end">
    </div>
    <input type="time" id="time-end" step="1">
    <button id="reset-range" title="Revenir a la session entiere">Session entiere</button>
    <button id="toggle-panel" title="Afficher / masquer l'analyse">Analyse</button>
    <button id="open-overview" title="Distribution des scores et croisement V1 x anomalie sur toutes les sessions">Vue d'ensemble</button>
    <button id="open-metrics" title="Comprendre les metriques">Métriques&nbsp;?</button>
  </div>
  <div id="stats"></div>
</header>
<main>
  <div id="plot-wrap">
    <div id="plot"></div>
    <div id="filters"></div>
  </div>
  <aside id="panel">
    <div>
      <div class="section-title">Minerai surveille</div>
      <div class="panel-head"><select id="ore-select"></select></div>
    </div>
    <div>
      <div class="section-title">Score de suspicion x-ray</div>
      <div class="score-block">
        <div class="score-ring">
          <svg width="108" height="108" viewBox="0 0 108 108">
            <circle cx="54" cy="54" r="47" fill="none" stroke="#242a35" stroke-width="9"/>
            <circle id="ring-fill" cx="54" cy="54" r="47" fill="none" stroke-width="9"
                    stroke-linecap="round" stroke-dasharray="295.3" stroke-dashoffset="295.3"/>
          </svg>
          <div class="num"><b id="score-num">-</b><span>/ 100</span></div>
        </div>
        <div class="score-side">
          <div class="who" id="score-who"></div>
          <div class="verdict"><span class="dot" id="verdict-dot"></span>
            <span id="verdict-label">-</span></div>
        </div>
      </div>
    </div>
    <div>
      <div class="section-title">Indicateurs du score</div>
      <div id="meters"></div>
    </div>
    <div>
      <div class="section-title">Écart au corpus (modèle d'anomalie)</div>
      <div id="anomaly-block"></div>
    </div>
    <div>
      <div class="section-title">Details de la session</div>
      <div class="tiles" id="tiles"></div>
    </div>
    <div>
      <div class="section-title">Localisation</div>
      <div class="loc">
        <div class="loc-info">
          <b id="loc-world">-</b>
          <span id="loc-coords">-</span>
        </div>
        <button id="copy-tp" title="Copier la commande de teleportation vers le centre de la zone minee">Copier /tp</button>
      </div>
    </div>
    <div id="annotation-section" hidden>
      <div class="section-title">Annotation de la session</div>
      <div class="anno-row">
        <button data-v="legit">Legit</button>
        <button data-v="suspect">Suspect</button>
        <button data-v="triche">Triche</button>
      </div>
      <div class="anno-row">
        <button id="anno-grotte" title="Tag special, cumulable avec le verdict : la session est une grotte / cavite naturelle">Grotte</button>
      </div>
      <button id="anno-export" title="Copie/telecharge les annotations au format data/labels/session_labels.csv">Exporter CSV</button>
    </div>
    <div>
      <div class="section-title">Classement des sessions</div>
      <div class="rank-controls">
        <input type="search" id="rank-search" placeholder="Chercher un joueur…"
               aria-label="Filtrer le classement par joueur">
        <select id="rank-sort" aria-label="Indicateur de tri du classement">
          <option value="score">Score V1</option>
          <option value="anomaly">Écart au corpus</option>
          <option value="mix">Mix V1 + écart</option>
          <option value="blocks">Blocs cassés</option>
          <option value="duration">Durée</option>
        </select>
      </div>
      <div class="rank" id="rank"></div>
    </div>
    <div class="footnote">
      Score heuristique V1 et ecart au corpus (Isolation Forest), calcules sur la
      session entiere (la fenetre temporelle filtre la scene 3D, pas l'analyse).
      Voir readmeAnalyse.md pour la methode.
    </div>
  </aside>
</main>

<div id="modal">
  <div class="modal-card">
    <div class="modal-head">
      <h2>Comprendre les métriques</h2>
      <button id="modal-close">Fermer</button>
    </div>
    <p>La trajectoire d'une session est reconstituée de bloc cassé en bloc cassé,
    dans l'ordre chronologique des logs CoreProtect. Un pas de plus de 4 blocs est
    considéré comme un déplacement sans minage (marche en grotte, chute,
    téléportation)&nbsp;: il coupe la continuité mais ne compte pas comme un virage.</p>

    <h3>Lecture de la scène 3D</h3>
    <p>Les points gris translucides sont la roche cassée&nbsp;: ils dessinent la forme
    des tunnels. La ligne fine relie les blocs dans l'ordre où ils ont été cassés.
    Les carrés colorés sont les minerais, une couleur par famille&nbsp;; le minerai
    surveillé est affiché plus gros (⭐). Le panneau de filtres en haut à droite
    permet d'afficher ou de masquer chaque couche, ou tout d'un coup.</p>

    <h3>Les trois indicateurs du score</h3>
    <div class="metric-def"><strong>Rendement (minerai cible / 100 blocs creusés)</strong>
      <p>Combien de blocs du minerai surveillé le joueur trouve pour 100 blocs
      <em>creusés</em> — seules les casses en phase de creusage comptent (pas
      ≤ 2 blocs entre casses consécutives) : les minerais ramassés en marchant
      dans une grotte, exposés et visibles, n'entrent pas dans ce rendement.
      Un strip-mineur légitime à Y-59 trouve environ 0,3 à 0,8 diamant
      pour 100 blocs&nbsp;; au-delà de 3, le rendement n'est plus explicable par la
      chance. Les bornes s'adaptent au minerai choisi (trouver 5 fers / 100 blocs
      est banal, 5 diamants ne l'est pas).</p></div>
    <div class="metric-def"><strong>Détour entre filons</strong>
      <p>Longueur du chemin réellement miné entre deux filons successifs, divisée
      par la distance à vol d'oiseau. 1× = ligne parfaitement droite de filon en
      filon. Un joueur légitime quadrille et retombe sur les filons par hasard
      (≥&nbsp;3×)&nbsp;; un x-rayeur va presque tout droit (≤&nbsp;1,4× déclenche l'alerte,
      en pratique ~2,5× à cause du tunnel de 2 de haut qui zigzague bloc à bloc).
      Seuls les filons <em>atteints en creusant</em> comptent, et une paire
      traversée par un pas de marche est ignorée&nbsp;: marcher droit vers un
      minerai visible en grotte n'est pas suspect.</p></div>
    <div class="metric-def"><strong>Virages orientés vers le prochain filon</strong>
      <p>À chaque changement de direction, la nouvelle direction rapproche-t-elle
      du prochain filon <strong>pas encore découvert</strong> (et atteint en
      creusant)&nbsp;? Un virage au hasard rapproche environ 1 fois sur 2 (50&nbsp;%).
      Viser juste presque à chaque virage trahit une information que le joueur ne
      devrait pas avoir.</p></div>

    <h3>Les détails de la session</h3>
    <div class="metric-def"><strong>Filons / blocs entre filons</strong>
      <p>Les casses du minerai cible sont regroupées en filons quand elles sont à
      moins de 2 blocs les unes des autres. «&nbsp;Blocs entre filons&nbsp;» compte les
      blocs minés entre la fin d'un filon et le début du suivant&nbsp;: c'est le
      «&nbsp;combien je creuse avant de trouver&nbsp;».</p></div>
    <div class="metric-def"><strong>Segments droits H / V</strong>
      <p>Longueur moyenne (en blocs) des tronçons parcourus sans changer de
      direction, séparés en horizontal et vertical. Attention&nbsp;: le minage en
      tunnel de 2 de haut alterne un pas avant / un pas vertical, ce qui écrase
      cette moyenne pour tout le monde (~1&nbsp;bloc)&nbsp;— métrique à affiner.</p></div>
    <div class="metric-def"><strong>Virages / 100 et pas verticaux</strong>
      <p>Fréquence des changements de direction pour 100 pas, et part des pas dont
      le mouvement dominant est vertical (plongées et remontées vers des cibles).</p></div>

    <h3>Le score de suspicion</h3>
    <p>Chaque indicateur est normalisé entre 0 et 1 par une rampe bornée (sous la
    borne basse&nbsp;: 0, au-dessus de la borne haute&nbsp;: 1 — les jauges du panneau
    montrent cette valeur normalisée), puis combiné&nbsp;: <strong>40&nbsp;% rendement,
    30&nbsp;% détour, 30&nbsp;% virages</strong>. Verdicts&nbsp;: <strong>≥&nbsp;60</strong>
    fortement suspect, <strong>≥&nbsp;30</strong> à surveiller, sinon RAS. Si un indicateur est
    incalculable (moins de deux filons par exemple), il est retiré et les poids
    sont renormalisés.</p>
    <p>C'est une <strong>heuristique V1</strong> calibrée sur la connaissance du jeu,
    pensée pour être remplacée par un modèle entraîné dès qu'un corpus étiqueté
    suffisant existera. Le score porte toujours sur la session entière&nbsp;: la
    fenêtre temporelle filtre la scène, pas l'analyse.</p>

    <h3>L'écart au corpus (modèle d'anomalie)</h3>
    <p>Un second regard, indépendant du score&nbsp;: un <strong>Isolation
    Forest</strong> (modèle non supervisé) entraîné sur les sessions de la vraie
    base mesure à quel point la session s'écarte d'une session <em>typique</em> du
    corpus, sur 13 features à la fois — y compris celles que le score V1
    n'utilise pas (forme du chemin, part creusée…). La jauge est normalisée
    0-100&nbsp;: <strong>50 = seuil de contamination</strong> (le repère sur la
    barre)&nbsp;; au-delà, la session est plus atypique que l'immense majorité du
    corpus d'entraînement. «&nbsp;Tiré par&nbsp;» indique la feature qui contribue le
    plus à l'écart.</p>
    <p><strong>Atypique ne veut pas dire tricheur.</strong> Le modèle n'a aucune
    étiquette&nbsp;: il dit «&nbsp;cette session ne ressemble pas au corpus&nbsp;», rien de
    plus. Une session très haute sur cette jauge mais RAS au score V1 mérite une
    inspection visuelle, pas une sanction. Voir readmeAnalyse.md pour la méthode
    (features, imputation, directionnalité, limites).</p>
  </div>
</div>

<div id="overview">
  <div class="modal-card wide">
    <div class="modal-head">
      <div style="display:flex; align-items:center; gap:12px">
        <h2>Vue d'ensemble</h2>
        <select id="ov-ore" aria-label="Minerai analysé dans la vue d'ensemble"></select>
      </div>
      <button id="overview-close">Fermer</button>
    </div>
    <p id="ov-summary"></p>
    <div class="overview-grid">
      <div class="chart-block">
        <h4>Distribution des scores de suspicion</h4>
        <p class="hint">Sessions par tranche de 5 points — échelle verticale logarithmique
          (la masse RAS écraserait tout en linéaire)</p>
        <svg id="ov-histo" viewBox="0 0 520 300" role="img"
             aria-label="Histogramme des scores de suspicion"></svg>
      </div>
      <div class="chart-block">
        <h4>Score V1 × écart au corpus (modèle d'anomalie)</h4>
        <p class="hint">Un point par session, couleur = verdict — cliquer un point ouvre la
          session dans la scène 3D</p>
        <canvas id="ov-canvas" width="1040" height="600"></canvas>
      </div>
    </div>
    <div class="ov-legend" id="ov-legend"></div>
  </div>
</div>
<div id="ov-tip"></div>

<script>/*__PLOTLY_JS__*/</script>
<script>
"use strict";
const DATA = /*__DATA_JSON__*/;

const el = (id) => document.getElementById(id);
const plotDiv = el("plot");
const state = { i: 0, tA: 0, tB: 0, target: "diamond", hidden: new Set(),
                rankSort: "score", rankQuery: "", world: "" };
if (!DATA.presentFamilies.includes(state.target)) state.target = DATA.presentFamilies[0];

const FAMILY_LABEL = {}, FAMILY_INDEX = {};
DATA.families.forEach((f, i) => { FAMILY_LABEL[f[0]] = f[1]; FAMILY_INDEX[f[0]] = i; });

// Libelles courts des features pour l'explication du modele d'anomalie.
const FEATURE_LABEL = {
  target_per_100_dig: "rendement (creusage)",
  target_per_100: "rendement (session entière)",
  ore_per_100: "minerais / 100 blocs",
  mean_blocks_between_veins: "blocs entre filons",
  detour_factor: "détour entre filons",
  turn_toward_ore_rate: "virages vers le filon",
  changes_per_100: "virages / 100",
  mean_run_h: "segments droits H",
  mean_run_v: "segments droits V",
  vertical_step_ratio: "pas verticaux",
  dig_ratio: "part creusée",
  walk_step_ratio: "pas de marche",
  path_straightness: "rectitude du chemin",
};

const VERDICT_STYLE = {
  "fortement suspect": ["var(--crit)", "fortement suspect"],
  "a surveiller": ["var(--warn)", "à surveiller"],
  "RAS": ["var(--good)", "RAS"],
  "indeterminable": ["var(--ink-3)", "indéterminable"],
};

const pad = (n) => String(n).padStart(2, "0");
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes()) + ":" + pad(d.getUTCSeconds());
}
const fmtVal = (v, suffix) => (v === null || v === undefined) ? "—" : v + (suffix || "");

// "HH:MM[:SS]" -> timestamp dans la journee de la session, borne a [t0, t1]
function parseTimeInput(value, S) {
  const parts = value.split(":").map(Number);
  if (parts.some(isNaN) || parts.length < 2) return null;
  const secs = parts[0] * 3600 + parts[1] * 60 + (parts[2] || 0);
  let ts = Math.floor(S.t0 / 86400) * 86400 + secs;
  if (ts < S.t0 && ts + 86400 <= S.t1) ts += 86400; // session a cheval sur minuit
  return Math.min(Math.max(ts, S.t0), S.t1);
}

function axisStyle(title) {
  return {
    title: { text: title },
    showbackground: true, backgroundcolor: DATA.surface,
    gridcolor: "#2a2f38", zerolinecolor: "#2a2f38", color: "#aab2bf",
  };
}

function buildTraces(S) {
  const idx = [];
  for (let k = 0; k < S.t.length; k++) {
    if (S.t[k] >= state.tA && S.t[k] <= state.tB) idx.push(k);
  }

  const hover = (k) =>
    DATA.materials[S.m[k]] + "<br>" + fmtTime(S.t[k]) + " UTC - Y=" + S.y[k];
  const empty = () => ({ x: [], y: [], z: [], text: [] });

  const tunnel = empty();
  const path = { x: [], y: [], z: [] };
  const byFam = new Map();
  const counts = { tunnel: 0, fam: new Map() };
  for (const k of idx) {
    path.x.push(S.x[k]); path.y.push(S.z[k]); path.z.push(S.y[k]);
    let bucket = null;
    if (S.f[k] >= 0) {
      counts.fam.set(S.f[k], (counts.fam.get(S.f[k]) || 0) + 1);
      if (!state.hidden.has(DATA.families[S.f[k]][0])) {
        if (!byFam.has(S.f[k])) byFam.set(S.f[k], empty());
        bucket = byFam.get(S.f[k]);
      }
    } else {
      counts.tunnel++;
      if (!state.hidden.has("tunnel")) bucket = tunnel;
    }
    if (bucket === null) continue;
    // Convention scene : x=X, y=Z, z=Y (la hauteur est verticale)
    bucket.x.push(S.x[k]); bucket.y.push(S.z[k]); bucket.z.push(S.y[k]);
    bucket.text.push(hover(k));
  }

  const traces = [];
  if (!state.hidden.has("tunnel")) {
    traces.push({
      type: "scatter3d", mode: "markers", name: "Roche / terrain",
      x: tunnel.x, y: tunnel.y, z: tunnel.z, text: tunnel.text, hoverinfo: "text",
      marker: { size: 2.5, color: DATA.tunnelColor, opacity: 0.35, symbol: "square" },
    });
  }
  if (!state.hidden.has("path")) {
    traces.push({
      type: "scatter3d", mode: "lines", name: "Progression", hoverinfo: "skip",
      x: path.x, y: path.y, z: path.z,
      line: { color: DATA.pathColor, width: 1.5 }, opacity: 0.45,
    });
  }
  for (const [fi, b] of [...byFam.entries()].sort((a, c) => a[0] - c[0])) {
    const [key, , color] = DATA.families[fi];
    traces.push({
      type: "scatter3d", mode: "markers", name: key, hoverinfo: "text",
      x: b.x, y: b.y, z: b.z, text: b.text,
      marker: { size: key === state.target ? 6.5 : 4.5, color: color, symbol: "square",
                line: { color: DATA.surface, width: 1 } },
    });
  }

  const targetIdx = FAMILY_INDEX[state.target];
  const nOres = idx.reduce((n, k) => n + (S.f[k] >= 0 ? 1 : 0), 0);
  const nTarget = idx.reduce((n, k) => n + (S.f[k] === targetIdx ? 1 : 0), 0);
  return { traces, nBlocks: idx.length, nOres, nTarget, counts };
}

function sessionFamilies(S) {
  return [...new Set(S.f.filter((v) => v >= 0))].sort((a, b) => a - b);
}

function renderFilters(counts) {
  const S = DATA.sessions[state.i];
  const groups = [
    { id: "tunnel", label: "Roche / terrain", color: DATA.tunnelColor,
      cnt: counts.tunnel },
    { id: "path", label: "Progression", color: DATA.pathColor, cnt: null },
  ];
  for (const fi of sessionFamilies(S)) {
    const [key, label, color] = DATA.families[fi];
    groups.push({
      id: key, label: label + (key === state.target ? " ⭐" : ""),
      color: color, cnt: counts.fam.get(fi) || 0,
    });
  }

  const hiddenHere = groups.filter((g) => state.hidden.has(g.id)).length;
  const allState = hiddenHere === 0 ? "on" : (hiddenHere === groups.length ? "" : "some");
  const mark = (s) => s === "on" ? "✓" : (s === "some" ? "–" : "");

  let html = "<div class=\"filter-row all " + allState + "\" data-id=\"__all__\">" +
    "<span class=\"box\">" + mark(allState) + "</span>" +
    "<span class=\"lbl\">Tout afficher</span></div>";
  for (const g of groups) {
    const on = !state.hidden.has(g.id);
    html += "<div class=\"filter-row " + (on ? "on" : "") + "\" data-id=\"" + g.id +
      "\"><span class=\"box\">" + (on ? "✓" : "") + "</span>" +
      "<span class=\"swatch\" style=\"background:" + g.color + "\"></span>" +
      "<span class=\"lbl\">" + g.label + "</span>" +
      (g.cnt === null ? "" : "<span class=\"cnt\">" + g.cnt + "</span>") + "</div>";
  }
  el("filters").innerHTML = html;

  for (const row of el("filters").querySelectorAll(".filter-row")) {
    row.addEventListener("click", () => {
      const id = row.dataset.id;
      if (id === "__all__") {
        if (state.hidden.size) state.hidden.clear();
        else {
          state.hidden = new Set(groups.map((g) => g.id));
        }
      } else if (state.hidden.has(id)) {
        state.hidden.delete(id);
      } else {
        state.hidden.add(id);
      }
      render();
    });
  }
}

function render() {
  const S = DATA.sessions[state.i];
  const { traces, nBlocks, nOres, nTarget, counts } = buildTraces(S);

  Plotly.react(plotDiv, traces, {
    paper_bgcolor: DATA.surface,
    font: { color: "#aab2bf", family: "system-ui, Segoe UI, sans-serif" },
    margin: { l: 0, r: 0, t: 8, b: 0 },
    uirevision: "s" + state.i,
    showlegend: false,
    scene: {
      bgcolor: DATA.surface, aspectmode: "data", uirevision: "s" + state.i,
      xaxis: axisStyle("X"), yaxis: axisStyle("Z"), zaxis: axisStyle("Y (altitude)"),
    },
  }, { responsive: true, displaylogo: false });

  const pct = nBlocks ? (100 * nOres / nBlocks).toFixed(1) : "0.0";
  el("stats").innerHTML =
    "<strong>" + nBlocks + "</strong> blocs - <strong>" + nOres + "</strong> minerais (" +
    pct + " %) - <strong class=\"diamond\">" + nTarget + "</strong> " +
    FAMILY_LABEL[state.target].toLowerCase();
  renderFilters(counts);
}

function renderPanel() {
  const S = DATA.sessions[state.i];
  // Session hors dimension du minerai surveille (ex. Nether pour le diamant) :
  // aucune analyse n'existe, le panneau l'explique au lieu d'afficher du vide.
  const offDim = !(state.target in S.analysis);
  const A = S.analysis[state.target] || {};
  const [color, verdictText] = offDim
    ? ["var(--ink-3)", "minerai absent de ce monde"]
    : (VERDICT_STYLE[A.verdict] || VERDICT_STYLE["indeterminable"]);
  const label = FAMILY_LABEL[state.target];

  const score = (A.score === null || A.score === undefined) ? null : A.score;
  el("score-num").textContent = score === null ? "—" : score;
  const ring = el("ring-fill");
  const C = 2 * Math.PI * 47;
  ring.style.stroke = color;
  ring.setAttribute("stroke-dashoffset", score === null ? C : C * (1 - score / 100));
  el("verdict-dot").style.background = color;
  el("verdict-label").textContent = verdictText;
  el("score-who").innerHTML =
    "<strong>" + S.player + "</strong><br>" + S.date + " · " +
    fmtTime(S.t0) + " → " + fmtTime(S.t1);

  const meters = [
    ["Rendement (creusage)", fmtVal(A.target_per_100_dig, " / 100 blocs"), A.ind_target_per_100_dig,
     label + " trouvés pour 100 blocs creusés"],
    ["Détour entre filons", fmtVal(A.detour_factor, "×"), A.ind_detour_factor,
     "1× = ligne droite de filon en filon"],
    ["Virages vers le filon", A.turn_toward_ore_rate === null ? "—"
      : Math.round(A.turn_toward_ore_rate * 100) + " %", A.ind_turn_toward_ore_rate,
     "50 % = hasard · viser juste trahit le x-ray"],
  ];
  el("meters").innerHTML = meters.map(([name, val, ind, hint]) =>
    "<div class=\"meter\"><div class=\"row\"><span class=\"name\">" + name +
    "</span><span class=\"val\">" + val + "</span></div><div class=\"bar\"><i style=\"width:" +
    (ind === null || ind === undefined ? 0 : Math.round(ind * 100)) +
    "%\"></i></div><div class=\"hint\">" + hint + "</div></div>"
  ).join("");

  const AM = (DATA.anomalyModels || {})[state.target];
  const an = A.anomaly_score;
  let anomalyHtml;
  if (!AM) {
    anomalyHtml = "<div class=\"anomaly-empty\">Pas de modèle entraîné pour " +
      label.toLowerCase() + " — voir scripts/train_anomaly_model.py.</div>";
  } else if (an === null || an === undefined) {
    anomalyHtml = "<div class=\"anomaly-empty\">Score indisponible pour cette session.</div>";
  } else {
    const topFeat = A.anomaly_top_feature ?
      (FEATURE_LABEL[A.anomaly_top_feature] || A.anomaly_top_feature) : null;
    const reading = an >= 50
      ? "plus atypique que " + Math.round(100 * (1 - AM.contamination)) +
        " % du corpus d'entraînement"
      : "dans la normale du corpus d'entraînement";
    anomalyHtml =
      "<div class=\"meter\"><div class=\"row\"><span class=\"name\">Isolation Forest (" +
      AM.n + " sessions)</span><span class=\"val\">" + an + " / 100</span></div>" +
      "<div class=\"anomaly-bar\"><i style=\"width:" + Math.round(an) +
      "%\"></i><span class=\"tick\" title=\"50 = seuil de contamination\"></span></div>" +
      "<div class=\"hint\">" + reading +
      (topFeat ? " · tiré par : <b>" + topFeat + "</b>" : "") +
      " · atypique ≠ tricheur</div></div>";
  }
  el("anomaly-block").innerHTML = anomalyHtml;

  const tiles = [
    [fmtVal(A.duration_min, " min"), "Durée"],
    [fmtVal(A.n_blocks), "Blocs cassés"],
    [fmtVal(A.blocks_per_min), "Blocs / min"],
    [fmtVal(A.ore_per_100), "Minerais / 100"],
    [fmtVal(A.n_target_veins), "Filons (" + label.toLowerCase() + ")"],
    [fmtVal(A.mean_blocks_between_veins), "Blocs entre filons"],
    [fmtVal(A.mean_run_h), "Segment droit H"],
    [fmtVal(A.mean_run_v), "Segment droit V"],
    [fmtVal(A.changes_per_100), "Virages / 100"],
    [A.vertical_step_ratio === null ? "—"
      : Math.round(A.vertical_step_ratio * 100) + " %", "Pas verticaux"],
  ];
  el("tiles").innerHTML = tiles.map(([v, name]) =>
    "<div class=\"tile\"><b>" + v + "</b><span>" + name + "</span></div>").join("");

  renderRank();
}

// Valeur de tri / affichage du classement selon l'indicateur choisi.
// "mix" = moyenne des deux regards (V1 et ecart au corpus) quand les deux existent.
function rankValue(a) {
  const s = a.score, an = a.anomaly_score;
  switch (state.rankSort) {
    case "anomaly": return an ?? null;
    case "mix":
      if (s !== null && s !== undefined && an !== null && an !== undefined)
        return Math.round((s + an) / 2 * 10) / 10;
      return s ?? an ?? null;
    case "blocks": return a.n_blocks ?? null;
    case "duration": return a.duration_min ?? null;
    default: return s ?? null;
  }
}

// Le selecteur de monde du bandeau filtre la liste des sessions, le classement
// et la vue d'ensemble ("" = tous les mondes).
const inWorld = (S) => !state.world || S.world === state.world;

function renderRank() {
  const q = state.rankQuery.trim().toLowerCase();
  const order = DATA.sessions
    .map((s, i) => [i, rankValue(s.analysis[state.target] || {})])
    .filter(([i]) => inWorld(DATA.sessions[i]) &&
      (state.target in DATA.sessions[i].analysis) &&
      (!q || DATA.sessions[i].player.toLowerCase().includes(q)))
    .sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1));
  const ANNO_MARK = { legit: ["✓", "var(--good)"], suspect: ["?", "var(--warn)"],
                      triche: ["✗", "var(--crit)"] };
  el("rank").innerHTML = order.map(([i, val]) => {
    const s = DATA.sessions[i];
    const v = (s.analysis[state.target] || {}).verdict;
    const [c] = VERDICT_STYLE[v] || VERDICT_STYLE["indeterminable"];
    let mark = "";
    if (DATA.annotation) {
      const anno = annoGet(s);
      const m = ANNO_MARK[anno.label];
      mark = (m ? "<span class=\"anno-mark\" style=\"color:" + m[1] + "\" title=\"" +
              anno.label + "\">" + m[0] + "</span>" : "") +
             (anno.grotte ? "<span class=\"anno-mark\" style=\"color:var(--accent)\" " +
              "title=\"grotte\">G</span>" : "");
    }
    return "<div class=\"rank-row" + (i === state.i ? " active" : "") +
      "\" data-i=\"" + i + "\"><div class=\"who\"><b>" + s.player + "</b><span>" +
      fmtTime(s.t0) + " → " + fmtTime(s.t1) + " · " + s.world +
      "</span></div>" + mark + "<span class=\"chip\" style=\"color:" + c +
      ";background:color-mix(in srgb, " + c + " 16%, transparent)\">" +
      (val === null || val === undefined ? "—" : val) + "</span></div>";
  }).join("") ||
    "<div class=\"footnote\">" + (q
      ? "Aucun joueur ne correspond à « " + state.rankQuery + " »."
      : "Aucune session analysable pour ce minerai dans ce monde.") + "</div>";
  for (const row of el("rank").querySelectorAll(".rank-row")) {
    row.addEventListener("click", () => {
      const i = Number(row.dataset.i);
      el("session-select").value = i;
      selectSession(i);
    });
  }
}

// --- Vue d'ensemble : histogramme des scores + nuage V1 x anomalie ---
// Couleurs concretes (les var(--x) de VERDICT_STYLE ne marchent pas dans un canvas).
const VERDICT_HEX = {
  "fortement suspect": "#d03b3b", "a surveiller": "#fab219",
  "RAS": "#0ca30c", "indeterminable": "#67707f",
};
const OV_ORDER = ["fortement suspect", "a surveiller", "indeterminable", "RAS"];
let ovPoints = [];
let ovOre = null; // minerai de la vue d'ensemble, independant du panneau lateral

function ovTip(html, x, y) {
  const tip = el("ov-tip");
  tip.innerHTML = html;
  tip.style.display = "block";
  tip.style.left = Math.min(x + 14, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.max(y - tip.offsetHeight - 12, 8) + "px";
}
function ovTipHide() { el("ov-tip").style.display = "none"; }

function renderOverview() {
  if (!ovOre || !DATA.presentFamilies.includes(ovOre)) ovOre = state.target;
  const rows = DATA.sessions.map((s, i) => ({ i, s, a: s.analysis[ovOre] || {} }))
    .filter(r => inWorld(r.s) && (ovOre in r.s.analysis));
  const scored = rows.filter(r => r.a.score !== null && r.a.score !== undefined);
  const counts = {};
  for (const r of rows) {
    const v = r.a.verdict || "indeterminable";
    counts[v] = (counts[v] || 0) + 1;
  }
  el("ov-summary").textContent = rows.length + " sessions analysées" +
    (state.world ? " dans " + state.world : "") + ", minerai surveillé : " +
    FAMILY_LABEL[ovOre].toLowerCase() + " (score sur la session entière, indépendant " +
    "de la fenêtre temporelle).";
  el("ov-legend").innerHTML = OV_ORDER.filter(v => counts[v]).map(v =>
    "<span><span class=\"sw\" style=\"background:" + VERDICT_HEX[v] + "\"></span>" +
    VERDICT_STYLE[v][1] + " <b>" + counts[v] + "</b></span>").join("");

  // Histogramme (echelle log, tranches de 5 points)
  const bins = new Array(20).fill(0);
  for (const r of scored) bins[Math.min(19, Math.floor(r.a.score / 5))]++;
  const maxN = Math.max(1, ...bins);
  const M = { l: 44, r: 8, t: 14, b: 34 }, W = 520, H = 300;
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const yLog = (n) => n <= 0 ? 0 : Math.log10(n + 1) / Math.log10(maxN + 1);
  let svg = "";
  for (const g of [1, 10, 100, 1000, 10000]) {
    if (g > maxN) break;
    const y = M.t + ih * (1 - yLog(g));
    svg += "<line x1=\"" + M.l + "\" y1=\"" + y + "\" x2=\"" + (W - M.r) +
      "\" y2=\"" + y + "\" stroke=\"#2a2f38\"/>" +
      "<text x=\"" + (M.l - 7) + "\" y=\"" + (y + 3.5) +
      "\" text-anchor=\"end\" font-size=\"10\" fill=\"#67707f\">" + g + "</text>";
  }
  const bw = iw / 20;
  bins.forEach((n, i) => {
    const h = ih * yLog(n);
    const x = M.l + i * bw + 1.5, y = M.t + ih - h;
    svg += "<rect data-bin=\"" + i + "\" data-n=\"" + n + "\" x=\"" + x.toFixed(1) + "\" y=\"" + y.toFixed(1) +
      "\" width=\"" + (bw - 3).toFixed(1) + "\" height=\"" + h.toFixed(1) +
      "\" rx=\"3\" fill=\"#35C7DB\"/>";
    if (n > 0) svg += "<text x=\"" + (x + (bw - 3) / 2).toFixed(1) + "\" y=\"" +
      (y - 4).toFixed(1) + "\" text-anchor=\"middle\" font-size=\"9\" " +
      "fill=\"#9aa3b2\">" + n + "</text>";
  });
  for (const v of [0, 25, 50, 75, 100]) {
    svg += "<text x=\"" + (M.l + iw * v / 100) + "\" y=\"" + (H - 14) +
      "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#67707f\">" + v + "</text>";
  }
  svg += "<text x=\"" + (M.l + iw / 2) + "\" y=\"" + (H - 2) +
    "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#67707f\">score de suspicion V1</text>";
  el("ov-histo").innerHTML = svg;

  // Nuage V1 x anomalie (canvas ; RAS en semi-transparent sous les autres verdicts)
  const cv = el("ov-canvas"), ctx = cv.getContext("2d");
  const CM = { l: 58, r: 14, t: 14, b: 54 }, CW = cv.width, CH = cv.height;
  const cw = CW - CM.l - CM.r, ch = CH - CM.t - CM.b;
  const X = (v) => CM.l + cw * v / 100, Y = (v) => CM.t + ch * (1 - v / 100);
  ctx.clearRect(0, 0, CW, CH);
  ctx.strokeStyle = "#2a2f38"; ctx.lineWidth = 1;
  ctx.font = "18px system-ui, sans-serif"; ctx.fillStyle = "#67707f";
  for (const v of [0, 25, 50, 75, 100]) {
    ctx.beginPath(); ctx.moveTo(X(v), CM.t); ctx.lineTo(X(v), CM.t + ch); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CM.l, Y(v)); ctx.lineTo(CM.l + cw, Y(v)); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(v, X(v), CH - 28);
    ctx.textAlign = "right"; ctx.fillText(v, CM.l - 8, Y(v) + 6);
  }
  ctx.textAlign = "center";
  ctx.fillText("score de suspicion V1", CM.l + cw / 2, CH - 6);
  ctx.save(); ctx.translate(14, CM.t + ch / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("écart au corpus (0-100)", 0, 0); ctx.restore();

  ovPoints = scored
    .filter(r => r.a.anomaly_score !== null && r.a.anomaly_score !== undefined)
    .map(r => ({ i: r.i, s: r.s, a: r.a,
                 px: X(r.a.score), py: Y(r.a.anomaly_score),
                 v: r.a.verdict || "indeterminable" }));
  if (!ovPoints.length) {
    ctx.textAlign = "center"; ctx.fillStyle = "#9aa3b2";
    ctx.fillText("Pas de modèle d'anomalie pour " + FAMILY_LABEL[ovOre].toLowerCase(),
                 CM.l + cw / 2, CM.t + ch / 2);
    return;
  }
  for (const v of [...OV_ORDER].reverse()) {
    ctx.fillStyle = VERDICT_HEX[v];
    ctx.globalAlpha = v === "RAS" ? 0.3 : 0.9;
    for (const p of ovPoints) {
      if (p.v !== v) continue;
      ctx.beginPath();
      ctx.arc(p.px, p.py, v === "RAS" ? 4 : 5.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function ovNearest(e) {
  const cv = el("ov-canvas"), r = cv.getBoundingClientRect();
  const mx = (e.clientX - r.left) * cv.width / r.width;
  const my = (e.clientY - r.top) * cv.height / r.height;
  let best = null, bd = 16 * 16;
  for (const p of ovPoints) {
    const dx = p.px - mx, dy = p.py - my, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

function initOverview() {
  const oreSel = el("ov-ore");
  for (const key of DATA.presentFamilies) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = FAMILY_LABEL[key];
    oreSel.appendChild(opt);
  }
  oreSel.addEventListener("change", () => { ovOre = oreSel.value; renderOverview(); });
  const open = () => {
    ovOre = ovOre || state.target;
    oreSel.value = ovOre;
    renderOverview();
    el("overview").classList.add("open");
  };
  const close = () => { el("overview").classList.remove("open"); ovTipHide(); };
  el("open-overview").addEventListener("click", open);
  el("overview-close").addEventListener("click", close);
  el("overview").addEventListener("click", (e) => {
    if (e.target === el("overview")) close();
  });
  el("ov-histo").addEventListener("mousemove", (e) => {
    const rect = e.target.closest("rect[data-bin]");
    if (!rect) { ovTipHide(); return; }
    const b = Number(rect.dataset.bin);
    ovTip("<b>Score " + b * 5 + "–" + (b * 5 + 5) + "</b><br>" + rect.dataset.n + " sessions",
          e.clientX, e.clientY);
  });
  el("ov-histo").addEventListener("mouseleave", ovTipHide);
  el("ov-canvas").addEventListener("mousemove", (e) => {
    const p = ovNearest(e);
    if (!p) { ovTipHide(); return; }
    ovTip("<b>" + p.s.player + "</b> · " + p.s.date + "<br>V1 " + p.a.score +
      " · écart " + p.a.anomaly_score + "<br><span style=\"color:" + VERDICT_HEX[p.v] +
      "\">●</span> " + VERDICT_STYLE[p.v][1], e.clientX, e.clientY);
  });
  el("ov-canvas").addEventListener("mouseleave", ovTipHide);
  el("ov-canvas").addEventListener("click", (e) => {
    const p = ovNearest(e);
    if (!p) return;
    close();
    el("session-select").value = p.i;
    selectSession(p.i);
  });
}

let rafPending = false;
function scheduleRender() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; render(); });
}

function syncControls() {
  const S = DATA.sessions[state.i];
  el("slider-start").value = state.tA;
  el("slider-end").value = state.tB;
  el("time-start").value = fmtTime(state.tA);
  el("time-end").value = fmtTime(state.tB);
  const span = Math.max(S.t1 - S.t0, 1);
  el("range-fill").style.left = (100 * (state.tA - S.t0) / span) + "%";
  el("range-fill").style.width = (100 * (state.tB - state.tA) / span) + "%";
  // Quand les deux poignees sont collees a droite, celle de debut doit rester saisissable
  el("slider-start").style.zIndex = state.tA > (S.t0 + S.t1) / 2 ? 5 : 3;
}

function selectSession(i) {
  state.i = i;
  const S = DATA.sessions[i];
  state.tA = S.t0;
  state.tB = S.t1;
  for (const id of ["slider-start", "slider-end"]) {
    el(id).min = S.t0; el(id).max = S.t1; el(id).step = 1;
  }
  const mins = Math.round((S.t1 - S.t0) / 60);
  el("session-info").innerHTML =
    "<strong>" + S.player + "</strong> - " + S.world + " - " + S.date +
    " - " + fmtTime(S.t0) + " -> " + fmtTime(S.t1) + " UTC (" + mins + " min)";
  syncControls();
  render();
  renderPanel();
  renderLocation(S);
  renderAnnotation();
}

function tpTarget(S) {
  // Milieu du parcours : un bloc casse au coeur de la galerie (donc de l'air),
  // ou l'on peut arriver sans etre dans la roche.
  const mid = Math.floor(S.x.length / 2);
  return { x: S.x[mid], y: S.y[mid], z: S.z[mid] };
}

function renderLocation(S) {
  const p = tpTarget(S);
  el("loc-world").textContent = S.world;
  el("loc-coords").textContent = "X " + p.x + " · Y " + p.y + " · Z " + p.z;
  const btn = el("copy-tp");
  btn.classList.remove("copied");
  btn.textContent = "Copier /tp";
}

// Copie dans le presse-papier avec retour visuel sur le bouton, et prompt en secours.
function copyWithFeedback(btnId, text, doneLabel, idleLabel) {
  const done = () => {
    const btn = el(btnId);
    btn.classList.add("copied");
    btn.textContent = doneLabel;
    setTimeout(() => { btn.classList.remove("copied"); btn.textContent = idleLabel; }, 1600);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, () => window.prompt("A copier :", text));
  } else {
    window.prompt("A copier :", text);
  }
}

// --- Mode annotation (page generee avec --annotation) ---
// Verdict exclusif legit / suspect / triche + tag "grotte" cumulable, persistes en
// localStorage (cle stable pseudo|monde|debut, independante des session_id).
const ANNO_PREFIX = "xray-anno::";
const isoTs = (ts) => new Date(ts * 1000).toISOString().replace(/\.\d{3}Z$/, "Z");
const annoKey = (S) => ANNO_PREFIX + S.player + "|" + S.world + "|" + S.t0;

function annoGet(S) {
  try { return JSON.parse(localStorage.getItem(annoKey(S))) || {}; }
  catch { return {}; }
}

function annoSet(S, anno) {
  if (!anno.label && !anno.grotte) {
    localStorage.removeItem(annoKey(S));
  } else {
    localStorage.setItem(annoKey(S), JSON.stringify({
      player: S.player, world: S.world, t0: S.t0, t1: S.t1,
      label: anno.label || "", grotte: !!anno.grotte,
    }));
  }
}

function annoCount() {
  let n = 0;
  for (let i = 0; i < localStorage.length; i++) {
    if (localStorage.key(i).startsWith(ANNO_PREFIX)) n++;
  }
  return n;
}

function renderAnnotation() {
  if (!DATA.annotation) return;
  const anno = annoGet(DATA.sessions[state.i]);
  for (const btn of document.querySelectorAll("#annotation-section [data-v]")) {
    btn.classList.toggle("on", anno.label === btn.dataset.v);
  }
  el("anno-grotte").classList.toggle("on", !!anno.grotte);
  el("anno-export").textContent = "Exporter CSV (" + annoCount() + " annotées)";
}

function annoExport() {
  const lines = ["pseudo,world,start_utc,end_utc,label,tags"];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key.startsWith(ANNO_PREFIX)) continue;
    try {
      const a = JSON.parse(localStorage.getItem(key));
      lines.push([a.player, a.world, isoTs(a.t0), isoTs(a.t1), a.label || "",
                  a.grotte ? "grotte" : ""].join(","));
    } catch {}
  }
  const csv = lines.join("\n") + "\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "session_labels_export.csv";
  a.click();
  URL.revokeObjectURL(a.href);
  copyWithFeedback("anno-export", csv, "Exporté + copié !",
                   "Exporter CSV (" + (lines.length - 1) + " annotées)");
}

function initAnnotation() {
  if (!DATA.annotation) return;
  el("annotation-section").hidden = false;
  for (const btn of document.querySelectorAll("#annotation-section [data-v]")) {
    btn.addEventListener("click", () => {
      const S = DATA.sessions[state.i];
      const anno = annoGet(S);
      anno.label = anno.label === btn.dataset.v ? "" : btn.dataset.v;
      annoSet(S, anno);
      renderAnnotation();
      renderRank();
    });
  }
  el("anno-grotte").addEventListener("click", () => {
    const S = DATA.sessions[state.i];
    const anno = annoGet(S);
    anno.grotte = !anno.grotte;
    annoSet(S, anno);
    renderAnnotation();
    renderRank();
  });
  el("anno-export").addEventListener("click", annoExport);
}

function copyTpCommand() {
  const S = DATA.sessions[state.i];
  const p = tpTarget(S);
  copyWithFeedback("copy-tp", "/tppos @p " + p.x + " " + p.y + " " + p.z,
                   "Copié !", "Copier /tp");
}

function rebuildSessionSelect() {
  const sel = el("session-select");
  sel.innerHTML = "";
  const groups = new Map();
  DATA.sessions.forEach((S, i) => {
    if (!inWorld(S)) return;
    if (!groups.has(S.player)) {
      const og = document.createElement("optgroup");
      og.label = S.player;
      groups.set(S.player, og);
      sel.appendChild(og);
    }
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = S.label;
    groups.get(S.player).appendChild(opt);
  });
}

function initControls() {
  el("copy-tp").addEventListener("click", copyTpCommand);
  const sel = el("session-select");
  rebuildSessionSelect();
  sel.addEventListener("change", () => selectSession(Number(sel.value)));

  const worlds = [...new Set(DATA.sessions.map((S) => S.world))].sort();
  const wsel = el("world-select");
  const allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "Tous les mondes";
  wsel.appendChild(allOpt);
  for (const w of worlds) {
    const opt = document.createElement("option");
    opt.value = w;
    opt.textContent = w;
    wsel.appendChild(opt);
  }
  wsel.addEventListener("change", () => {
    state.world = wsel.value;
    rebuildSessionSelect();
    if (!inWorld(DATA.sessions[state.i])) {
      const first = DATA.sessions.findIndex(inWorld);
      sel.value = first;
      selectSession(first);
    } else {
      sel.value = state.i;
      renderRank();
    }
  });

  const ore = el("ore-select");
  for (const key of DATA.presentFamilies) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = FAMILY_LABEL[key];
    ore.appendChild(opt);
  }
  ore.value = state.target;
  ore.addEventListener("change", () => {
    state.target = ore.value;
    render();
    renderPanel();
  });

  el("rank-search").addEventListener("input", (e) => {
    state.rankQuery = e.target.value;
    renderRank();
  });
  el("rank-sort").addEventListener("change", (e) => {
    state.rankSort = e.target.value;
    renderRank();
  });

  el("toggle-panel").addEventListener("click", () => {
    el("panel").classList.toggle("hidden");
    Plotly.Plots.resize(plotDiv);
  });

  el("open-metrics").addEventListener("click", () => el("modal").classList.add("open"));
  el("modal-close").addEventListener("click", () => el("modal").classList.remove("open"));
  el("modal").addEventListener("click", (e) => {
    if (e.target === el("modal")) el("modal").classList.remove("open");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      el("modal").classList.remove("open");
      el("overview").classList.remove("open");
      ovTipHide();
    }
  });
}

el("slider-start").addEventListener("input", (e) => {
  state.tA = Math.min(Number(e.target.value), state.tB);
  syncControls(); scheduleRender();
});
el("slider-end").addEventListener("input", (e) => {
  state.tB = Math.max(Number(e.target.value), state.tA);
  syncControls(); scheduleRender();
});
el("time-start").addEventListener("change", (e) => {
  const ts = parseTimeInput(e.target.value, DATA.sessions[state.i]);
  if (ts !== null) state.tA = Math.min(ts, state.tB);
  syncControls(); render();
});
el("time-end").addEventListener("change", (e) => {
  const ts = parseTimeInput(e.target.value, DATA.sessions[state.i]);
  if (ts !== null) state.tB = Math.max(ts, state.tA);
  syncControls(); render();
});
el("reset-range").addEventListener("click", () => {
  const S = DATA.sessions[state.i];
  state.tA = S.t0; state.tB = S.t1;
  syncControls(); render();
});

initControls();
initOverview();
initAnnotation();
selectSession(0);
</script>
</body>
</html>
"""


def write_html(payload: dict, output: Path) -> None:
    from plotly.offline import get_plotlyjs

    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("/*__PLOTLY_JS__*/", get_plotlyjs())
    html = html.replace("/*__DATA_JSON__*/", data_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Base CoreProtect SQLite (defaut : {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--window",
        choices=["all", "last-12h", "last-24h", "yesterday"],
        default="all",
        help="Fenetre temporelle relative a charger (defaut : toute la base).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Debut UTC ISO 8601 (ex: 2026-07-18T00:00:00Z).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Fin UTC ISO 8601 (ex: 2026-07-19T00:00:00Z).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=300,
        help="Trou temporel en secondes qui coupe une session (defaut : 300).",
    )
    parser.add_argument(
        "--min-blocks",
        type=int,
        default=50,
        help="Nombre minimal de blocs pour garder une session (defaut : 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Fichier HTML de sortie (defaut : {DEFAULT_OUTPUT}, suffixe _anon si --anonymize).",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Remplace les pseudos par des pseudos inventes (partage public).",
    )
    parser.add_argument(
        "--annotation",
        action="store_true",
        help="Ajoute le mode annotation : verdict legit/suspect/triche + tag grotte "
             "par session, persiste dans le navigateur, export CSV vers data/labels/.",
    )
    parser.add_argument(
      "--include-cave-sessions",
      action="store_true",
      help="Garde aussi les sessions qui ressemblent a des cavernes / geodes naturelles.",
    )
    parser.add_argument(
        "--include-surface-sessions",
        action="store_true",
        help="Garde aussi les sessions dominees par la recolte de surface "
             "(bois, sable, gres).",
    )
    parser.add_argument(
        "--split",
        choices=["monthly"],
        default=None,
        help="Genere une page par mois calendaire UTC au lieu d'une page unique "
             "(demande --start et --end ; suffixe _AAAA-MM sur chaque fichier). "
             "Indispensable sur une longue periode : une page unique de plusieurs "
             "centaines de Mo ne charge pas dans un navigateur.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        suffix = "_anon" if args.anonymize else ""
        args.output = DEFAULT_OUTPUT.with_stem(DEFAULT_OUTPUT.stem + suffix)
    return args


def month_ranges(start_ts: int, end_ts: int) -> list[tuple[int, int, str]]:
    """Decoupe [start_ts, end_ts] en mois calendaires UTC : (debut, fin, "AAAA-MM").

    Les bornes de fin sont exclusives d'une seconde pour ne pas compter deux fois
    l'evenement pile sur la frontiere (load_breaks filtre en time <= end).
    """
    ranges = []
    cur = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    while cur < end:
        nxt = (cur.replace(year=cur.year + 1, month=1) if cur.month == 12
               else cur.replace(month=cur.month + 1))
        nxt = nxt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        ranges.append((int(cur.timestamp()),
                       min(int(nxt.timestamp()), end_ts) - 1,
                       cur.strftime("%Y-%m")))
        cur = nxt
    return ranges


def render_one(
    args: argparse.Namespace,
    start_ts: int | None,
    end_ts: int | None,
    output: Path,
) -> int:
    """Extrait, analyse et ecrit une page pour la fenetre donnee.

    Retourne le nombre de sessions rendues (0 = aucune page ecrite).
    """
    t0 = time.perf_counter()
    df, worlds = load_breaks(args.db, start_ts=start_ts, end_ts=end_ts)
    t_extract = time.perf_counter() - t0
    print(
        f"{len(df)} blocs casses par {df['pseudo'].nunique()} joueurs "
        f"charges depuis {args.db.name} en {t_extract:.1f} s"
    )
    if df.empty:
        print("Aucun evenement dans la fenetre temporelle demandee.")
        return 0

    if start_ts is not None or end_ts is not None:
        window_start = start_ts if start_ts is not None else int(df["time"].min())
        window_end = end_ts if end_ts is not None else int(df["time"].max())
        print(f"Fenetre temporelle: {fmt_date(window_start)} -> {fmt_date(window_end)}")
    else:
        print("Fenetre temporelle: toute la base")

    if args.anonymize:
        df, mapping = anonymize_players(df)
        print("Anonymisation (mapping console uniquement, absent du HTML) :")
        for real, anon in mapping.items():
            print(f"  {real} -> {anon}")

    df, dropped = segment_sessions(df, gap_seconds=args.gap, min_blocks=args.min_blocks)
    if dropped:
        print(f"Sessions ignorees (< {args.min_blocks} blocs) : {dropped}")
    df, end_dropped = filter_end_world_sessions(df, worlds)
    if end_dropped:
        print(f"Sessions exclues car minees dans l'End (aucun minerai) : {end_dropped}")
    if not args.include_cave_sessions:
      df, cave_dropped = filter_cave_like_sessions(df)
      if cave_dropped:
        print(f"Sessions exclues car ressemblant a des grottes/geodes : {cave_dropped}")
    if not args.include_surface_sessions:
        df, surface_dropped = filter_surface_gathering_sessions(df)
        if surface_dropped:
            print(f"Sessions exclues car recolte de surface (bois/sable/gres) : "
                  f"{surface_dropped}")
    if df.empty:
        print("Aucune session retenue avec ces seuils.")
        return 0

    t1 = time.perf_counter()
    payload = build_payload(df, worlds)
    payload["annotation"] = bool(args.annotation)
    write_html(payload, output)
    t_render = time.perf_counter() - t1

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Rendu ecrit : {output} ({size_mb:.1f} Mo, "
          f"{len(payload['sessions'])} sessions)")
    print(f"Temps : extraction {t_extract:.1f} s - analyse et rendu {t_render:.1f} s "
          f"- total {time.perf_counter() - t0:.1f} s")
    return len(payload["sessions"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.db.exists():
        raise SystemExit(f"Base introuvable : {args.db}")

    start_ts, end_ts = requested_time_window(args)

    if args.split == "monthly":
        if start_ts is None or end_ts is None:
            raise SystemExit("--split monthly demande une fenetre bornee : "
                             "--start ET --end (ou --window relative).")
        if args.anonymize:
            print("Attention : avec --split, le mapping d'anonymisation est "
                  "recalcule par mois — un meme joueur peut changer de pseudo "
                  "d'une page a l'autre.")
        pages = 0
        for m_start, m_end, tag in month_ranges(start_ts, end_ts):
            print(f"\n=== {tag} ===")
            output = args.output.with_stem(args.output.stem + "_" + tag)
            if render_one(args, m_start, m_end, output):
                pages += 1
        if not pages:
            raise SystemExit("Aucune page generee sur la periode demandee.")
        print(f"\n{pages} page(s) generee(s).")
        return 0

    if not render_one(args, start_ts, end_ts, args.output):
        raise SystemExit("Aucune session retenue dans la fenetre demandee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
