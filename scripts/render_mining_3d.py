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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from xray_detector.features import compute_session_features, score_session
from xray_detector.mining import (
    ORE_FAMILIES,
    anonymize_players,
    filter_cave_like_sessions,
    load_breaks,
    parse_utc_datetime,
    segment_sessions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "database_testserv.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "figures" / "mining_sessions_3d.html"

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


def build_payload(df: pd.DataFrame, worlds: dict[int, str]) -> dict:
    """Serialise sessions, analyse par minerai cible et palette pour le JS de la page."""
    materials = sorted(df["material"].unique())
    mat_index = {m: i for i, m in enumerate(materials)}
    families = [[key, label, FAMILY_COLORS[key]] for key, label in ORE_FAMILIES.items()]
    fam_index = {key: i for i, (key, _, _) in enumerate(families)}
    present = [key for key in ORE_FAMILIES if (df["ore"] == key).any()]

    sessions = []
    for (pseudo, wid, _sid), seg in df.groupby(["pseudo", "wid", "session_id"], sort=True):
        seg = seg.sort_values("time")
        t0, t1 = int(seg["time"].min()), int(seg["time"].max())
        n_ores = int(seg["ore"].notna().sum())

        analysis = {}
        for key in present:
            features = compute_session_features(seg, target=key)
            analysis[key] = _json_safe({**features, **score_session(features, target=key)})

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

  #modal {
    position: fixed; inset: 0; z-index: 50; display: none;
    align-items: center; justify-content: center;
    background: rgba(8, 10, 13, 0.62);
  }
  #modal.open { display: flex; }
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

  .tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .tile {
    background: var(--raised); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 10px;
  }
  .tile b {
    display: block; font-size: 14.5px; font-variant-numeric: tabular-nums;
  }
  .tile span { font-size: 10.5px; color: var(--ink-3); }

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
</style>
</head>
<body>
<header>
  <div class="app-title">Minage 3D <span class="dim">- reconstruction &amp; analyse</span></div>
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
      <div class="section-title">Details de la session</div>
      <div class="tiles" id="tiles"></div>
    </div>
    <div>
      <div class="section-title">Classement des sessions</div>
      <div class="rank" id="rank"></div>
    </div>
    <div class="footnote">
      Score heuristique V1, calcule sur la session entiere (la fenetre temporelle
      filtre la scene 3D, pas l'analyse). Voir readmeAnalyse.md pour la methode.
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
  </div>
</div>

<script>/*__PLOTLY_JS__*/</script>
<script>
"use strict";
const DATA = /*__DATA_JSON__*/;

const el = (id) => document.getElementById(id);
const plotDiv = el("plot");
const state = { i: 0, tA: 0, tB: 0, target: "diamond", hidden: new Set() };
if (!DATA.presentFamilies.includes(state.target)) state.target = DATA.presentFamilies[0];

const FAMILY_LABEL = {}, FAMILY_INDEX = {};
DATA.families.forEach((f, i) => { FAMILY_LABEL[f[0]] = f[1]; FAMILY_INDEX[f[0]] = i; });

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
  const A = S.analysis[state.target] || {};
  const [color, verdictText] = VERDICT_STYLE[A.verdict] || VERDICT_STYLE["indeterminable"];
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

  const order = DATA.sessions.map((s, i) => [i, (s.analysis[state.target] || {}).score])
    .sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1));
  el("rank").innerHTML = order.map(([i, sc]) => {
    const s = DATA.sessions[i];
    const v = (s.analysis[state.target] || {}).verdict;
    const [c] = VERDICT_STYLE[v] || VERDICT_STYLE["indeterminable"];
    return "<div class=\"rank-row" + (i === state.i ? " active" : "") +
      "\" data-i=\"" + i + "\"><div class=\"who\"><b>" + s.player + "</b><span>" +
      fmtTime(s.t0) + " → " + fmtTime(s.t1) + " · " + s.world +
      "</span></div><span class=\"chip\" style=\"color:" + c +
      ";background:color-mix(in srgb, " + c + " 16%, transparent)\">" +
      (sc === null || sc === undefined ? "—" : sc) + "</span></div>";
  }).join("");
  for (const row of el("rank").querySelectorAll(".rank-row")) {
    row.addEventListener("click", () => {
      const i = Number(row.dataset.i);
      el("session-select").value = i;
      selectSession(i);
    });
  }
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
}

function initControls() {
  const sel = el("session-select");
  const groups = new Map();
  DATA.sessions.forEach((S, i) => {
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
  sel.addEventListener("change", () => selectSession(Number(sel.value)));

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
    if (e.key === "Escape") el("modal").classList.remove("open");
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
      "--include-cave-sessions",
      action="store_true",
      help="Garde aussi les sessions qui ressemblent a des cavernes / geodes naturelles.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        suffix = "_anon" if args.anonymize else ""
        args.output = DEFAULT_OUTPUT.with_stem(DEFAULT_OUTPUT.stem + suffix)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.db.exists():
        raise SystemExit(f"Base introuvable : {args.db}")

    start_ts, end_ts = requested_time_window(args)
    df, worlds = load_breaks(args.db, start_ts=start_ts, end_ts=end_ts)
    print(
        f"{len(df)} blocs casses par {df['pseudo'].nunique()} joueurs "
        f"charges depuis {args.db.name}"
    )

    if start_ts is not None or end_ts is not None:
        if df.empty:
            raise SystemExit("Aucun evenement dans la fenetre temporelle demandee.")
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
    if not args.include_cave_sessions:
      df, cave_dropped = filter_cave_like_sessions(df)
      if cave_dropped:
        print(f"Sessions exclues car ressemblant a des grottes/geodes : {cave_dropped}")
    if df.empty:
        raise SystemExit("Aucune session retenue avec ces seuils.")

    payload = build_payload(df, worlds)
    write_html(payload, args.output)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Rendu ecrit : {args.output} ({size_mb:.1f} Mo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
