"""Decoupe une page mining_sessions_3d deja generee en pages mensuelles.

Recuperation sans recalcul : la page embarque tout le payload (sessions,
features, scores) dans `const DATA = {...};`. On relit ce JSON session par
session (parsing incremental, jamais tout en memoire d'un coup), on regroupe
par mois UTC du debut de session, et on reecrit une page par mois avec le
meme habillage (Plotly, template, options annotation) que l'original.

A utiliser quand une generation longue a produit une page trop grosse pour un
navigateur ; pour les prochaines generations, prefererer directement
scripts/render_mining_3d.py --split monthly.

Usage:
    python scripts/split_mining_page.py reports/figures/ma_grosse_page.html
    -> ecrit ma_grosse_page_2026-01.html, ma_grosse_page_2026-02.html, ...
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_MARKER = "const DATA = "
SESSIONS_PREFIX = '{"sessions":['


# Format du tag temporel par granularite (semaine : annee-semaine ISO).
TAG_FORMATS = {"month": "%Y-%m", "week": "%G-W%V", "day": "%Y-%m-%d"}


def time_tag(t0: int, by: str) -> str:
    return datetime.fromtimestamp(t0, tz=timezone.utc).strftime(TAG_FORMATS[by])


def split_page(
    source: Path, by: str = "month", drop_players: set[str] | None = None
) -> list[Path]:
    t0 = time.perf_counter()
    text = source.read_text(encoding="utf-8")
    print(f"Page lue : {source} ({len(text) / 1e6:.0f} Mo)")

    # Le JSON est compact (aucun retour ligne) : la ligne "const DATA = {...};"
    # se termine au premier \n qui suit le marqueur.
    idx = text.index(DATA_MARKER) + len(DATA_MARKER)
    end = text.index("\n", idx)
    if text[end - 1] != ";":
        raise SystemExit("Structure inattendue : la ligne DATA ne finit pas par ';'.")
    prefix, payload, suffix = text[:idx], text[idx:end - 1], text[end - 1:]
    del text

    if not payload.startswith(SESSIONS_PREFIX):
        raise SystemExit('Payload inattendu : il ne commence pas par {"sessions":[.')

    # Parcours incremental du tableau sessions : un objet a la fois, en gardant
    # son texte brut (aucune re-serialisation, sortie identique a l'original).
    decoder = json.JSONDecoder()
    pos = len(SESSIONS_PREFIX)
    by_month: dict[str, list[str]] = {}
    n = dropped = 0
    while payload[pos] != "]":
        session, pos_end = decoder.raw_decode(payload, pos)
        if drop_players and session["player"] in drop_players:
            dropped += 1
        else:
            by_month.setdefault(time_tag(session["t0"], by), []).append(payload[pos:pos_end])
            n += 1
        pos = pos_end
        if payload[pos] == ",":
            pos += 1
    tail = payload[pos + 1:]  # ,"materials":...} — metadonnees communes a tous les mois
    del payload
    unit = {"month": "mois", "week": "semaine(s)", "day": "jour(s)"}[by]
    if dropped:
        print(f"Sessions ecartees ({', '.join(sorted(drop_players))}) : {dropped}")
    print(f"{n} sessions reparties sur {len(by_month)} {unit} "
          f"(analyse relue, rien n'est recalcule)")

    outputs = []
    for tag in sorted(by_month):
        out = source.with_stem(source.stem + "_" + tag)
        month_json = SESSIONS_PREFIX + ",".join(by_month[tag]) + "]" + tail
        out.write_text(prefix + month_json + suffix, encoding="utf-8")
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  {tag} : {len(by_month[tag])} sessions -> {out.name} ({size_mb:.0f} Mo)")
        outputs.append(out)

    print(f"Temps total : {time.perf_counter() - t0:.1f} s")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Page HTML generee par render_mining_3d.py.")
    parser.add_argument("--by", choices=sorted(TAG_FORMATS), default="month",
                        help="Granularite du decoupage (defaut : month).")
    parser.add_argument("--drop-player", action="append", default=[],
                        help="Pseudo a ecarter des pages produites (repetable). "
                             "Utile pour un compte machine qui noie tout le reste.")
    args = parser.parse_args(argv)
    if not args.source.exists():
        raise SystemExit(f"Page introuvable : {args.source}")
    split_page(args.source, by=args.by, drop_players=set(args.drop_player) or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
