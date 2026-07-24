"""Smoke test de l'API (api/main.py) : l'app boote et /health repond.

On ne teste ni /sync ni /report de bout en bout (ils exigent une vraie
passerelle) : le but est de verifier le cablage sans reseau — imports, cycle de
vie (lifespan), chargement optionnel des modeles — et deux garde-fous HTTP
(/health -> 200, minerai inconnu -> 400).

api/ n'est pas sur le pythonpath du projet (main.py n'est pas un package) : on
l'ajoute localement au sys.path du test, et on fournit GATEWAY_TOKEN car main.py
leve RuntimeError a l'import s'il manque.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# API deps declarees dans l'extra [dev] ; si absentes, on skip sans casser la suite.
pytest.importorskip("fastapi", reason="pip install -e .[dev]")
pytest.importorskip("httpx", reason="httpx requis par TestClient : pip install -e .[dev]")

from fastapi.testclient import TestClient  # noqa: E402

API_DIR = Path(__file__).resolve().parents[1] / "api"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GATEWAY_TOKEN", "test-token")
    monkeypatch.delenv("GATEWAY_URL", raising=False)  # pas de passerelle en test
    monkeypatch.setenv("MIRROR_PATH", str(tmp_path / "mirror.db"))
    monkeypatch.syspath_prepend(str(API_DIR))
    sys.modules.pop("main", None)  # import frais a chaque test
    import main

    with TestClient(main.app) as c:  # le with declenche startup/shutdown du lifespan
        yield c
    sys.modules.pop("main", None)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["gateway_configured"] is False  # GATEWAY_URL absent en test
    assert body["mirror_exists"] is False  # aucune synchro effectuee
    assert isinstance(body["models_loaded"], list)  # api/models absent -> []


def test_report_unknown_ore_is_rejected(client):
    # Le controle du minerai est fait AVANT toute tentative de synchro : pas de
    # passerelle requise pour ce chemin d'erreur.
    resp = client.get("/report", params={"ore": "obsidian", "sync_first": "false"})
    assert resp.status_code == 400
