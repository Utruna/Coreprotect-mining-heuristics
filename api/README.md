# XRayGateway Analysis API (Docker)

**Pivot important par rapport à une version précédente de ce dossier** :
l'ancienne approche (un plugin Java exportait un CSV et le poussait vers
cette API) est abandonnée. Le vrai plugin — `xray-gateway-plugin`, déjà
codé et testé par vous — expose un serveur HTTP en lecture seule
(`/health`, `/blocks`, `/users`, `/materials`, `/worlds`) et
`gateway_client.py` sait déjà synchroniser un miroir SQLite local de façon
incrémentale. Cette API se contente d'orchestrer ce qui existe déjà :
synchroniser, puis lancer votre pipeline d'analyse réel sur le miroir.

Si vous avez un dossier `xray-indexer-plugin` (Java, avec `ReadOnlyCoreProtectDb`,
`LocalIndex`, `IndexBuilder`, `ReportExporter`, `ApiClient`) issu d'une session
précédente : **il est obsolète**, ne le déployez pas. `xray-gateway-plugin`
fait déjà ce travail.

## Architecture

```
[xray-gateway-plugin sur le serveur MC]
        | HTTP (Bearer token), /blocks?since=X&limit=Y (CSV gzip)
        v
[gateway_client.sync()] --> mirror.db (SQLite, volume persistant)
        |
        v
[xray_detector.mining.load_breaks(mirror.db)] -> segment_sessions
        -> filter_cave_like_sessions -> filter_end_world_sessions
        -> filter_surface_gathering_sessions
        -> compute_session_features -> score_session -> score_anomalies
        |
        v
   GET /report?ore=diamond&start=...&end=...   (JSON)
```

## Avant de builder

```
xrayindexer-api/
├── Dockerfile
├── requirements.txt
├── main.py
├── xray_detector/       <-- copié depuis src/xray_detector (déjà fait ici)
├── gateway_client.py    <-- copié (déjà fait ici)
└── models/               <-- À COPIER : anomaly_iforest_<ore>.joblib
```

```bash
cp /chemin/vers/votre/projet/data/models/anomaly_iforest_diamond.joblib ./models/
cp /chemin/vers/votre/projet/data/models/anomaly_iforest_ancient_debris.joblib ./models/
cp .env.example .env   # puis y mettre le vrai gateway.token du plugin
```

## Build et déploiement

```bash
docker build -t xrayindexer-api:0.2.0 .
docker compose up -d
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/sync
curl "http://127.0.0.1:8000/report?ore=diamond&start=2026-07-01&end=2026-07-08"
```

`GATEWAY_URL` dans `docker-compose.yml` pointe vers
`http://host.docker.internal:8787` — à ajuster si le plugin écoute sur un
autre port (voir sa config `gateway.port` ou équivalent). L'entrée
`extra_hosts` est **obligatoire sur un serveur Linux** (contrairement à
Docker Desktop Mac/Windows où `host.docker.internal` est résolu nativement).

## Endpoints

- `GET /health` — état du miroir, des modèles chargés, config gateway présente ou non.
- `POST /sync` — synchronise le miroir sans lancer d'analyse (utile en tâche planifiée).
- `GET /report?ore=diamond&start=...&end=...&sync_first=true` — synchronise (sauf si `sync_first=false`) puis renvoie le rapport JSON, trié par score décroissant.

## Ce qui est vérifié vs supposé — honnêteté avant tout

**Vérifié** : `main.py` appelle `gateway_client.sync`, `mining.load_breaks`,
`mining.segment_sessions`, les trois filtres, `features.compute_session_features`,
`features.score_session`, `anomaly_model.score_anomalies` avec les signatures
réelles telles qu'elles apparaissent dans les fichiers que vous avez fournis
dans cette conversation — pas des suppositions cette fois.

**Non couvert, à votre charge** :
- **`ORE_DIMENSIONS`** (dans `mining.py`) existe pour éviter de scorer un
  minerai impossible dans une dimension donnée (diamant au Nether, par
  exemple), mais je n'ai pas vu le code qui l'utilise réellement (probablement
  dans votre `scripts/analyze_mining_sessions.py` actuel, que je n'ai pas).
  `/report` ne fait donc **pas** ce filtrage croisé minerai/dimension —
  ajoutez-le si votre pipeline réel le fait.
- **Port et chemin réels de la passerelle** (`gateway.port`, préfixe
  d'URL éventuel) : à confirmer dans la config du plugin, je n'ai vu que le
  client Python, pas la config Java.
- **Non testé** : build Docker non exécuté dans ce sandbox (pas d'accès à
  PyPI complet pour ces images de base ni au réseau du gateway plugin
  depuis ici). À valider chez vous.

## Prochaine vérification suggérée

`POST /sync` une première fois (là où ça peut prendre du temps sur l'historique
complet), vérifier `GET /health` (`mirror_exists: true`), puis comparer un
`GET /report` avec ce que donnait votre script d'analyse existant sur la même
fenêtre, si vous en avez un qui tourne directement sur `database_testserv.db`.
