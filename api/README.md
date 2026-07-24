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

**Aucune copie manuelle à faire.** Ce dossier ne duplique NI le pipeline
(`src/xray_detector/`, qui contient déjà `gateway_client.py`) NI les modèles
(`data/models/*.joblib`) : ils ont une source unique dans le repo, et le
Dockerfile va les chercher là. C'est pour ça qu'on **build depuis la racine du
repo**, avec `-f api/Dockerfile` (voir ci-dessous), et surtout pas depuis `api/`.

```
AntiCheat/                      <-- contexte de build (racine du repo)
├── .dockerignore               <-- garde le contexte léger (exclut data/raw ~102 Go)
├── src/xray_detector/          <-- source unique du pipeline + gateway_client.py
├── data/models/*.joblib        <-- source unique des modèles entraînés
└── api/
    ├── Dockerfile              COPY src/xray_detector, COPY data/models/*.joblib
    ├── docker-compose.yml
    ├── requirements.txt
    ├── main.py
    ├── .env.example
    └── README.md               (ce fichier)
```

Seule préparation nécessaire : le token de la passerelle.

```bash
cp api/.env.example api/.env   # puis y mettre le vrai gateway.token du plugin
```

## Build et déploiement

Depuis la **racine du repo** (le contexte de build doit voir `src/` et `data/`) :

```bash
docker build -f api/Dockerfile -t xrayindexer-api:0.2.0 .
# ou, en une fois via compose (context=.. déjà configuré) :
docker compose -f api/docker-compose.yml up -d --build

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

## Rendu 3D dans le dossier du plugin

Le renderer est un service Compose ponctuel, sans port HTTP. Il lit le miroir
en lecture seule et ecrit une page HTML autonome dans
`plugins/XRayGateway/figures/` du serveur Minecraft.

Dans `api/.env`, renseignez le chemin **absolu de l'hote Docker** :

```dotenv
XRAY_GATEWAY_FIGURES_DIR=E:/MinecraftServer/plugins/XRayGateway/figures
```

Puis, apres une synchronisation reussie :

```powershell
docker compose -f api/docker-compose.yml --profile render run --rm xrayindexer-renderer
```

La page est ecrite sous `mining_sessions_3d.html`. Des options du renderer
peuvent etre ajoutees a la commande, par exemple `--window last-24h` ou
`--anonymize`. Le rendu contient des pseudos et coordonnees lorsqu'il n'est pas
anonymise : ne le rendez pas public sans `--anonymize`.

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
