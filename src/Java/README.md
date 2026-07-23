# XRayGateway (plugin Paper/Spigot)

Passerelle HTTP **en lecture seule stricte** au-dessus de la base CoreProtect.
Le plugin ne fait qu'ouvrir une porte : il sert des tranches brutes de
`co_block` (paginées par `rowid`) et les tables de correspondance, en **CSV
gzippé**, pour un pipeline d'analyse x-ray **déporté** (Python sur un serveur de
calcul distant). Il n'écrit jamais dans le fichier CoreProtect.

## Pourquoi cette forme

Le calcul est déplacé hors du serveur de jeu (Docker/OVH) pour ne pas peser sur
les TPS. Le serveur de jeu ne fait que streamer des données ; toute la logique
d'analyse (filtres `action=0`, `uuid`, « posé-puis-recassé », scoring) vit côté
Python, sur un **miroir local** que le client reconstruit et tient à jour.

Trois faits vérifiés sur le schéma CoreProtect réel qui justifient le design :

1. **`co_block` n'a pas de colonne `id`** — la clé est le `rowid` implicite de
   SQLite. La passerelle pagine donc par `rowid`.
2. **`journal_mode = WAL`** — lecture concurrente sûre pendant que CoreProtect
   écrit, tant qu'on ouvre en lecture seule (c'est le cas).
3. **`time` est monotone avec `rowid`** (journal append-only) — un simple
   curseur `rowid > dernier_vu` suffit à ne transférer que les nouveautés,
   sans aucun index à construire.

Mesure de charge (base de test, schéma réel) : après filtrage `action=0 + uuid`,
une ligne cassée pèse **4,6 octets en CSV gzippé** (ratio ×11,5). Le backfill
initial ne se paie qu'une fois ; ensuite, seuls les deltas transitent. Les BLOBs
`meta`/`blockdata` (le gros des 120 Go) ne sont **jamais** envoyés.

## Endpoints

Tous exigent un en-tête `Authorization: Bearer <token>`.

```
GET /health                    -> { "status": "ok", "max_rowid": N }
GET /blocks?since=<rowid>&limit=<n>   -> co_block brut (CSV gzip, paginé par rowid)
                                  colonnes : cp_rowid,time,user,wid,x,y,z,type,action
GET /users                     -> co_user       (id, uuid, user)     CSV gzip
GET /materials                 -> co_material_map (id, material)      CSV gzip
GET /worlds                    -> co_world       (id, world)          CSV gzip
```

Le client suit le curseur : il repart du plus grand `cp_rowid` reçu jusqu'à
obtenir un lot vide, puis recharge les trois petites tables de correspondance.

## Configuration (`config.yml`)

- `gateway.token` : **obligatoire**. Le plugin refuse de démarrer tant qu'il
  vaut `CHANGE_ME`. C'est la seule protection de l'endpoint — gardez-le secret.
- `gateway.bind` : `127.0.0.1` par défaut (accès local, à exposer au serveur
  distant via tunnel SSH/VPN). `0.0.0.0` seulement derrière un firewall.
- `gateway.max-page-size` : borne le lot `/blocks` (mémoire/temps par requête).

## Organisation

Layout Maven standard :

```
src/Java/
  pom.xml
  src/main/java/net/utruna/xrayindexer/
    GatewayPlugin.java          cycle de vie du plugin, commande /xraygateway
    db/  ReadOnlyCoreProtectDb   ouverture lecture seule stricte (une connexion/requête)
    http/                        serveur HTTP et routes
      GatewayServer              création du serveur, enregistrement des routes
      AuthFilter                 jeton Bearer (comparaison à temps constant)
      BlocksHandler, MapHandler, HealthHandler   les endpoints
      QueryResponder, Csv, Query  utilitaires (streaming CSV gzip, parsing)
  src/main/resources/
    plugin.yml, config.yml
```

Seuls `ReadOnlyCoreProtectDb` et `GatewayServer` sont `public` (frontières entre
paquets) ; le reste du paquet `http` est package-private.

## Build

```bash
mvn clean package
```

Nécessite Maven Central + le repo Spigot (`hub.spigotmc.org`) — **pas testable
dans le sandbox de cette conversation** (accès réseau limité). À builder et
tester chez vous. Le jar shadé relocalise `org.sqlite` pour éviter tout conflit
avec le `sqlite-jdbc` déjà embarqué par CoreProtect dans la même JVM. Le serveur
HTTP utilise `com.sun.net.httpserver` (fourni par le JDK, aucune dépendance).

## Côté Python

Le client (`xray_detector.gateway_client`) reconstruit un miroir local :

```bash
python -m xray_detector.gateway_client --url http://127.0.0.1:8787 \
    --token <votre-token> --mirror data/raw/mirror.db
```

Puis le pipeline lit ce miroir comme une base CoreProtect classique
(`load_breaks(mirror.db)`), inchangé.

## Sécurité — à valider avant déploiement

1. Le jeton transite en clair : mettez la passerelle derrière **HTTPS** (reverse
   proxy) ou un **tunnel SSH/VPN** dès que l'accès n'est plus purement local.
2. Confirmer sur la base live : `PRAGMA table_info(co_block);`,
   `PRAGMA journal_mode;` (WAL attendu).
3. La colonne `rolled_back` de `co_block` est transmise telle quelle mais pas
   filtrée : à traiter côté Python si les rollbacks admin doivent être exclus.
