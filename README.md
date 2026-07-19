# AntiCheat X-Ray Detection

Base de projet pour analyser a posteriori les logs CoreProtect et produire un score de suspicion x-ray par joueur ou session de minage.

## Ce qui est déjà en place

- Environnement Python virtuel configuré dans `.venv`
- Structure `src/` prête pour le code du pipeline
- Dossiers de travail pour les données brutes, intermédiaires et traitées
- Entrée CLI minimale pour préparer le workspace et afficher la configuration
- Preview 3D interactive des sessions de minage — voir [readmePreview.md](readmePreview.md)
- Analyse statistique des trajectoires et score de suspicion V1 — voir [readmeAnalyse.md](readmeAnalyse.md)

## Démarrage rapide

1. Ouvrir le projet dans VS Code.
2. Vérifier que l'interpréteur utilisé est `.venv/Scripts/python.exe`.
3. Lancer la configuration du workspace:

```powershell
python -m xray_detector init
```

4. Afficher la configuration détectée:

```powershell
python -m xray_detector show-config
```

## Accès Docker à CoreProtect

Un mini environnement Docker est disponible pour exposer [data/raw/CoreProtect/database.db](data/raw/CoreProtect/database.db) dans PostgreSQL via `sqlite_fdw`.

```powershell
docker compose up --build
```

Une fois le conteneur démarré, les tables SQLite sont importées dans le schéma `coreprotect` de PostgreSQL. Tu peux ensuite t’y connecter sur le port `5433` avec `postgres / postgres`.

Paramètres de connexion à saisir dans ton client PostgreSQL:

- Hôte: `localhost`
- Port: `5433`
- Base de données: `coreprotect` ou laisser vide si ton client l’autorise
- Utilisateur: `postgres`
- Mot de passe: `postgres`

Ne mets pas `5432:5432` dans le champ port du client. Cette valeur sert seulement dans Docker pour faire correspondre le port du conteneur avec celui de la machine. Ici j’ai déplacé le service sur `5433` parce que `5432` est déjà pris sur Windows.

Si la base est très grosse, il vaut mieux importer les tables à la demande plutôt qu’au démarrage. Exemple:

```sql
IMPORT FOREIGN SCHEMA main
	LIMIT TO (co_user, co_session, co_block)
	FROM SERVER coreprotect_sqlite
	INTO coreprotect;
```

Tu peux aussi créer une seule table étrangère manuellement si tu veux tester avant d’importer le reste.

## Structure

- `src/xray_detector/` : logique applicative
- `tests/` : tests unitaires
- `data/raw/` : archive et exports bruts
- `data/interim/` : sorties temporaires
- `data/processed/` : tables prêtes pour l'analyse
- `reports/figures/` : visuels et exports
- `notebooks/` : exploration et EDA
- `scripts/` : utilitaires ponctuels

## Prochaine étape

Brancher le premier vrai flux sur l'archive CoreProtect: inspection, extraction ciblée, puis export Parquet partitionné.
