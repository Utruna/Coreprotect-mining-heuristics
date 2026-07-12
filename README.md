# AntiCheat X-Ray Detection

Base de projet pour analyser a posteriori les logs CoreProtect et produire un score de suspicion x-ray par joueur ou session de minage.

## Ce qui est déjà en place

- Environnement Python virtuel configuré dans `.venv`
- Structure `src/` prête pour le code du pipeline
- Dossiers de travail pour les données brutes, intermédiaires et traitées
- Entrée CLI minimale pour préparer le workspace et afficher la configuration

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
