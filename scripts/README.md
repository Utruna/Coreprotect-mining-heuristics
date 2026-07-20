# Scripts

Utilitaires ponctuels pour l'inspection de l'archive, les exports ciblés et les vérifications manuelles.

## benchmark_pipeline.py

Benchmark reproductible du pipeline d'analyse (extraction SQL, sessionization, features + score, modèle d'anomalie) sur une fenêtre **fixe** de 30 jours (juin 2026) de la vraie base CoreProtect. À chaque exécution, il mesure le temps mur par étape, le temps CPU et le pic de RAM, ajoute une ligne par run dans `reports/benchmarks/benchmark_history.csv` (avec date, commit git et note libre), puis régénère `reports/figures/benchmark_evolution.png` qui montre l'évolution des mesures dans le temps.

La fenêtre et la base ne doivent pas changer d'une mesure à l'autre : la charge reste identique, donc les écarts entre deux lignes reflètent l'optimisation du code, pas l'évolution des données.

```powershell
# Mesure de référence (base réelle, juin 2026)
.venv\Scripts\python.exe scripts\benchmark_pipeline.py --label "avant optimisation X"

# Moyenne plus stable : 3 runs (la figure prend la médiane)
.venv\Scripts\python.exe scripts\benchmark_pipeline.py --runs 3

# Essai rapide sur la base de test
.venv\Scripts\python.exe scripts\benchmark_pipeline.py --db data\raw\database_testserv.db --start 2026-07-14 --end 2026-07-15
```
