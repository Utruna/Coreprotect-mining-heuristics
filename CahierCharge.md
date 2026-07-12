# Cahier des charges — Détection de X-Ray par analyse de données CoreProtect

**Projet** : Pipeline Data Engineering + Data Science pour la détection a posteriori de comportements de minage suspects (x-ray) sur un serveur Minecraft communautaire.
**Auteur** : Colin Liaud
**Terrain de test** : Serveur LaTaverne.me (consentement des joueurs acquis, logging déjà en place)
**Durée cible** : ~1 semaine intensive (soirs + week-end), extensible

---

## 1. Contexte et motivation

Les serveurs Minecraft communautaires font face à la triche par "x-ray" (client ou resource pack modifié permettant de voir les minerais à travers la roche). Les solutions existantes sont soit :
- des plugins de **détection temps réel** à base de règles ou de scoring pondéré simple (XrayDetector, MineWatch AntiXray, [MAX] ML Anti XRay) ;
- de l'**obfuscation préventive** (Paper Anti-Xray, Orebfuscator) ;
- de l'**inspection manuelle** des logs CoreProtect par le staff (`/co lookup`).

**Aucune solution publique identifiée** (état de l'art vérifié en juillet 2026) n'exploite l'historique massif déjà loggé par CoreProtect via un pipeline de données offline avec approche Machine Learning. Ce projet comble ce créneau.

**Atout différenciant** : accès à ~93 Go de données historiques réelles d'un serveur en production, avec consentement des joueurs.

## 2. Objectifs

### Objectif principal
Construire un pipeline complet (extraction → transformation → features → scoring) qui attribue à chaque joueur/session de minage un **score de suspicion x-ray**, exploitable par le staff du serveur.

### Objectifs secondaires (portfolio)
- Démontrer des compétences **Data Engineering** : ETL sur volume réel (~93 Go), optimisation de requêtes, choix d'outils adaptés au volume (DuckDB / SQL push-down), orchestration.
- Démontrer des compétences **Data Science** : feature engineering métier, détection d'anomalies non supervisée, évaluation honnête en l'absence de labels.
- Produire un README/rapport de qualité professionnelle (schéma d'architecture, méthodologie, limites).

### Non-objectifs (hors périmètre V1)
- Sanction automatique des joueurs (ban/kick) — le système reste un outil d'aide à la décision pour le staff.
- Détection temps réel — la V1 est un traitement batch/offline. (Extension possible en V2.)
- Détection d'autres triches (fly, kill aura, dupe) — uniquement le minage.

## 3. Données

### Source
Archive tar.gz (~93 Go annoncés — taille compressée ou décompressée à confirmer) contenant l'historique de logging du serveur.

⚠️ **Hypothèse à valider en tout premier** : le format exact est inconnu (dump MySQL, fichiers SQLite, autre). Le plugin source est vraisemblablement CoreProtect (tables attendues : `co_block`, `co_user`, `co_world`...), mais cela doit être confirmé par inspection de l'archive. Le schéma exact des tables devra être documenté à partir des données réelles, pas de suppositions.

### Données attendues (à confirmer)
- Événements de cassage de blocs : joueur, monde, coordonnées x/y/z, type de bloc, timestamp, action.
- Probablement PAS disponibles : rotation/direction du regard du joueur, position exacte du joueur (seulement celle du bloc cassé). → Impact sur les features possibles (voir §7 Limites).

### Conformité
- Consentement des joueurs : acquis (règlement du serveur, logging CoreProtect déjà annoncé).
- Pseudonymisation recommandée pour toute publication (GitHub, captures) : remplacer les pseudos par des identifiants anonymes dans les exports et visualisations publiques.

## 4. Architecture cible

```
[Archive tar.gz]
      │  (1) Inspection + extraction ciblée
      ▼
[Base source: MySQL dump ou SQLite]
      │  (2) Extraction filtrée (SQL push-down : blocs cassés, minerais + roche)
      ▼
[Zone de staging: fichiers Parquet partitionnés]
      │  (3) Transformation DuckDB / Python
      ▼
[Table de features par joueur × session de minage]
      │  (4) Modèle de détection d'anomalies
      ▼
[Scores de suspicion + rapport / dashboard]
```

### Choix techniques proposés
| Brique | Outil | Justification |
|---|---|---|
| Exploration archive | tar / CLI | Éviter toute extraction complète inutile |
| Extraction | SQL natif (mysql/sqlite3 CLI) | Filtrage à la source, pas de SELECT * sur 93 Go |
| Format intermédiaire | Parquet | Colonnaire, compressé, standard Data Eng |
| Transformation | DuckDB + Python | SQL analytique rapide sur mono-machine, outil en forte croissance, dimensionné pour ce volume sans cluster |
| Features & modèle | pandas/polars + scikit-learn | Isolation Forest / LOF en non supervisé |
| Orchestration (optionnel V1) | Makefile ou Prefect léger | Reproductibilité du pipeline |
| Restitution | Notebook + rapport HTML ou petit dashboard (Streamlit) | Visualisation des scores pour le staff |

## 5. Features envisagées (feature engineering métier)

Par joueur et par **session de minage** (séquence d'événements séparés par moins de N minutes d'inactivité) :

1. **Ratio minerais rares / blocs totaux cassés** — un x-rayer casse peu de pierre pour beaucoup de diamant.
2. **Ratio par type de minerai** (diamant, ancient debris vs charbon, cuivre) — pondération par rareté.
3. **Distance euclidienne moyenne entre minerais rares consécutifs** — le "beeline" produit des sauts directs de filon en filon.
4. **Linéarité de trajectoire** — colinéarité des segments entre blocs cassés successifs (approximation de la trajectoire sans données de rotation).
5. **Vitesse de découverte** — minerais rares par minute, comparée à la distribution du serveur.
6. **Profondeur de minage vs couches optimales** — un joueur légitime suit souvent les couches méta connues ; un x-rayer va où sont les minerais, ce qui peut paradoxalement être similaire → feature à évaluer, potentiellement peu discriminante.
7. **Taux d'exposition** (si calculable) : le minerai cassé bordait-il une cavité existante ? ⚠️ Nécessite l'état du monde au moment T, difficile à reconstruire depuis les seuls logs — voir §7.
8. **Ratio branch-mining** : régularité géométrique du minage (tunnels droits espacés) vs trajectoires erratiques ciblées.

## 6. Plan d'action (7 jours)

### Jour 0 (préalable, ~1h) — Inspection de l'archive
- `tar -tzvf` : inventaire du contenu sans extraction.
- Identification du format (dump SQL / SQLite / autre) et du schéma des tables.
- **Jalon GO/NO-GO** : confirmer que les événements de blocs cassés avec coordonnées + timestamps + joueur sont bien présents. Sans ça, le projet doit être repensé.

### Jour 1 — Extraction ciblée
- Extraire uniquement les tables utiles (co_block, co_user, mapping des types de blocs).
- Écrire les requêtes de filtrage (action = cassage, types = minerais + pierre/deepslate).
- Export vers Parquet partitionné (par mois ou par monde).
- Documenter le schéma réel constaté.

### Jour 2 — Exploration des données (EDA)
- Volumétrie réelle : nb d'événements, nb de joueurs, période couverte.
- Distributions : minerais par joueur, activité temporelle, profondeurs.
- Qualité : trous dans les données, joueurs à très faible activité (à exclure du scoring).

### Jour 3 — Sessionisation + features
- Découpage en sessions de minage (fenêtre d'inactivité à calibrer sur les données).
- Calcul des features §5 en DuckDB/polars.
- Vérification de cohérence sur quelques joueurs connus (toi-même, ton pote = joueurs légitimes de référence).

### Jour 4 — Modélisation
- Baseline : scoring par règles (seuils sur ratios) pour comparaison.
- Isolation Forest / Local Outlier Factor sur les features de session.
- Génération de sessions synthétiques "x-ray" (simulation de beeline parfait) pour tester la sensibilité du modèle — en documentant clairement que c'est synthétique.

### Jour 5 — Restitution
- Score agrégé par joueur, top N sessions les plus anormales.
- Visualisations : trajectoire 3D d'une session suspecte vs normale (très parlant en entretien).
- Mini dashboard Streamlit ou rapport HTML.

### Jour 6 — Industrialisation légère
- Pipeline reproductible bout en bout (Makefile ou Prefect) : une commande = re-calcul complet.
- Tests unitaires sur les fonctions de features.
- Dockerisation optionnelle.

### Jour 7 — Documentation & publication
- README : contexte, architecture (schéma), méthodologie, résultats, **limites assumées**.
- Pseudonymisation des données dans tout ce qui est publié.
- Push GitHub + section CV.

## 7. Étude des limites

### Limites liées aux données
- **Absence de labels** : aucun x-rayer confirmé n'est étiqueté dans les données. Conséquence : impossible de mesurer précision/rappel réels. Le projet doit assumer une évaluation en trois volets : (a) validation sur données synthétiques, (b) revue manuelle par le staff des top sessions flaguées, (c) cohérence avec les joueurs de confiance connus. Toute affirmation de performance chiffrée « en conditions réelles » serait malhonnête.
- **Pas de données de rotation/regard** : CoreProtect logge (a priori) le bloc cassé, pas où regardait le joueur. Les features de trajectoire sont donc des approximations à partir des positions de blocs — moins précises que ce que fait un plugin temps réel.
- **Reconstitution de l'exposition impossible ou coûteuse** : savoir si un minerai était visible ou enfoui au moment du cassage exige de reconstruire l'état du monde à l'instant T (rejouer les logs + connaître la génération initiale du monde). Probablement hors budget temps V1 → feature abandonnée ou approximée (ex : « un bloc non-minerai a-t-il été cassé en position adjacente dans les N secondes précédentes ? »).
- **Biais de survie** : les tricheurs déjà bannis manuellement peuvent être présents dans l'historique (intéressant : sessions potentiellement vérifiables auprès du staff) ou avoir été purgés.
- **Purge CoreProtect** : si le serveur a un paramètre de purge automatique des vieux logs, l'historique peut être tronqué de façon non uniforme.

### Limites méthodologiques
- **Anomalie ≠ triche** : la détection d'anomalies flague ce qui est *rare*, pas ce qui est *interdit*. Un joueur très chanceux, un speedrunner de minage efficace, ou un joueur suivant un tutoriel de branch-mining optimal peut ressortir comme anormal. Inversement, un x-rayer prudent (qui mine « normalement » et ne dévie que rarement) peut passer sous le radar.
- **Connaissance méta légitime** : depuis les versions récentes, les couches optimales de spawn des minerais sont publiques et connues. Miner « pile aux bonnes couches » n'est pas un signal de triche.
- **Faux positifs coûteux socialement** : sur un petit serveur communautaire, accuser à tort un joueur est pire que rater un tricheur. D'où le choix ferme : outil d'aide à la décision, jamais de sanction automatique.
- **Distribution non stationnaire** : mises à jour du jeu (nouvelles couches de génération, nouveaux minerais) et resets de map changent les distributions au fil de l'historique → le modèle doit soit segmenter par période, soit être entraîné sur une ère homogène.

### Limites techniques
- **Volume vs machine** : 93 Go (si décompressé) reste faisable sur une machine correcte avec DuckDB/Parquet, mais l'extraction initiale depuis un dump SQL peut être longue (import MySQL complet potentiellement nécessaire si le dump n'est pas parsable directement). Prévoir un plan B : parser le dump en streaming pour n'extraire que les INSERT des tables utiles.
- **Adversarial** : si le système était publié et connu, un tricheur peut adapter son comportement pour rester sous les seuils (limite intrinsèque à toute détection comportementale, à mentionner honnêtement).
- **Généralisabilité** : le modèle est calibré sur la population d'UN serveur ; les seuils/distributions ne se transfèrent pas directement à un autre serveur.

## 8. Critères de réussite

1. Pipeline reproductible qui va de l'archive brute aux scores en une commande documentée.
2. Temps de traitement raisonnable (ordre de grandeur : < 1h pour un recalcul complet des features hors import initial).
3. Au moins une visualisation de trajectoire suspecte vs normale exploitable en entretien.
4. Revue par le staff de LaTaverne d'un top 10 de sessions flaguées, avec retour qualitatif documenté.
5. README avec limites explicitement assumées (section 7 condensée).

## 9. Extensions possibles (V2+)

- Plugin Java léger de scoring temps réel consommant le modèle exporté (ONNX) — boucle avec l'écosystème Paper déjà maîtrisé.
- Étiquetage progressif : le staff confirme/infirme les flags → constitution d'un vrai dataset labellisé → passage au supervisé.
- Intégration Discord (alertes staff).
- Comparaison de plusieurs algorithmes d'anomaly detection avec analyse critique.
