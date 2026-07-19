# Analyse statistique des sessions de minage

Calcul de features de trajectoire par session de minage et score de suspicion x-ray (V1 heuristique), à partir d'une base CoreProtect SQLite. C'est le pendant chiffré de la [preview 3D](readmePreview.md) : ce que l'œil voit dans la reconstruction (tunnels qui foncent de filon en filon), l'analyse le transforme en indicateurs mesurables et en un score 0-100 par session.

L'analyse est disponible sous deux formes :

- **intégrée à la preview web** (`scripts/render_mining_3d.py`) : panneau latéral avec score en anneau, verdict, jauges des indicateurs, détails et classement des sessions, avec sélecteur de minerai surveillé — le Python précalcule les features pour chaque minerai présent en base, le changement de cible est instantané dans la page. Le bouton « Métriques ? » y ouvre une page d'explication intégrée de toutes les métriques et du score ;
       la page peut aussi être chargée sur une fenêtre temporelle donnée (`--window`, `--start`, `--end`), sans modifier le score calculé sur la session retenue ;
- **en ligne de commande** (`scripts/analyze_mining_sessions.py`, ci-dessous) : tableau console, export CSV et figure comparative, pour le travail d'exploration et la calibration.

![Comparaison des features par session](reports/figures/session_features_diamond.png)

## Lancement

Depuis la racine du projet :

```powershell
.venv\Scripts\python.exe scripts\analyze_mining_sessions.py
```

Sortie attendue :

```
13171 blocs casses par 3 joueurs charges depuis database_testserv.db
Sessions ignorees (< 50 blocs) : 6
Sessions exclues car ressemblant a des grottes/geodes : 4
Minerai surveille : diamant (diamond)

--- Features et score par session (trie par score decroissant) ---
  pseudo  n_blocks  n_dig_blocks  dig_ratio  ...  score           verdict
Joueur 1       887           878      0.990  ...   64.7 fortement suspect
Joueur 2       371           366      0.987  ...   54.9      a surveiller
Joueur 3      2964          2830      0.955  ...   24.9               RAS

Tableau complet ecrit : data\processed\session_features_diamond.csv
Figure ecrite : reports\figures\session_features_diamond.png
```

### Options

| Option | Défaut | Rôle |
|---|---|---|
| `--db` | `data/raw/database_testserv.db` | Base CoreProtect SQLite à analyser |
| `--start` / `--end` | toute la base | Fenêtre temporelle UTC (`2026-06-01` ou `2026-06-01T00:00:00Z`, fin incluse). Le filtre est poussé dans le SQL avant chargement — indispensable sur une grosse base |
| `--gap` | `300` | Trou d'inactivité (secondes) qui coupe une session en deux |
| `--min-blocks` | `50` | Nombre minimal de blocs cassés pour garder une session |
| `--ore` | `diamond` | Minerai surveillé : `diamond`, `gold`, `iron`, `copper`, `emerald`, `redstone`, `lapis`, `coal`, `quartz`, `ancient_debris` |
| `--output` | `data/processed/session_features_<ore>.csv` | Tableau complet en CSV |
| `--figure` | `reports/figures/session_features_<ore>.png` | Figure comparative PNG |
| `--no-figure` | — | Ne pas générer la figure |
| `--anonymize` | — | Pseudos inventés dans toutes les sorties (suffixe `_anon`), mapping affiché en console uniquement |
| `--anomaly-model` | `data/models/anomaly_iforest_<ore>.joblib` | Modèle d'anomalie à charger pour les colonnes `anomaly_*` (omises si le fichier n'existe pas — voir section Isolation Forest) |

Exemple : surveiller l'or plutôt que le diamant, avec une segmentation plus fine :

```powershell
.venv\Scripts\python.exe scripts\analyze_mining_sessions.py --ore gold --gap 180
```

Le choix du minerai change la cible des features de filons (rendement, détour, virages) **et** les bornes de rendement du score : trouver 5 fers / 100 blocs est banal, trouver 5 diamants / 100 blocs ne l'est pas.

## Le pipeline

1. **Extraction** ([src/xray_detector/mining.py](src/xray_detector/mining.py)) — blocs cassés (`action = 0`) par les vrais joueurs uniquement (uuid non nul, ce qui exclut `#lava`, `#gravity`…), en ignorant les blocs que le joueur avait lui-même posés. Tous les matériaux sont gardés : la roche décrit le chemin, les minerais sont la cible.
2. **Segmentation en sessions** — coupure par joueur et par monde dès qu'un trou d'inactivité dépasse `--gap` secondes ; les micro-sessions sous `--min-blocks` blocs sont écartées.
3. **Filtre d'environnement** — les sessions qui ressemblent à des cavernes / géodes naturelles sont exclues par défaut des sorties d'analyse et de la preview 3D, car elles ne relèvent pas du strip-mining que le score x-ray V1 cherche à comparer. L'option `--include-cave-sessions` permet de les réinclure pour inspection manuelle.
4. **Reconstitution de la trajectoire** ([src/xray_detector/features.py](src/xray_detector/features.py)) — la session est lue comme une suite de positions, de bloc cassé en bloc cassé. Un pas de plus de 4 blocs (`JUMP_DISTANCE`) est un déplacement sans minage (marche en grotte, chute, téléportation) : il coupe la continuité directionnelle mais ne compte pas comme un virage.
5. **Classement creusage / marche** — chaque casse est classée en phase de *creusage* (pas ≤ 2 blocs entre casses consécutives, `DIG_STEP_DISTANCE`, sur au moins 4 casses enchaînées, `DIG_PHASE_MIN_BLOCKS`) ou de *marche*. En strip-mining le joueur creuse son chemin (~1 bloc par pas) ; en grotte il marche dans l'air entre les minerais exposés. Le rendement du score n'est calculé que sur les blocs creusés, et un filon ne porte le signal d'intentionnalité que s'il a été **atteint en creusant** (les 3 pas précédents sont du creusage, `APPROACH_DIG_STEPS`) : marcher droit vers un minerai qu'on *voit* en grotte n'est pas une fuite d'information.
6. **Features puis score** — détail ci-dessous.

## Les features

Trois familles, calculées par session pour le minerai cible choisi.

### Forme du chemin

| Feature | Calcul | Ce qu'elle raconte |
|---|---|---|
| `changes_per_100` | Nombre de changements de direction dominante (axe + sens) pour 100 pas | Un chemin qui slalome beaucoup ou des galeries au cordeau |
| `mean_run_h` / `mean_run_v` | Longueur moyenne (en blocs) des segments droits horizontaux / verticaux | Les « lignes droites » du joueur, séparées par orientation |
| `vertical_step_ratio` | Part des pas dont l'axe dominant est vertical | Un joueur qui plonge/remonte sans cesse vers des cibles |

### Rendement

| Feature | Calcul | Ce qu'elle raconte |
|---|---|---|
| `ore_per_100` | Minerais (toutes familles) pour 100 blocs cassés | Rendement global |
| `target_per_100` | Minerai cible pour 100 blocs cassés (session entière) | Rendement brut, gardé pour comparaison — gonflé mécaniquement en grotte |
| `target_per_100_dig` | Minerai cible pour 100 blocs **creusés** (NaN sous 30 blocs creusés, `MIN_DIG_BLOCKS_FOR_RATE`) | **Le** signal du x-ray : un rendement en creusant inexplicable par la chance |
| `n_dig_blocks` / `dig_ratio` | Blocs creusés et leur part dans la session | Contexte : strip-mining (~1.0) ou exploration de grotte (faible) |
| `walk_step_ratio` | Part des pas > 4 blocs | Indicateur bon marché de contexte grotte |
| `path_straightness` | Vol d'oiseau début→fin ÷ chemin de la trajectoire simplifiée (un ancrage tous les 8 blocs, ce qui gomme le zigzag du geste de minage) | ≥ 0.65 = session « couloir » : creuser tout droit n'est pas viser |
| `n_target_veins` / `n_dig_veins` | Filons distincts de la cible (Chebyshev ≤ 2) / dont atteints en creusant | Volume de découvertes et part « creusée » |
| `mean_blocks_between_veins` | Blocs minés entre la fin d'un filon et le début du suivant | « Combien je creuse avant de trouver » |

### Intentionnalité

C'est la famille la plus discriminante : elle mesure si le joueur *sait où il va*.

| Feature | Calcul | Ce qu'elle raconte |
|---|---|---|
| `detour_factor` | Longueur du chemin miné entre deux filons successifs **atteints en creusant** ÷ distance à vol d'oiseau (moyenne sur les paires ; paires < 3 blocs ou traversées par un pas de marche > 4 blocs exclues) | 1.0 = ligne parfaitement droite de filon en filon. Un joueur légitime quadrille : ≥ 3 |
| `turn_toward_ore_rate` | À chaque virage, la nouvelle direction rapproche-t-elle du prochain filon **creusé, pas encore découvert** ? (taux sur tous les virages évaluables) | Un virage aléatoire rapproche ~1 fois sur 2. Viser juste presque à chaque virage trahit une information que le joueur ne devrait pas avoir |

`n_detour_pairs` et `n_turns_evaluated` exposent la base de preuve de chaque indicateur (utilisés par le garde-fou du score, ci-dessous).

## Le score V1

Score 0-100 = moyenne pondérée de trois indicateurs, chacun normalisé par une rampe linéaire bornée (en dessous de la borne basse → 0, au-dessus de la borne haute → 1) :

| Indicateur | Bornes (diamant) | Poids |
|---|---|---|
| `target_per_100_dig` | 0.8 → 3.0 (bornes par minerai, voir `TARGET_RATE_RAMPS`) | 0.4 |
| `detour_factor` | 3.0 → 1.4 (inversées : petit détour = suspect) | 0.3 |
| `turn_toward_ore_rate` | 0.5 → 0.85 | 0.3 |

Verdicts : **≥ 60** fortement suspect · **≥ 30** à surveiller · **< 30** RAS. Si un indicateur est incalculable (moins de 2 filons par exemple), il est retiré et les poids sont renormalisés.

**Garde-fous de preuve** : un indicateur sans base suffisante est écarté (`detour_factor` sous 2 paires évaluées, `turn_toward_ore_rate` sous 5 virages, voir `EVIDENCE_REQUIREMENTS`). Dans une session **couloir** (`path_straightness` ≥ 0.65, `CORRIDOR_STRAIGHTNESS`), les deux indicateurs d'intentionnalité sont écartés d'office : le joueur n'a fait aucun choix de navigation, un détour de 1 et des « virages vers le filon » n'y prouvent rien — les filons se trouvaient sur la ligne (couloir de transport, escalier vers la profondeur). Si le poids total des indicateurs restants est sous 0.6 (`MIN_WEIGHT_SUM`), le score est plafonné à 59.9 : le rendement seul ne peut jamais produire « fortement suspect ». La colonne `evidence_weight` du CSV donne le poids effectivement utilisé, et le verdict `indeterminable` apparaît quand aucun indicateur n'est calculable (typiquement une session de grotte sans creusage).

Les colonnes `ind_*` du CSV donnent la contribution normalisée de chaque indicateur : on voit *pourquoi* une session score haut, pas juste combien.

**Assumé :** c'est une heuristique calibrée sur la connaissance du jeu (rendement d'un strip-mineur à Y-59, géométrie d'un quadrillage…), pas un modèle appris. Elle sert de base de comparaison et d'étiqueteur grossier en attendant un corpus suffisant pour entraîner un vrai classifieur. Un premier modèle non supervisé la complète depuis le Jour 4 (section suivante) — il ne la remplace pas.

## Résultats sur la base test (vérité terrain connue)

Meilleure session par joueur (pipeline complet avec classement creusage/marche) :

| Joueur | Comportement réel | `target_per_100_dig` | `detour_factor` | `turn_toward_ore_rate` | Score | Verdict |
|---|---|---|---|---|---|---|
| Joueur 1 | X-ray simulé | 5.24 | 2.83 | 0.75 | **64.7** | fortement suspect |
| Joueur 2 | X-ray simulé | 6.28 | 2.94 | 0.66 | **54.9** | à surveiller |
| Joueur 3 | Strip-mining légitime | 1.52 | 2.59 | 0.55 | **24.9** | RAS |

Le classement est correct et l'écart net entre les deux profils. À noter :

- Le **rendement** discrimine très bien ; le **détour** s'est resserré depuis que les paires traversées par un pas de marche sont exclues (le quadrillage légitime perdait surtout ses paires longues), à recalibrer sur données réelles.
- `mean_run_h` (~1.1 bloc pour tout le monde) et `changes_per_100` (~85 partout) ne discriminent **pas** en l'état : le minage en tunnel de 2 de haut alterne un pas avant / un pas vertical, ce qui écrase la macro-structure du chemin. Voir limites ci-dessous.

## Détection d'anomalies non supervisée (Isolation Forest)

Depuis le Jour 4, un second regard **complète** le score heuristique : un Isolation Forest ([src/xray_detector/anomaly_model.py](src/xray_detector/anomaly_model.py)) entraîné sans étiquettes sur les sessions de la vraie base. Là où le score V1 encode une connaissance du jeu (rampes calibrées à la main sur 3 indicateurs), le modèle apprend ce qu'est une session *typique* du corpus et mesure l'écart — sur 13 features à la fois, y compris celles que le score V1 n'utilise pas. Les deux colonnes coexistent dans toutes les sorties (`score`/`verdict` et `anomaly_score`) : c'est la confrontation des deux qui est informative, pas l'une ou l'autre seule.

### Entraînement et scoring

```powershell
# Entraîner depuis un CSV de features déjà produit (voie normale pour la grosse base)…
.venv\Scripts\python.exe scripts\train_anomaly_model.py --from-csv data\processed\session_features_diamond_anon.csv

# …ou en rejouant tout le pipeline sur une base (fenêtre poussée dans le SQL)
.venv\Scripts\python.exe scripts\train_anomaly_model.py --db data\raw\CoreProtect\database.db --start 2026-06-01 --end 2026-06-30
```

Le modèle est écrit dans `data/models/anomaly_iforest_<ore>.joblib` (un modèle par minerai cible) et `scripts/analyze_mining_sessions.py` le charge automatiquement s'il existe : le tableau, le CSV et la console gagnent `anomaly_raw` (decision_function sklearn), `anomaly_score` (0-100), `anomaly_top_feature` et `anomaly_top_delta`. L'entraînement refuse un corpus de moins de 30 sessions : sur une poignée de points, l'Isolation Forest isole d'abord *le point seul de son côté* — sur la base de test, ce serait le joueur légitime.

### Choix de conception

- **Isolation Forest plutôt que LOF** : pas de choix de `k` voisins ni de métrique de distance à justifier (avec 13 features hétérogènes, une distance euclidienne mélange des blocs, des taux et des ratios), insensible à l'échelle des features (coupes par feature), score exploitable hors échantillon pour scorer de nouvelles sessions sans réentraîner — LOF en mode `novelty` le permet aussi mais reste sensible à la densité locale, mal définie sur quelques centaines de points.
- **Features** : les 13 features de forme / rendement / intentionnalité invariantes à la taille de session (voir `ANOMALY_FEATURES`). Exclues : `n_blocks`, `duration_min`, `blocks_per_min` (taille et vitesse ne sont pas des signaux de x-ray), les comptages `n_*` (corrélés à la longueur), et tout ce qui sort du score V1 (`score`, `ind_*`) — le modèle reste indépendant de l'heuristique pour que la comparaison ait un sens. Entraînement uniquement sur des sessions ayant passé le filtre grotte/géode, pour ne pas réapprendre les faux positifs déjà réglés en amont.
- **Directionnalité** : un Isolation Forest brut est non-directionnel — première leçon de la vérité terrain : la longue session patiente du joueur légitime (200 blocs creusés entre deux filons, un record de malchance) sortait *plus atypique qu'un tricheur*. Pour les 6 features de rendement / intentionnalité dont la direction suspecte est connue (`SUSPICIOUS_DIRECTION`), le côté « légitime » est donc écrêté à la médiane du corpus : être très malchanceux ou très quadrilleur ne rend plus atypique. Les features de forme, sans direction évidente, restent bilatérales.
- **NaN** : imputation par la **médiane du corpus d'entraînement**. Un NaN signifie « pas assez de preuve » (rendement sous 30 blocs creusés, indicateurs d'intentionnalité écartés) ; la médiane est la valeur neutre du corpus, donc une session incomplète ne peut pas devenir anormale *à cause de ses trous* — vérifié par un test dédié. L'alternative (exclure les sessions incomplètes) jetait une part importante du corpus pour des NaN très fréquents sur `detour_factor`.
- **Normalisation 0-100** : `anomaly_score` est ancré sur la decision_function — **50 = seuil de contamination** ([0, max du corpus] → [50, 0] et [min du corpus, 0] → [100, 50], borné aux extrêmes du corpus d'entraînement). Un score ≥ 50 se lit « plus atypique que (1 − contamination) du corpus », pas « probabilité de triche ».
- **`contamination` = 0.05 par défaut** : part de sessions supposées atypiques dans le corpus. Sans vérité terrain, ce n'est **pas calibrable** — c'est un hyperparamètre documenté qui déplace le « 50 » du score, pas la qualité du classement (le rang des sessions n'en dépend pas).
- **Explication** : `anomaly_top_feature` est la feature dont le remplacement par la médiane du corpus rapproche le plus la session de la normale (perturbation une-feature-à-la-fois), même esprit que les colonnes `ind_*` du score V1 : on voit *pourquoi*, pas juste combien.

### Ce que ça donne

**Sur la vérité terrain** (modèle entraîné sur 413 sessions de la vraie base, jamais vues) :

| Joueur | Comportement réel | Score V1 | `anomaly_score` | `anomaly_top_feature` |
|---|---|---|---|---|
| Joueur 1 (2 sessions) | X-ray simulé | 64.7 / 59.5 | **66.1 / 60.9** | detour_factor, target_per_100_dig |
| Joueur 2 | X-ray simulé | 54.9 | **49.5** | target_per_100 |
| Joueur 3 | Strip-mining légitime | 24.9 | **42.0** | detour_factor |

Le classement est le bon et les deux x-rayeurs passent devant, mais la marge est plus mince qu'avec le score V1 (49.5 vs 42.0 pour le cas le plus serré) : le joueur légitime de la base de test est un mineur *efficace* pour le corpus réel (détour sous la médiane), et le modèle n'a aucun moyen de le savoir innocent. C'est le comportement attendu d'un détecteur d'écart, pas un défaut à sur-corriger. Le test [tests/test_anomaly_model.py](tests/test_anomaly_model.py) verrouille cette séparation (et saute proprement si la base ou le modèle manquent).

**Sur les sessions de la vraie base** (le corpus lui-même, contrôle de cohérence affiché par le script d'entraînement) : corrélation de rang de 0.43 avec le score V1 — assez corrélé pour se conforter, assez décorrélé pour apporter autre chose. 8 des 10 sessions les plus atypiques sont « à surveiller » ou « fortement suspect » au score V1 ; les 2 restantes sont atypiques par la **forme** (`mean_run_v`, `mean_blocks_between_veins`), des features que le score V1 n'exploite pas — exactement le genre de session qu'un humain doit aller regarder dans la preview 3D, et un rappel qu'**atypique ≠ tricheur**.

**Limites, sans détour** : aucun chiffre de précision/rappel n'est annonçable — il n'y a pas d'étiquettes, et le corpus d'entraînement contient lui-même des tricheurs éventuels (c'est le rôle de `contamination` de l'encaisser). Le modèle dit « cette session ne ressemble pas au corpus », rien de plus ; le verdict reste celui du score V1 plus l'inspection visuelle.

## Limites connues et pistes

- **Micro-structure vs macro-structure** : les longueurs de segments droits mesurent le geste de minage (tunnel 2 de haut), pas la forme de la galerie. Amélioration prévue : simplifier la trajectoire avant mesure (fusion des paires verticales, ou simplification type Douglas-Peucker), ce qui devrait faire ressortir la grille du strip-mineur face aux vers de terre du x-rayeur.
- **Calibration** : les bornes des rampes (et `TARGET_RATE_RAMPS` pour les minerais autres que le diamant) sont des estimations à recalibrer sur données réelles étiquetées.
- **Virages en cours de filon** : les changements de direction pendant qu'on casse un filon diluent `turn_toward_ore_rate`. Le biais est le même pour tous les joueurs, mais le signal serait plus net en les excluant.
- **Le score est par session** : un tricheur qui alterne minage propre et x-ray dans la même session dilue son score. La fenêtre temporelle de la preview 3D permet déjà d'inspecter visuellement ; un scoring par fenêtre glissante est la suite logique.
- **Évasion par la marche** : un x-rayeur qui parcourt un réseau de grottes et ne creuse que les 2-3 derniers blocs vers chaque filon échappe en partie au rendement « creusage ». Sans donnée d'exposition des blocs (collecteur figé), on ne peut pas fermer complètement ce trou ; les contre-mesures prévues sont un plancher « à surveiller » sur le rendement pleine-session anormal et le comptage des filons atteints par un creusage court *dans un mur* (percer un mur pile sur un filon reste une fuite d'information, même en grotte).

## Sorties

- **CSV** (`data/processed/session_features_<ore>.csv`) : une ligne par session — identification (`pseudo`, `world`, `session`, `target`), toutes les features, contributions `ind_*`, `score`, `verdict`, et si un modèle d'anomalie est présent : `anomaly_raw`, `anomaly_score`, `anomaly_top_feature`, `anomaly_top_delta`.
- **Figure** (`reports/figures/session_features_<ore>.png`) : 6 panneaux de features (une barre par session, couleur stable par joueur, lignes pointillées = références « ligne droite » et « hasard ») + panneau du score coloré par verdict (rouge / ambre / vert).
- **Console** : tableau trié par score décroissant.

## Tests

Les briques de calcul sont couvertes par [tests/test_features.py](tests/test_features.py) sur des chemins synthétiques : tunnel droit, virage en L, escalier vertical, saut de continuité, regroupement en filons, détour, virage vers un filon caché, séparation des profils par le score, bornes par minerai. Le modèle d'anomalie est couvert par [tests/test_anomaly_model.py](tests/test_anomaly_model.py) : mécanique sur corpus synthétique (bornes, NaN neutres, garde-fou directionnel, sauvegarde/rechargement) et séparation sur la vérité terrain avec le modèle réellement entraîné.

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```
