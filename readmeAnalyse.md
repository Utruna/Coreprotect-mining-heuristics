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
Sessions ignorees (< 50 blocs) : 2
Minerai surveille : diamant (diamond)

--- Features et score par session (trie par score decroissant) ---
       pseudo  n_blocks  ...  score           verdict
IxLikexYoou44      6617  ...   67.2 fortement suspect
     acsterix      3343  ...   59.7      a surveiller
       Utruna      3197  ...   18.8               RAS

Tableau complet ecrit : data\processed\session_features_diamond.csv
Figure ecrite : reports\figures\session_features_diamond.png
```

### Options

| Option | Défaut | Rôle |
|---|---|---|
| `--db` | `data/raw/database_testserv.db` | Base CoreProtect SQLite à analyser |
| `--gap` | `300` | Trou d'inactivité (secondes) qui coupe une session en deux |
| `--min-blocks` | `50` | Nombre minimal de blocs cassés pour garder une session |
| `--ore` | `diamond` | Minerai surveillé : `diamond`, `gold`, `iron`, `copper`, `emerald`, `redstone`, `lapis`, `coal`, `quartz`, `ancient_debris` |
| `--output` | `data/processed/session_features_<ore>.csv` | Tableau complet en CSV |
| `--figure` | `reports/figures/session_features_<ore>.png` | Figure comparative PNG |
| `--no-figure` | — | Ne pas générer la figure |
| `--anonymize` | — | Pseudos inventés dans toutes les sorties (suffixe `_anon`), mapping affiché en console uniquement |

Exemple : surveiller l'or plutôt que le diamant, avec une segmentation plus fine :

```powershell
.venv\Scripts\python.exe scripts\analyze_mining_sessions.py --ore gold --gap 180
```

Le choix du minerai change la cible des features de filons (rendement, détour, virages) **et** les bornes de rendement du score : trouver 5 fers / 100 blocs est banal, trouver 5 diamants / 100 blocs ne l'est pas.

## Le pipeline

1. **Extraction** ([src/xray_detector/mining.py](src/xray_detector/mining.py)) — blocs cassés (`action = 0`) par les vrais joueurs uniquement (uuid non nul, ce qui exclut `#lava`, `#gravity`…), en ignorant les blocs que le joueur avait lui-même posés. Tous les matériaux sont gardés : la roche décrit le chemin, les minerais sont la cible.
2. **Segmentation en sessions** — coupure par joueur et par monde dès qu'un trou d'inactivité dépasse `--gap` secondes ; les micro-sessions sous `--min-blocks` blocs sont écartées.
3. **Reconstitution de la trajectoire** ([src/xray_detector/features.py](src/xray_detector/features.py)) — la session est lue comme une suite de positions, de bloc cassé en bloc cassé. Un pas de plus de 4 blocs (`JUMP_DISTANCE`) est un déplacement sans minage (marche en grotte, chute, téléportation) : il coupe la continuité directionnelle mais ne compte pas comme un virage.
4. **Features puis score** — détail ci-dessous.

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
| `target_per_100` | Minerai cible pour 100 blocs cassés | **Le** signal brut du x-ray : un rendement inexplicable par la chance |
| `n_target_veins` | Nombre de filons distincts de la cible (regroupement des casses à distance de Chebyshev ≤ 2) | Volume de découvertes |
| `mean_blocks_between_veins` | Blocs minés entre la fin d'un filon et le début du suivant | « Combien je creuse avant de trouver » |

### Intentionnalité

C'est la famille la plus discriminante : elle mesure si le joueur *sait où il va*.

| Feature | Calcul | Ce qu'elle raconte |
|---|---|---|
| `detour_factor` | Longueur du chemin miné entre deux filons successifs ÷ distance à vol d'oiseau (moyenne sur les paires ; paires < 3 blocs ou avec téléportation exclues) | 1.0 = ligne parfaitement droite de filon en filon. Un joueur légitime quadrille : ≥ 3 |
| `turn_toward_ore_rate` | À chaque virage, la nouvelle direction rapproche-t-elle du prochain filon **pas encore découvert** ? (taux sur tous les virages évaluables) | Un virage aléatoire rapproche ~1 fois sur 2. Viser juste presque à chaque virage trahit une information que le joueur ne devrait pas avoir |

## Le score V1

Score 0-100 = moyenne pondérée de trois indicateurs, chacun normalisé par une rampe linéaire bornée (en dessous de la borne basse → 0, au-dessus de la borne haute → 1) :

| Indicateur | Bornes (diamant) | Poids |
|---|---|---|
| `target_per_100` | 0.8 → 3.0 (bornes par minerai, voir `TARGET_RATE_RAMPS`) | 0.4 |
| `detour_factor` | 3.0 → 1.4 (inversées : petit détour = suspect) | 0.3 |
| `turn_toward_ore_rate` | 0.5 → 0.85 | 0.3 |

Verdicts : **≥ 60** fortement suspect · **≥ 30** à surveiller · **< 30** RAS. Si un indicateur est incalculable (moins de 2 filons par exemple), il est retiré et les poids sont renormalisés.

Les colonnes `ind_*` du CSV donnent la contribution normalisée de chaque indicateur : on voit *pourquoi* une session score haut, pas juste combien.

**Assumé :** c'est une heuristique calibrée sur la connaissance du jeu (rendement d'un strip-mineur à Y-59, géométrie d'un quadrillage…), pas un modèle appris. Elle sert de base de comparaison et d'étiqueteur grossier en attendant un corpus suffisant pour entraîner un vrai classifieur (scikit-learn est déjà dans les dépendances).

## Résultats sur la base test (vérité terrain connue)

| Joueur | Comportement réel | `target_per_100` | `detour_factor` | `turn_toward_ore_rate` | Score | Verdict |
|---|---|---|---|---|---|---|
| IxLikexYoou44 | X-ray simulé | 5.83 | 2.51 | 0.71 | **67.2** | fortement suspect |
| acsterix | X-ray simulé | 5.12 | 2.70 | 0.66 | **59.7** | à surveiller |
| Utruna | Strip-mining légitime | 1.50 | 9.76 | 0.57 | **18.8** | RAS |

Le classement est correct et l'écart net entre les deux profils. À noter :

- Le **rendement** et le **détour** discriminent très bien (9.76 pour le quadrillage légitime contre ~2.6 pour les x-rayeurs — pas 1.0, car même un tunnel « droit » en 2 de haut zigzague bloc par bloc).
- `mean_run_h` (~1.1 bloc pour tout le monde) et `changes_per_100` (~85 partout) ne discriminent **pas** en l'état : le minage en tunnel de 2 de haut alterne un pas avant / un pas vertical, ce qui écrase la macro-structure du chemin. Voir limites ci-dessous.

## Limites connues et pistes

- **Micro-structure vs macro-structure** : les longueurs de segments droits mesurent le geste de minage (tunnel 2 de haut), pas la forme de la galerie. Amélioration prévue : simplifier la trajectoire avant mesure (fusion des paires verticales, ou simplification type Douglas-Peucker), ce qui devrait faire ressortir la grille du strip-mineur face aux vers de terre du x-rayeur.
- **Calibration** : les bornes des rampes (et `TARGET_RATE_RAMPS` pour les minerais autres que le diamant) sont des estimations à recalibrer sur données réelles étiquetées.
- **Virages en cours de filon** : les changements de direction pendant qu'on casse un filon diluent `turn_toward_ore_rate`. Le biais est le même pour tous les joueurs, mais le signal serait plus net en les excluant.
- **Le score est par session** : un tricheur qui alterne minage propre et x-ray dans la même session dilue son score. La fenêtre temporelle de la preview 3D permet déjà d'inspecter visuellement ; un scoring par fenêtre glissante est la suite logique.

## Sorties

- **CSV** (`data/processed/session_features_<ore>.csv`) : une ligne par session — identification (`pseudo`, `world`, `session`, `target`), toutes les features, contributions `ind_*`, `score`, `verdict`.
- **Figure** (`reports/figures/session_features_<ore>.png`) : 6 panneaux de features (une barre par session, couleur stable par joueur, lignes pointillées = références « ligne droite » et « hasard ») + panneau du score coloré par verdict (rouge / ambre / vert).
- **Console** : tableau trié par score décroissant.

## Tests

Les briques de calcul sont couvertes par [tests/test_features.py](tests/test_features.py) sur des chemins synthétiques : tunnel droit, virage en L, escalier vertical, saut de continuité, regroupement en filons, détour, virage vers un filon caché, séparation des profils par le score, bornes par minerai.

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```
