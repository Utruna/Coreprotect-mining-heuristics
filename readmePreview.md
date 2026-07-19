# Preview 3D des sessions de minage

Reconstruction 3D interactive des sessions de minage à partir d'une base CoreProtect SQLite, pour visualiser d'un coup d'œil la différence entre un minage légitime et un comportement x-ray. La page embarque aussi le panneau d'analyse ([readmeAnalyse.md](readmeAnalyse.md)) : score de suspicion, indicateurs et classement des sessions, par minerai surveillé.

![Session x-ray](reports/figures/preview_IxLikexYoou44.png)

🔗 **[Démo live (anonymisée)](https://utruna.github.io/Coreprotect-mining-heuristics/)** — la page ci-dessous, directement dans le navigateur.

## Lancement

Depuis la racine du projet, avec l'environnement virtuel du projet :

```powershell
.venv\Scripts\python.exe scripts\render_mining_3d.py
```

Le script lit par défaut `data/raw/database_testserv.db` et écrit le rendu dans `reports/figures/mining_sessions_3d.html`. Sans filtre temporel, il prend toute la base ; il suffit ensuite d'ouvrir ce fichier dans un navigateur : il est autonome (Plotly embarqué), aucune connexion ni serveur nécessaire.

Sortie attendue :

```
13171 blocs casses par 3 joueurs charges depuis database_testserv.db
3 sessions de minage retenues (gap > 300s, >= 50 blocs)
Rendu ecrit : reports/figures/mining_sessions_3d.html (5.0 Mo)
```

### Options

| Option | Défaut | Rôle |
|---|---|---|
| `--db` | `data/raw/database_testserv.db` | Base CoreProtect SQLite à analyser |
| `--window` | `all` | Fenêtre relative à charger : `all`, `last-12h`, `last-24h`, `yesterday` |
| `--start` | — | Début UTC ISO 8601 si tu veux une borne absolue |
| `--end` | — | Fin UTC ISO 8601 si tu veux une borne absolue |
| `--gap` | `300` | Trou d'inactivité (en secondes) qui coupe une session en deux |
| `--min-blocks` | `50` | Nombre minimal de blocs cassés pour qu'une session soit gardée |
| `--output` | `reports/figures/mining_sessions_3d.html` | Fichier HTML généré (`_anon` ajouté si `--anonymize`) |
| `--anonymize` | — | Remplace les pseudos par des pseudos inventés pour pouvoir partager la page |

Exemple sur une autre base, avec des sessions plus fines :

```powershell
.venv\Scripts\python.exe scripts\render_mining_3d.py --db data\raw\CoreProtect\database.db --gap 180 --min-blocks 100
```

Exemples avec filtre temporel :

```powershell
.venv\Scripts\python.exe scripts\render_mining_3d.py --window last-12h
.venv\Scripts\python.exe scripts\render_mining_3d.py --start 2026-07-18T00:00:00Z --end 2026-07-19T00:00:00Z
```

### Dépendances

Seul `plotly` est requis (la page embarque plotly.js, le rendu se fait dans le navigateur). Il est déclaré dans les extras `viz` :

```powershell
.venv\Scripts\python.exe -m pip install -e .[viz]
```

## Ce que fait le script

1. **Extraction** — récupère les blocs cassés (`action = 0`) par les vrais joueurs uniquement (uuid non nul, ce qui exclut `#lava`, `#gravity`, etc.), en ignorant les blocs qu'un joueur avait lui-même posés avant de les recasser (mêmes règles que `extract_mining_events.py`).
2. **Segmentation en sessions** — trie les cassages par joueur et par monde, puis coupe une session dès qu'un trou d'inactivité dépasse `--gap` secondes. Les micro-sessions sous `--min-blocks` blocs sont écartées.
3. **Analyse** — features de trajectoire et score de suspicion V1 calculés par session et par minerai cible (voir [readmeAnalyse.md](readmeAnalyse.md)), embarqués dans la page.
4. **Rendu 3D** — sérialise le tout dans une page HTML unique : bandeau de contrôle, scène Plotly plein écran, panneau d'analyse latéral.

Si tu fournis `--window`, `--start` ou `--end`, la page ne charge que les événements de cette fenêtre temporelle. Sans ces options, elle prend toute la base.

## Fonctionnalités de la page

La scène occupe tout l'écran et s'adapte à la taille de la fenêtre. Le bandeau supérieur regroupe les contrôles :

- **Sélecteur de session** groupé par joueur. Chaque entrée affiche plage horaire, nombre de blocs et de minerais ; à côté, le rappel complet de la session (joueur, monde, date, durée).
- **Fenêtre temporelle** : un double curseur (début / fin, à la seconde) restreint l'affichage à une sous-période de la session. Les deux champs d'heure à côté des curseurs sont éditables directement (`HH:MM:SS`), et le bouton **Session entière** réinitialise la fenêtre.
- **Stats live** à droite : blocs, minerais (avec pourcentage) et minerai surveillé recalculés à chaque changement de fenêtre — pratique pour isoler le moment exact d'un pic de diamants.
- **Bouton Analyse** : affiche / masque le panneau latéral.
- **Bouton Métriques ?** : ouvre la page d'explication intégrée — lecture de la scène, définition de chaque métrique, fonctionnement du score (fermeture par Échap, clic hors de la fenêtre ou bouton).

Le panneau d'analyse (à droite) :

- **Minerai surveillé** : sélecteur (diamant, or, fer, cuivre…) qui pilote l'analyse, les stats du bandeau et la mise en avant des marqueurs de la cible dans la scène (plus gros, ⭐ dans la légende).
- **Score de suspicion** en anneau (0-100) avec verdict coloré (RAS / à surveiller / fortement suspect).
- **Indicateurs du score** : trois jauges (rendement, détour entre filons, virages vers le filon) avec valeur brute et rappel du seuil de référence.
- **Écart au corpus (modèle d'anomalie)** : jauge 0-100 de l'Isolation Forest entraîné sur la vraie base (voir [readmeAnalyse.md](readmeAnalyse.md)), avec repère à 50 (seuil de contamination) et la feature qui tire le plus l'écart. Affiché uniquement si un modèle existe dans `data/models/` pour le minerai choisi (`scripts/train_anomaly_model.py`) — c'est un second regard indépendant du score V1, et atypique ≠ tricheur.
- **Détails de la session** : durée, blocs, blocs/min, filons, blocs entre filons, segments droits H/V, virages/100, pas verticaux.
- **Localisation** : monde et coordonnées du centre de la zone minée, avec un bouton **Copier /tp** qui met `/tp @s X Y Z` dans le presse-papier — à coller en jeu pour aller inspecter sur place (le point visé est un bloc cassé au milieu du parcours, donc dans la galerie creusée ; penser à se mettre dans le bon monde avant).
- **Classement des sessions** trié par score pour le minerai choisi — cliquer sur une ligne ouvre la session.

L'analyse porte sur la **session entière** : la fenêtre temporelle filtre la scène 3D, pas le score.

## Partager la page (anonymisation)

Pour diffuser la preview sans exposer les vrais pseudos :

```powershell
.venv\Scripts\python.exe scripts\render_mining_3d.py --anonymize
```

Le fichier généré (`mining_sessions_3d_anon.html`) remplace chaque joueur par un pseudo inventé (Silexis, Cobaltin, Grimval…), attribué de façon déterministe. La correspondance réel → inventé n'est affichée **que dans la console** ; elle n'apparaît nulle part dans le HTML (vérifié : aucun pseudo ni uuid réel dans le fichier). Le script d'analyse CLI accepte le même flag (`analyze_mining_sessions.py --anonymize`) pour produire CSV et figure anonymisés.

Dans la scène :

- **Roche et terrain** en petits points gris translucides : c'est la forme des tunnels et des galeries.
- **Minerais** en marqueurs plus gros, colorés par famille (diamant, émeraude, or, redstone, lapis, cuivre, fer, charbon…). Les variantes deepslate sont regroupées avec leur famille.
- **Panneau de filtres** en haut à droite de la scène : une case à cocher par couche (roche, progression, chaque famille de minerai avec son compte dans la fenêtre affichée), plus une ligne **Tout afficher** pour tout cocher / décocher d'un coup.
- **Trace de progression** : une ligne fine relie les blocs dans l'ordre chronologique de cassage — c'est elle qui rend le x-ray flagrant, quand elle file en ligne droite de filon en filon.
- **Hover** sur chaque bloc : matériau exact, heure UTC, altitude Y.
- **Navigation 3D** Plotly standard : rotation au clic, zoom à la molette, l'échelle des axes respecte les proportions réelles du monde (`aspectmode='data'`). La caméra est conservée quand on ajuste la fenêtre temporelle, et réinitialisée quand on change de session.

## Lire les motifs

Sur la base de test, les trois profils sont immédiatement reconnaissables :

| Session | Motif visuel | Verdict |
|---|---|---|
| [IxLikexYoou44](reports/figures/preview_IxLikexYoou44.png) | Tunnels en vers de terre qui serpentent de filon en filon, 386 diamants en 58 min | X-ray |
| [acsterix](reports/figures/preview_acsterix.png) | Tunnels dirigés + traversées verticales, 171 diamants et presque aucun autre minerai | X-ray |
| [Utruna](reports/figures/preview_Utruna.png) | Grille de galeries parallèles à Y≈-60, minerais trouvés le long des branches | Minage légitime |

Un joueur légitime creuse des structures régulières et ramasse tout ce qu'il croise ; un x-rayeur trace des chemins irréguliers mais étonnamment efficaces, presque exclusivement vers le minerai de valeur.

## Export PNG statique

Le plus simple est le bouton appareil photo de la barre d'outils Plotly (en haut à droite de la scène), qui télécharge la vue courante en PNG.

Pour automatiser une capture de la page complète (bandeau compris), Edge en mode headless fonctionne bien :

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --use-angle=swiftshader --virtual-time-budget=15000 --window-size=1920,1000 --screenshot="capture.png" "file:///E:/Projet/AntiCheat/reports/figures/mining_sessions_3d.html"
```
