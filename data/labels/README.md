# Étiquettes de sessions de minage (vérité terrain)

`session_labels.csv` recense les sessions vérifiées visuellement, une ligne par
session :

| colonne | contenu |
|---|---|
| `pseudo` | pseudo exact (attention à la casse, ex : `Lyjo_`) |
| `world` | monde tel qu'affiché dans la preview (`survie`, `ressource_5`, ...) |
| `start_utc` / `end_utc` | bornes de la session en ISO UTC (`2026-03-04T20:15:33Z`) |
| `label` | `legit`, `suspect`, ou `triche` (bannable au visuel sans doute) |
| `tags` | tags spéciaux, séparés par `;` — pour l'instant : `grotte` (session en cavité naturelle, cumulable avec le verdict) |

## Workflow d'annotation

Générer la preview avec le mode annotation :

```powershell
.venv\Scripts\python.exe scripts\render_mining_3d.py --db <base> --start ... --end ... --annotation --output ...
```

Le panneau de droite gagne une section **Annotation** : trois cases exclusives
(Legit / Suspect / Triche) + le tag **Grotte**. Les annotations persistent dans
le navigateur (localStorage, clé stable pseudo|monde|début — elles survivent à
une régénération de la page). Le bouton **Exporter CSV** télécharge et copie le
tout au format de ce dossier : coller/fusionner dans `session_labels.csv`.

Pourquoi cette clé : les `session_id` internes dépendent des paramètres de
segmentation et changent d'un run à l'autre ; (pseudo, monde, heure de début)
est stable. L'appariement avec une analyse se fait sur pseudo + monde + début.

## Usages prévus

1. **Jeu d'évaluation** : après chaque changement du score V1 / des filtres, on
   vérifie que les sessions `triche` remontent et que les `legit` redescendent.
2. **Validation du filtre grotte** : les tags `grotte` donnent la vérité terrain
   du filtre anti-cavernes (faux positifs ET faux négatifs).
3. **Corpus d'entraînement purgé** : entraîner l'Isolation Forest en excluant
   les sessions `triche` pour que « la session typique » soit apprise sur du
   minage propre (option à ajouter à scripts/train_anomaly_model.py).
4. **À terme** : corpus supervisé pour remplacer l'heuristique V1 (prévu dès
   l'origine, voir readmeAnalyse.md).

Étiquettes joueur déjà connues (fenêtre fév-avr 2026, avant ce fichier) :
x-ray certains = Blackh_ole, wasaby54, Nejmiia, Lyjo_ ; le reste du top score
jugé légitime ou douteux.
