# Maintenance prédictive — condition de la valve hydraulique

Projet Machine Learning I (EFREI ING2-BDML, 2026/2027). Objectif : prédire, pour chaque cycle de production,
si la condition de la valve est **optimale (100 %)** ou **non**, à partir des signaux de pression (PS2) et de débit (FS1).

**Résultats** — validation croisée 5-fold (2000 premiers cycles) : F1 = 1,00 ; test final (205 derniers cycles) :
205/205 cycles correctement classés (accuracy, précision, rappel, F1 et AUC = 1,00). Modèle retenu : régression logistique
sur 10 variables (à égalité parfaite avec Extra Trees, on garde le plus simple).

## Structure
```
├── notebooks/valve_condition_ml.ipynb   # analyse complète (EDA, sélection, 8 modèles, Grid/Random Search, test)
├── src/valve/                           # code réutilisable : features, pipeline, entraînement, prédiction
├── app/                                 # application web Flask (prédiction par numéro de cycle)
├── tests/                               # tests unitaires (pytest)
├── models/                              # modèle entraîné + métriques
├── data/                                # profile.txt, features_all_cycles.csv (signaux bruts : voir data/README.md)
└── Dockerfile · .github/workflows/ci.yml
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Utilisation
| Action | Commande |
|---|---|
| Notebook | `jupyter lab notebooks/valve_condition_ml.ipynb` (placer `PS2.txt` et `FS1.txt` dans `data/`) |
| Réentraîner le modèle | `PYTHONPATH=src python -m valve.train --data-dir data --out-dir .` (PowerShell : `$env:PYTHONPATH="src"` puis `python -m valve.train --data-dir data --out-dir .`) |
| Tests unitaires | `pytest -v` |
| Application web (local) | `PYTHONPATH=src python -m flask --app app.app run` → http://127.0.0.1:5000 |

## Docker
```bash
docker build -t valve-app .
docker run --rm -p 8000:8000 valve-app          # → http://localhost:8000
```
API : `GET /api/predict/<n>` (n de 1 à 2205) · `GET /health`. Exemple : `curl http://localhost:8000/api/predict/500`

## Méthode (résumé)
1. **Prétraitement** : pas de valeurs manquantes ; cible binaire (`valve == 100`) ; classes équilibrées en train (52,6 % d'optimales), plus rares en test (35,6 %).
2. **Variables** : 80 statistiques par cycle (globales, dynamique, 10 segments temporels) → 40 après filtres variance/corrélation → 10 retenues (ANOVA F).
3. **Modèles** : régression logistique, k-NN, arbre, Naive Bayes, SVM, Random Forest, Extra Trees, HistGradientBoosting.
4. **Évaluation** : accuracy, précision, rappel, F1, matrice de confusion, ROC/AUC, écart train/validation (biais-variance).
5. **Optimisation** : Grid Search et Random Search, validation croisée 5-fold ; la sélection de variables est *dans* le pipeline (pas de fuite).

## Limites
Le test ne contient que 205 cycles d'une même installation : 100 % ne garantit pas l'absence d'erreur ailleurs
(borne basse de l'intervalle de confiance à 95 % de l'accuracy ≈ 98,2 %).
