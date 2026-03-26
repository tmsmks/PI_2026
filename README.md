# Prédiction de coupures d'électricité en hôpitaux

Projet de data science / machine learning pour **prédire les coupures d'électricité** dans les hôpitaux :
probabilité, moment estimé, durée et causes probables.

## Données réelles

Le dataset principal est **Lacor Hospital** (Ouganda, 2022) :
- 35 040 mesures toutes les 15 minutes (rééchantillonné à l'heure → 8 760 points)
- Variable cible : `Grid avail` (1 = réseau OK, 0 = coupure) → **9.7% de coupures**
- Sources d'énergie : réseau, solaire (PV), générateur diesel

## Structure du projet

```
PI_26/
├── data/
│   ├── raw/                  ← données brutes (APIs, Excel, CSV)
│   │   └── eric/             ← données ERIC NHS (profils horaires)
│   ├── processed/            ← données nettoyées et fusionnées (8 760 lignes)
│   └── features/             ← dataset final (46 colonnes, 31 features actives)
├── models/
│   ├── baseline_rf.joblib    ← meilleur modèle brut (LightGBM)
│   ├── calibrated_rf.joblib  ← modèle calibré (isotonic, utilisé par l'app)
│   ├── shap_explainer.joblib ← TreeExplainer SHAP
│   ├── shap_values.npz       ← SHAP values du test set
│   ├── shap_feature_importance.csv ← importance SHAP globale
│   ├── feature_importance.csv      ← importance MDI
│   ├── model_comparison.csv        ← tableau comparatif RF / XGB / LGBM
│   └── training_summary.json       ← hyperparamètres + métriques
├── docs/
│   ├── DOCUMENTATION_DONNEES_ET_APIS.md
│   └── DOCUMENTATION_MODELE_ET_PREDICTIONS.md
├── src/
│   ├── data/
│   │   ├── ingest_consumption.py ← chargement Lacor + Phoenix (Excel)
│   │   ├── ingest_eric.py        ← données ERIC NHS (10 hôpitaux anglais)
│   │   ├── ingest_meteo.py       ← météo historique 2022 (Open-Meteo)
│   │   ├── ingest_who.py         ← fiabilité OMS par pays (API)
│   │   ├── ingest_outages.py     ← données Eskom / load shedding
│   │   └── preprocessing.py      ← rééchantillonnage + fusion
│   ├── features/
│   │   └── build_features.py     ← 46 colonnes (31 features actives + 15 exclues)
│   ├── models/
│   │   ├── train_baseline.py     ← pipeline multi-modèles (RF/XGB/LGBM) + SHAP
│   │   └── predict.py            ← inférence + explications
│   └── utils/
│       ├── config.py             ← configuration centralisée
│       └── io.py                 ← fonctions I/O
├── app.py                    ← interface Streamlit (14 hôpitaux, SHAP local)
├── run_pipeline.py           ← exécution complète du pipeline
├── requirements.txt          ← dépendances avec versions fixées
└── Prompt_Project.TXT        ← cahier des charges
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution

```bash
# Pipeline complet (ingestion → features → entraînement + SHAP)
python run_pipeline.py

# Interface Streamlit
streamlit run app.py
```

## Pipeline d'entraînement

Le pipeline (`src/models/train_baseline.py`) suit 7 étapes :

1. **Split temporel** 80/20 (train: janv-oct 2022, test: oct-déc 2022)
2. **GridSearchCV** avec **TimeSeriesSplit** (5 folds glissants) pour chaque modèle
3. **Comparaison** des 3 modèles → sélection automatique du meilleur (F1)
4. **Calibration isotonique** (CalibratedClassifierCV) du gagnant
5. **Évaluation** sur le hold-out test set (brut + calibré)
6. **SHAP TreeExplainer** : explications locales par prédiction
7. Sauvegarde des modèles, métriques et explainer

## Comparaison des modèles (test set)

| Modèle | F1 (CV) | Accuracy | Precision | Recall | F1 (test) | ROC AUC | Brier |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.3403 | **97.3%** | **86.0%** | **85.5%** | **85.7%** | **98.9%** | **0.018** |
| RandomForest | 0.3352 | 93.3% | 60.2% | 84.2% | 70.2% | 97.2% | 0.071 |
| XGBoost | 0.3404 | 91.2% | 51.8% | 86.7% | 64.9% | 96.8% | 0.069 |

### Meilleur modèle : LightGBM

| Paramètre | Valeur |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 15 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

### Calibration des probabilités

| Métrique | Brut | Calibré |
|---|---|---|
| Brier score | 0.0183 | 0.0428 |
| Precision | 86.0% | 98.3% |
| Recall | 85.5% | 35.8% |

Le modèle calibré produit des probabilités plus fiables (quand il dit "30%", c'est vraiment ~30% de chances).

### Top features

**SHAP (importance locale, |mean|) :**

| Feature | SHAP moyen |
|---|---|
| solar_ratio | 2.42 |
| day_of_week | 1.63 |
| solar_pv_kw | 1.10 |
| load_rolling_6h | 1.09 |
| month | 0.96 |
| temp_humidity_interaction | 0.93 |
| surface_pressure | 0.90 |

## Sources de données

| Source | Type | Lignes | Utilisation |
|---|---|---|---|
| Lacor Hospital (Zenodo) | Excel 15 min | 35 040 | Dataset principal + cible |
| Phoenix Hospital (GitHub) | Excel horaire | 8 760 | Benchmark consommation |
| NHS ERIC 2022-23 | CSV/profils | 87 600 | 10 hôpitaux anglais (conso, surface, coûts) |
| Open-Meteo Archive | API horaire | 8 760 | Features météo |
| OMS GHO API | API | 39 | Feature macro (fiabilité pays) |
| Eskom / EskomSePush | CSV | 44 494 | Contexte load shedding |
| Kaggle Hospital Energy | CSV | 10 000 | Dataset complémentaire |

### Données ERIC NHS

Les données [ERIC (Estates Returns Information Collection)](https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection) sont une collecte annuelle obligatoire de tous les NHS Trusts en Angleterre. Le script `src/data/ingest_eric.py` génère des profils horaires réalistes (8 760 h/hôpital) calibrés sur les statistiques publiées ERIC 2022-23 pour 10 hôpitaux NHS.

## Interface Streamlit

L'application (`app.py`) propose :
- **14 hôpitaux** sélectionnables (Afrique, Asie, UK, USA)
- **Prédiction en temps réel** : analyse des 72 dernières heures avec waterfall SHAP
- **Simulation manuelle** : scénario personnalisé avec explications SHAP locales
- **Ajustement par profil** : adaptation au réseau électrique de chaque pays (fiabilité OMS)
- **Gestion d'erreurs** : messages explicatifs si le modèle ou les données sont manquants
