# Prédiction de coupures d'électricité en hôpitaux

Projet de data science / machine learning pour **prédire les coupures d'électricité** dans les hôpitaux :
probabilité, moment estimé, durée et causes probables.

## Données d'entraînement

Le pipeline d'entraînement est **multi-hôpitaux** :
- **Lacor Hospital** (Ouganda, historique 15 min rééchantillonné à l'heure) — site de référence
- **ERIC NHS** (UK, profils horaires par hôpital, 10 sites)
- **NYC LL84** (USA, profils horaires bâtiment, 5 sites)
- **Enrichissement contextuel** selon disponibilité : météo (Open-Meteo Archive + Forecast), qualité de l'air, sismique (USGS), catastrophes (GDACS), signal médiatique (GDELT), tempêtes (NOAA), réseau électrique (Electricity Maps)

La variable cible est `is_outage` (1 = coupure, 0 = pas de coupure).

## Structure du projet

```
PI_26/
├── data/
│   ├── raw/                  ← données brutes (APIs, Excel, CSV)
│   │   ├── eric/             ← profils horaires ERIC NHS
│   │   ├── nyc_ll84/         ← profils horaires NYC LL84
│   │   └── noaa_storm/       ← cache NOAA Storm Events
│   ├── processed/            ← hospital_merged.csv (multi-hôpitaux fusionné)
│   └── features/             ← features_dataset.csv (dataset d'entraînement)
├── models/
│   ├── baseline_rf.joblib          ← meilleur modèle brut (nom historique conservé)
│   ├── calibrated_rf.joblib        ← modèle calibré isotonique (utilisé par l'app)
│   ├── shap_explainer.joblib       ← TreeExplainer SHAP
│   ├── shap_values.npz             ← SHAP values du test set
│   ├── shap_feature_importance.csv ← importance SHAP globale
│   ├── feature_importance.csv      ← importance MDI
│   ├── model_comparison.csv        ← tableau comparatif RF / XGB / LGBM
│   └── training_summary.json       ← hyperparamètres + métriques
├── docs/
│   ├── DOCUMENTATION_DONNEES_ET_APIS.md
│   └── DOCUMENTATION_MODELE_ET_PREDICTIONS.md
├── src/
│   ├── data/
│   │   ├── ingest_consumption.py        ← Lacor Hospital (Excel → CSV)
│   │   ├── ingest_eric.py               ← ERIC NHS (10 sites UK, profils horaires)
│   │   ├── ingest_nyc_ll84.py           ← NYC LL84 (5 sites NYC, profils horaires)
│   │   ├── ingest_meteo.py              ← Open-Meteo Archive (météo historique)
│   │   ├── ingest_openmeteo_forecast.py ← Open-Meteo Forecast (prévisions 7 j)
│   │   ├── ingest_air_quality.py        ← Open-Meteo Air Quality (PM, ozone, dust)
│   │   ├── ingest_usgs_earthquake.py    ← USGS Earthquake Catalog (sismique)
│   │   ├── ingest_gdacs.py              ← GDACS (catastrophes naturelles UE/OCHA)
│   │   ├── ingest_gdelt.py              ← GDELT DOC 2.0 (signal médiatique)
│   │   ├── ingest_noaa_storm.py         ← NOAA Storm Events (USA uniquement)
│   │   ├── ingest_electricitymaps.py    ← Electricity Maps (charge & mix réseau)
│   │   └── preprocessing.py             ← rééchantillonnage + fusion multi-hôpitaux
│   ├── features/
│   │   └── build_features.py     ← feature engineering (temporel, charge, météo,
│   │                                  air, sismique, GDACS, GDELT, NOAA…)
│   ├── models/
│   │   └── train_baseline.py     ← pipeline RF/XGB/LGBM + calibration + SHAP
│   └── utils/
│       ├── config.py             ← configuration centralisée (APIs, hôpitaux)
│       └── io.py                 ← helpers I/O + logging
├── app.py                    ← interface Streamlit (28 hôpitaux, SHAP local)
├── run_pipeline.py           ← exécution complète du pipeline (CLI train|live)
├── requirements.txt          ← dépendances avec versions fixées
└── README.md
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

`run_pipeline.py` orchestre 4 étapes :

1. **Ingestion** — appelle séquentiellement les scripts `src/data/ingest_*.py`
2. **Preprocessing** ([src/data/preprocessing.py](src/data/preprocessing.py)) — rééchantillonnage Lacor 15 min → 1 h, fusion multi-hôpitaux et signaux externes
3. **Feature engineering** ([src/features/build_features.py](src/features/build_features.py)) — features temporelles cycliques, rolling de charge, interactions météo, dérivées GDELT/GDACS/NOAA/USGS/Air Quality
4. **Entraînement** ([src/models/train_baseline.py](src/models/train_baseline.py)) :
   1. Split temporel 80/20 **par hôpital**
   2. **GridSearchCV** + **TimeSeriesSplit** (5 folds) pour RF / XGBoost / LightGBM
   3. Comparaison → sélection automatique du meilleur (F1 sur CV)
   4. **Calibration isotonique** (`CalibratedClassifierCV`)
   5. Évaluation hold-out (brut + calibré)
   6. **SHAP TreeExplainer** + sauvegarde des artefacts

## Métriques et features importantes

Les métriques exactes du run courant sont écrites par `train_baseline.py` dans `models/training_summary.json` (modèle gagnant, hyperparamètres, F1 CV, accuracy, precision, recall, ROC AUC, Brier brut + calibré).

Les classements de features sont disponibles ici :
- `models/feature_importance.csv` — importance MDI du modèle gagnant
- `models/shap_feature_importance.csv` — importance SHAP globale (|mean|)

Le modèle calibré (`calibrated_rf.joblib`) produit des probabilités fiables (calibration isotonique sur 3 folds temporels). C'est lui qui est chargé par défaut dans l'app.

## Sources de données

| Source | Type | Granularité | Utilisation |
|---|---|---|---|
| Lacor Hospital (Zenodo) | Excel | 15 min → horaire | Dataset principal + cible `is_outage` |
| NHS ERIC 2022-23 | Statistiques publiées + profils dérivés | horaire | 10 hôpitaux anglais |
| NYC LL84 | CSV mensuel + profils horaires dérivés | horaire | 5 hôpitaux NYC |
| Open-Meteo Archive | API publique | horaire | Météo historique 2022 |
| Open-Meteo Forecast | API publique | horaire (7 j) | Prévisions pour le mode live / app |
| Open-Meteo Air Quality | API publique | horaire | PM2.5, PM10, ozone, dust, AQI |
| USGS Earthquake | API publique | événementiel | Séismes M ≥ 3 dans 500 km |
| GDACS (UE/OCHA) | API publique | événementiel | Catastrophes naturelles majeures |
| GDELT DOC 2.0 | API publique | quotidien | Signal médiatique (4 thèmes) |
| NOAA Storm Events | CSV public NCEI | événementiel | Tempêtes USA (Phoenix) |
| Electricity Maps | API token | horaire | Charge & mix réseau de zone |

### Données ERIC NHS

Les données [ERIC (Estates Returns Information Collection)](https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection) sont une collecte annuelle obligatoire des NHS Trusts en Angleterre. Le script [src/data/ingest_eric.py](src/data/ingest_eric.py) génère des profils horaires réalistes (8 760 h/hôpital) calibrés sur les statistiques publiées ERIC 2022-23 pour 10 hôpitaux NHS.

## Interface Streamlit

L'application [app.py](app.py) propose :
- **28 hôpitaux** sélectionnables (Afrique, Asie, UK, USA), filtrables sur les sources réelles uniquement (Lacor + ERIC + NYC)
- **Prédiction en temps réel** : analyse des 72 dernières heures avec waterfall SHAP local
- **Mode prévisionnel** : trajectoire de risque sur 7 jours via Open-Meteo Forecast
- **Simulation manuelle** : scénario personnalisé avec explications SHAP locales
- **Ajustement par profil** : adaptation au réseau électrique de chaque hôpital (fiabilité estimée + stabilité du réseau)
- **Gestion d'erreurs** : messages explicatifs si le modèle ou les données sont manquants

## Hôpitaux couverts

| Catégorie | Nb | Hôpitaux |
|---|---|---|
| Référence (terrain) | 1 | Lacor (Ouganda) |
| ERIC NHS (UK) | 10 | St Thomas', Guy's, John Radcliffe, Addenbrooke's, Manchester Royal, Leeds General, Birmingham Heartlands, Royal Victoria Newcastle, Royal Devon, King's College |
| NYC LL84 (USA) | 5 | Bellevue, NYU Tisch, NYP Brooklyn Methodist, Elmhurst, Lincoln |
| Profils estimés `africa_grid` | 12 | Kenyatta, Tikur Anbessa, Groote Schuur, Dhaka, Fann, Parirenyatwa, Muhimbili, LUTH Lagos, Korle Bu, Ibn Sina, Kasr Al Ainy, CHUK Kigali |

Le fichier [src/utils/config.py](src/utils/config.py) référence les coordonnées de 19 sites (`HOSPITAL_LOCATIONS`) utilisés par les ingestions géo-localisées (Open-Meteo, USGS, GDACS, NOAA, GDELT, Electricity Maps).

## Facteurs utilisés (features)

Le dataset de features contient plus de 100 colonnes numériques ; le modèle exclut explicitement les colonnes à fuite, constantes ou redondantes définies dans `COLS_TO_DROP` (`src/models/train_baseline.py`).

Familles de facteurs effectivement utilisées :

- **Charge/énergie** : `total_load_kw`, `solar_pv_kw`, `base_load_kw`, `load_rolling_*`, `solar_ratio`, `peak_ratio`, etc.
- **Temporels** : `hour`, `day_of_week`, `month`, `is_weekend`, encodages cycliques.
- **Météo** : température, humidité, vent, pluie, pression, rayonnement + interactions.
- **Qualité de l'air** : PM2.5, PM10, ozone, poussières, AQI + agrégats temporels.
- **Sismique / catastrophes** : variables USGS (`eq_*`) et GDACS (`gdacs_*`).
- **Signal média événementiel** : variables GDELT (`gdelt_*`) pour les sites configurés.
- **Réseau local** : variables Electricity Maps (`em_*`) selon la zone électrique de l'hôpital.

### Disponibilité des sources selon hôpital

La disponibilité de certaines familles dépend de la configuration source dans [config.py](src/utils/config.py) :

- **GDELT (`gdelt_*`)** : Lacor et Phoenix (requêtes thématiques configurées dans `GDELT_QUERIES`).
- **NOAA Storm (`storm_*`)** : sites USA uniquement (filtre dans `NOAA_STORM_FILTERS`).
- **GDACS (`gdacs_*`)** : tous les sites listés dans `GDACS_FILTERS` (Afrique, Asie, USA).
- **Météo, air quality, USGS, Electricity Maps** : exploités sur l'ensemble des hôpitaux de `HOSPITAL_LOCATIONS`.
- **NHS (UK)** : météo / air quality / USGS / Electricity Maps disponibles ; pas de GDELT/NOAA configurés.

Pour les hôpitaux affichés en mode `africa_grid` dans l'app, les signaux externes site-spécifiques sont neutralisés (mis à 0) pour éviter de prédire avec les signaux d'un autre site (cf. `_neutralize_external_signals` dans [app.py](app.py)).

## Temps réel : périmètre exact

Le système n'est **pas** un flux streaming strict (seconde par seconde). Il fonctionne en :

- **Mode `train`** : données historiques (principalement 2022).
- **Mode `live`** : fenêtre glissante récente (`--window-days`) avec rafraîchissement par appels API.

On parle donc de **quasi temps réel** / **near real-time** : données récentes agrégées par pas horaire, pas de streaming continu.
