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

> ⚠️ **Seules les coupures de Lacor sont réellement observées.** Les coupures
> ERIC/NYC sont générées par une formule stochastique (profils de consommation
> réalistes mais cible synthétique). C'est pourquoi l'entraînement se fait
> **par défaut sur Lacor uniquement** (`--scope real`) : voir la section
> *Exécution*. Le multi-hôpitaux complet reste disponible via `--scope all`.

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
│   ├── baseline_model.joblib       ← meilleur modèle brut (RF / XGB / LGBM selon la run)
│   ├── calibrated_model.joblib     ← modèle calibré isotonique (utilisé par l'app)
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
# Pipeline complet (ingestion historique 2022 → features → entraînement + SHAP)
# Par défaut : --scope real (entraînement sur coupures réellement observées)
python run_pipeline.py

# Entraînement multi-hôpitaux complet (inclut coupures synthétiques ERIC/NYC)
python run_pipeline.py --scope all

# Pipeline en mode "live" (fenêtre glissante récente, par défaut 30 jours)
python run_pipeline.py --mode live --window-days 30

# Pipeline rapide pour itération (CV réduit, grille compacte, SHAP échantillonné)
python run_pipeline.py --fast

# Tuning fin (override CV folds, taille SHAP, taille de grille…)
python run_pipeline.py --grid-scale compact --cv-folds 3 --shap-sample-size 2000

# Interface Streamlit
streamlit run app.py
```

Tous les flags CLI sont définis dans `run_pipeline.py` (`--mode {train,live}`, `--window-days`, `--fast`, `--grid-scale {compact,full}`, `--cv-folds`, `--shap-sample-size`, `--no-full-artifacts`, `--scope {real,all}`).

> **Portée d'entraînement (`--scope`)** — `real` (défaut) n'entraîne et n'évalue
> que sur les hôpitaux à coupures **réellement observées** (Lacor). Les sites
> ERIC/NYC ont des coupures **générées par une formule stochastique** : les
> inclure (`--scope all`) revient à entraîner sur ~94 % de bruit et gonfle
> artificiellement le F1 global. Le défaut `real` garantit des métriques honnêtes.

## Pipeline d'entraînement

`run_pipeline.py` orchestre 4 étapes :

1. **Ingestion** — appelle séquentiellement les scripts `src/data/ingest_*.py`
2. **Preprocessing** ([src/data/preprocessing.py](src/data/preprocessing.py)) — rééchantillonnage Lacor 15 min → 1 h, fusion multi-hôpitaux et signaux externes
3. **Feature engineering** ([src/features/build_features.py](src/features/build_features.py)) — features temporelles cycliques, rolling de charge, interactions météo, dérivées GDELT/GDACS/NOAA/USGS/Air Quality
4. **Entraînement** ([src/models/train_baseline.py](src/models/train_baseline.py)) :
   1. Split temporel 80/20 **par hôpital** (train réordonné chronologiquement pour une vraie CV temporelle)
   2. **GridSearchCV** + **TimeSeriesSplit** (5 folds) pour RF / XGBoost / LightGBM
   3. Comparaison → sélection automatique du meilleur (F1 sur CV)
   4. **Calibration adaptative** (`--calibration auto`) : compare *aucune calibration* / isotonique / sigmoïde sur une validation interne et ne recalibre que si le Brier s'améliore d'une marge nette (un GBM est souvent déjà bien calibré)
   5. Évaluation hold-out (brut + calibré)
   6. **SHAP TreeExplainer** + sauvegarde des artefacts

> **Signaux externes exclus du modèle (`config.EXTERNAL_SIGNAL_PREFIXES`)** — les
> familles `gdelt_/gdacs_/eq_/air_/em_/noaa_/storm_` sont calculées et stockées
> dans `features_dataset` (inspection), mais **exclues du jeu de features du
> modèle** : elles sont mises à 0 hors Lacor à l'inférence (décalage
> entraînement/service) et les volumes de presse GDELT agissaient comme un proxy
> temporel spurieux. Le modèle s'appuie donc sur ~49 features robustes
> (météo + charge + temporel + historique des coupures), identiques en
> entraînement et en service pour tous les hôpitaux.

## Métriques et features importantes

Les métriques exactes du run courant sont écrites par `train_baseline.py` dans `models/training_summary.json` (modèle gagnant, hyperparamètres, F1 CV, accuracy, precision, recall, ROC AUC, Brier brut + calibré).

**Run courante** (`models/training_summary.json`, `--scope real`, 49 features, signaux externes exclus) :
- **Modèle gagnant : RandomForest** (`n_estimators=300`, `max_depth=25`, `min_samples_leaf=8`, `class_weight={0:1, 1:10}`)
- Hold-out test (brut)    : F1 = 0.82 · ROC AUC = 0.99 · Brier = 0.034 · Precision = 0.82 · Recall = 0.82
- Hold-out test (calibré, servi) : F1 = 0.77 · ROC AUC = 0.98 · Brier = 0.031 · Precision = 0.84 · Recall = 0.72

> Ces chiffres portent **uniquement sur des coupures réellement observées** (Lacor,
> hold-out n=1752, 9.4 % de coupures). Ils sont plus bas — mais honnêtes — que les
> ~0.88 de l'ancienne run `--scope all`, qui étaient gonflés par des coupures
> synthétiques (F1 synthétique = 1.0) et un proxy temporel GDELT. La calibration
> est sélectionnée automatiquement (ici isotonique : Brier 0.034 → 0.031).

### Validation temporelle (généralisation dans le temps)

Comme on ne dispose que d'**un site × une année**, la robustesse temporelle est
évaluée explicitement par [`src/models/backtest.py`](src/models/backtest.py)
(`python -m src.models.backtest`) — bien plus honnête qu'un hold-out unique :

- **Hold-out chronologique** (train mois 1–9 → test oct–déc) : F1 = 0.78 · ROC AUC = 0.98 · Recall = 0.75 · Brier = 0.040
- **Backtest walk-forward** (origine glissante, 6 folds mensuels) : F1 = **0.75 ± 0.04** [0.71–0.81] · Recall = 0.73 · ROC AUC = **0.96 ± 0.03** · Brier = 0.046

Lecture : la discrimination (ROC AUC) reste élevée toute l'année ; le F1 progresse
avec l'historique disponible (≈0.71 aux premiers mois → ≈0.81 en fin d'année).
⚠️ Ceci mesure la stabilité **dans le temps sur Lacor** — pas la généralisation à
**d'autres sites** (qui exigerait des coupures réelles multi-sites). Détail par
mois : `models/backtest_by_month.csv` + `models/backtest_summary.json`.

Les classements de features sont disponibles ici :
- `models/feature_importance.csv` — importance MDI du modèle gagnant
- `models/shap_feature_importance.csv` — importance SHAP globale (|mean|)

Le fichier `calibrated_model.joblib` est chargé par défaut dans l'app. Il contient le **gagnant** courant de la comparaison RF / XGBoost / LightGBM, servi avec la stratégie de calibration retenue (`auto` : aucune / isotonique / sigmoïde selon le Brier de validation — `calibration_method` dans `training_summary.json`). L'app sait encore lire l'ancien nom `calibrated_rf.joblib` en repli.

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
- **28 hôpitaux** sélectionnables (Afrique, Asie, UK, USA), filtrables sur les sources réelles uniquement (Lacor + ERIC + NYC) via la bascule « Données réelles uniquement »
- **Bandeau réseau temps réel** (Electricity Maps) par hôpital : zone, charge MW, intensité carbone, mix renouvelable/fossile, conso hôpital estimée
- **Onglet 1 — Prédiction historique** : période d'analyse au choix (7 presets : 72 h, mois, saisons, année 2022) et probabilité par heure + SHAP waterfall local
- **Onglet 2 — Prévisions J+7** : trajectoire de risque heure par heure sur 7 jours via Open-Meteo Forecast (presets : seuils 50% / 70%, top 5 heures critiques, synthèse par jour)
- **Onglet 3 — Simulation manuelle** : 13 paramètres (3 temporel · 4 énergie · 6 météo) + jauge de risque + waterfall SHAP + comparaison aux conditions moyennes
- **Ajustement par profil** : adaptation au réseau électrique de chaque hôpital (fiabilité OMS estimée + stabilité du réseau, voir `adjust_for_hospital_profile`)
- **Garde-fou features** : détection automatique d'une désynchronisation entre le dataset (`features_dataset.csv`) et le modèle entraîné (`feature_names_in_`)
- **Gestion d'erreurs** : messages explicatifs si le modèle ou les données sont manquants

## Hôpitaux couverts

L'app Streamlit propose **28 hôpitaux** ; en mode strict (« Données réelles uniquement », activé par défaut) seuls **16 hôpitaux** sont sélectionnables (Lacor + 10 NHS ERIC + 5 NYC LL84). Les profils `africa_grid` ne sont visibles que lorsque la bascule est désactivée.

| Catégorie | Nb | Hôpitaux |
|---|---|---|
| Référence (terrain) | 1 | Lacor (Ouganda) |
| ERIC NHS (UK) | 10 | St Thomas', Guy's, John Radcliffe, Addenbrooke's, Manchester Royal, Leeds General, Birmingham Heartlands, Royal Victoria Newcastle, Royal Devon, King's College |
| NYC LL84 (USA) | 5 | Bellevue, NYU Tisch, NYP Brooklyn Methodist, Elmhurst, Lincoln |
| Profils estimés `africa_grid` | 12 | Kenyatta, Tikur Anbessa, Groote Schuur, Dhaka, Fann, Parirenyatwa, Muhimbili, LUTH Lagos, Korle Bu, Ibn Sina, Kasr Al Ainy, CHUK Kigali |

Le fichier [src/utils/config.py](src/utils/config.py) référence les coordonnées de 19 sites (`HOSPITAL_LOCATIONS`) utilisés par les ingestions géo-localisées (Open-Meteo, USGS, GDACS, NOAA, GDELT, Electricity Maps). Les profils `africa_grid` clonent le profil temporel de Lacor mis à l'échelle (`avg_load_kw`) puis y injectent météo locale + Electricity Maps.

## Facteurs utilisés (features)

Le dataset de features contient plus de 100 colonnes numériques ; le modèle exclut explicitement les colonnes à fuite, constantes ou redondantes définies dans `COLS_TO_DROP` (`src/models/train_baseline.py`).

Familles de facteurs effectivement utilisées :

- **Charge/énergie** : `total_load_kw`, `solar_pv_kw`, `base_load_kw`, `load_rolling_*`, `load_diff_*`, `solar_ratio`, `peak_ratio`, `base_load_ratio`, etc.
- **Historique coupures** : `hours_since_last_outage`, `last_outage_duration_h`, `outage_frequency_7d`, `avg_outage_duration_7d`, `outage_trend_7d` (toutes calculées avec un `shift(1)` par hôpital pour éviter le leakage)
- **Temporels** : `hour`, `day_of_week`, `month`, `is_weekend`, `is_public_holiday`, encodages cycliques (`hour_sin/cos`, `month_sin/cos`)
- **Météo** : température, humidité, point de rosée, vent (vitesse + rafales), pluie, pression, rayonnement, CAPE, weathercode + interactions (`temp_humidity_interaction`, `wind_precipitation_interaction`, `heat_stress`, `solar_available`)
- **Météo avancée** : `cloud_cover_pct`, `visibility_m`, `evapotranspiration`, `rain_intensity`, `thermal_amplitude_24h`, `humidity_change_3h`, `pressure_change_3h`
- **Qualité de l'air** : PM2.5, PM10, ozone, NO₂, SO₂, CO, dust, UV, AQI européen + moyennes 6 h / 24 h + indicateurs (`air_pollution_high`, `air_dust_storm`, `air_heat_pollution_stress`)
- **Sismique** : variables USGS `eq_*` (`eq_stress`, cumul 24 h / 7 j, magnitude max, distance min, événement majeur)
- **Catastrophes** : variables GDACS `gdacs_*` (alerte 24 h / 7 j, type de catastrophe, combo tempête × catastrophe)
- **Tempêtes (USA)** : variables NOAA `storm_*` (orage, inondation, vent, chaleur, hiver, poussière)
- **Signal média événementiel** : variables GDELT `gdelt_*` (4 thèmes × volume / tonalité / anomalie / stress) pour les sites configurés (Lacor, Phoenix)
- **Réseau local** : variables Electricity Maps `em_*` (zone, charge MW, intensité carbone gCO₂/kWh, % renouvelable / fossile / bas carbone)

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
