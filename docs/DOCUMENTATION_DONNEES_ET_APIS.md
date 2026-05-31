# Documentation des données et APIs

> **Mise à jour 2026-05** — ce document décrit l'état réel du pipeline tel
> qu'orchestré par `run_pipeline.py` (modes `train` et `live`).

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Source 1 — Lacor Hospital (dataset principal de la cible)](#2-source-1--lacor-hospital-dataset-principal-de-la-cible)
3. [Source 2 — NHS ERIC (10 hôpitaux UK)](#3-source-2--nhs-eric-10-hôpitaux-uk)
4. [Source 3 — NYC LL84 (5 hôpitaux USA)](#4-source-3--nyc-ll84-5-hôpitaux-usa)
5. [Source 4 — Open-Meteo (Archive + Forecast)](#5-source-4--open-meteo-archive--forecast)
6. [Source 5 — Open-Meteo Air Quality](#6-source-5--open-meteo-air-quality)
7. [Source 6 — USGS Earthquake Catalog](#7-source-6--usgs-earthquake-catalog)
8. [Source 7 — GDACS (catastrophes)](#8-source-7--gdacs-catastrophes)
9. [Source 8 — GDELT DOC 2.0 (signal médiatique)](#9-source-8--gdelt-doc-20-signal-médiatique)
10. [Source 9 — NOAA Storm Events (USA)](#10-source-9--noaa-storm-events-usa)
11. [Source 10 — Electricity Maps (réseau local)](#11-source-10--electricity-maps-réseau-local)
12. [Schéma de fusion des données](#12-schéma-de-fusion-des-données)
13. [Dictionnaire des variables](#13-dictionnaire-des-variables)
14. [Modes train / live et fenêtres temporelles](#14-modes-train--live-et-fenêtres-temporelles)
15. [Sources historiques retirées du pipeline](#15-sources-historiques-retirées-du-pipeline)

---

## 1. Vue d'ensemble

Le projet agrège **10 sources de données** complémentaires pour prédire les
coupures d'électricité dans les hôpitaux. La cible (`is_outage`) provient
exclusivement du dataset **Lacor Hospital** (relevés terrain). Les autres
sources construisent le contexte (consommation comparée, météo, pollution,
événements, réseau électrique).

| Catégorie | Sources principales | Usage |
|-----------|---------------------|-------|
| Consommation hospitalière | Lacor (terrain), ERIC NHS (UK), NYC LL84 (USA) | Variable cible (Lacor) + profils de charge (16 sites) |
| Météorologie historique | Open-Meteo Archive | Features climatiques |
| Météorologie prévisionnelle | Open-Meteo Forecast | Onglet « Prévisions J+7 » de l'app |
| Qualité de l'air | Open-Meteo Air Quality | Pollution, dust, AQI |
| Risque sismique | USGS Earthquake Catalog | Magnitude, stress 24 h / 7 j |
| Catastrophes naturelles | GDACS (JRC/OCHA) | Inondations, cyclones, sécheresses, etc. |
| Signal médiatique | GDELT DOC 2.0 | 4 thèmes (Lacor + Phoenix uniquement) |
| Tempêtes USA | NOAA Storm Events | Phoenix uniquement |
| Réseau électrique local | Electricity Maps API | Zone, charge MW, mix, carbone |

Le pipeline s'exécute via `run_pipeline.py` (CLI : `--mode {train,live}`,
`--window-days`, `--fast`, `--grid-scale`, `--cv-folds`,
`--shap-sample-size`, `--no-full-artifacts`).

```
ingest_consumption  →  ingest_meteo  →  ingest_eric  →  ingest_nyc_ll84
        ↓                ↓                ↓                  ↓
ingest_gdelt → ingest_noaa_storm → ingest_air_quality
        ↓                ↓                ↓
ingest_usgs_earthquake → ingest_gdacs → ingest_openmeteo_forecast
                            ↓
                ingest_electricitymaps
                            ↓
                  preprocessing (fusion multi-hôpitaux)
                            ↓
                  build_features (feature engineering)
                            ↓
                  train_baseline (RF / XGB / LGBM + calibration + SHAP)
```

Toutes les ingestions sont enveloppées dans un `try/except` : si une source
externe est indisponible, le pipeline continue avec les autres signaux et
l'entraînement final ne plante pas.

---

## 2. Source 1 — Lacor Hospital (dataset principal de la cible)

### Description

Le dataset principal est celui du **St. Mary's Hospital Lacor** situé à
Gulu, dans le nord de l'Ouganda. C'est un hôpital de 482 lits alimenté par
un mix réseau / solaire / générateur diesel. **C'est la seule source qui
contient la variable cible `is_outage`** ; tous les autres hôpitaux sont
donc utilisés en train comme contexte de profil de charge.

### Métadonnées

| Attribut | Valeur |
|----------|--------|
| **Source** | Zenodo |
| **DOI** | `10.5281/zenodo.7466652` |
| **Format** | Excel (.xlsx), feuille "Sheet1" |
| **Résolution** | 15 minutes |
| **Période** | 1er janvier 2022 → 31 décembre 2022 |
| **Volume brut** | 35 040 lignes × 7 colonnes |
| **Volume horaire** | 8 760 lignes × 8 colonnes |
| **Taux de coupures (horaire)** | ~9.7 % des heures |
| **Fichier local** | `data/raw/lacor_hospital.xlsx` |
| **Script** | `src/data/ingest_consumption.py` |

### Colonnes brutes

| Colonne originale | Colonne renommée | Type | Description |
|-------------------|-----------------|------|-------------|
| `Unnamed: 0` | `datetime` | datetime | Horodatage (15 min) |
| `Solar PV kW` | `solar_pv_kw` | float | Production solaire photovoltaïque (kW) |
| `Total load kW` | `total_load_kw` | float | Consommation électrique totale (kW) |
| `Generators kW` | `generators_kw` | float | Production des générateurs diesel (kW) |
| `Sterilization kW` | `sterilization_kw` | float | Consommation stérilisation (kW) |
| `Base load kW` | `base_load_kw` | float | Charge de base (kW) |
| `Grid avail` | `grid_available` | int (0/1) | 1 = réseau disponible, 0 = coupure |

### Variable cible

```
is_outage = 1 - grid_available
```

- `is_outage = 1` → coupure de réseau en cours
- `is_outage = 0` → réseau fonctionnel

### Rééchantillonnage 15 min → 1 h

Effectué par `src/data/preprocessing.py` (`resample_lacor_hourly`) :

| Variable | Règle d'agrégation horaire |
|----------|----------------------------|
| `solar_pv_kw`, `total_load_kw`, `generators_kw`, `sterilization_kw`, `base_load_kw` | Moyenne |
| `grid_available` | Moyenne → renommée `grid_availability_ratio` (∈ [0, 1]) |
| `is_outage` | Max → 1 si au moins une coupure dans l'heure |

Résultat : **8 760 lignes horaires**.

---

## 3. Source 2 — NHS ERIC (10 hôpitaux UK)

### Description

**ERIC** (Estates Returns Information Collection) est une collecte annuelle
**obligatoire** de tous les NHS Trusts en Angleterre. Elle contient les
données d'utilités (électricité, gaz, eau), les coûts et la surface de
chaque site hospitalier.

| Attribut | Valeur |
|----------|--------|
| **Source** | NHS England Digital |
| **URL officielle** | https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection |
| **Édition de référence** | ERIC 2022-23 (publiée le 14 décembre 2023) |
| **Couverture** | ~1 200 sites hospitaliers en Angleterre |
| **Énergie totale NHS** | 11.1 TWh (2022-23) |
| **Coût moyen électricité** | ~£115/MWh |
| **Script** | `src/data/ingest_eric.py` |
| **Répertoire local** | `data/raw/eric/` |

### Stratégie d'accès

Le site NHS Digital bloque les accès programmatiques (HTTP 403). Le script
`ingest_eric.py` adopte donc une double stratégie :

1. **Si `data/raw/eric/eric_site_level.csv` existe** : chargement direct.
2. **Sinon** : génération d'un dataset réaliste à partir des statistiques
   agrégées publiées (consommations annuelles + ratios kWh/m² ERIC).

### Hôpitaux référencés (10 sites)

| Site | Code | Trust | Ville | Lits | Surface (m²) | Électricité (GWh/an) |
|------|------|-------|-------|------|---------------|----------------------|
| St Thomas' Hospital | RJ121 | Guy's & St Thomas' | London | 840 | 150 000 | 82 |
| Guy's Hospital | RJ122 | Guy's & St Thomas' | London | 400 | 82 000 | 48 |
| John Radcliffe Hospital | RTH01 | Oxford Uni. Hospitals | Oxford | 832 | 120 000 | 62 |
| Addenbrooke's Hospital | RGT01 | Cambridge Uni. Hospitals | Cambridge | 1 000 | 160 000 | 78 |
| Manchester Royal Infirmary | R0A01 | Manchester Uni. | Manchester | 752 | 115 000 | 58 |
| Leeds General Infirmary | RR801 | Leeds Teaching | Leeds | 700 | 100 000 | 52 |
| Birmingham Heartlands | RQ301 | Uni. Hospitals Birmingham | Birmingham | 660 | 95 000 | 46 |
| Royal Victoria Infirmary | RA701 | Newcastle Hospitals | Newcastle | 900 | 130 000 | 68 |
| Royal Devon & Exeter | RA401 | Royal Devon Uni. | Exeter | 600 | 80 000 | 38 |
| King's College Hospital | RXH01 | King's College | London | 950 | 140 000 | 72 |

### Génération des profils horaires (8 760 h × 10 sites)

Le script génère **8 760 heures** de données par hôpital en modélisant :

| Composante | Formule | Description |
|------------|---------|-------------|
| **Cycle journalier** | `0.85 + 0.15 × sin(π(h−7)/13)` si 7h-20h, sinon `0.60 + 0.10 × sin(πh/24)` | Pic 10h-14h, creux nocturne |
| **Saisonnalité** | `1.0 + 0.15 × cos(2π(m−1)/12)` | Consommation plus haute en hiver (chauffage UK) |
| **Week-end** | `× 0.82` si samedi/dimanche | Réduction d'activité |
| **Bruit** | `N(1.0, 0.05)` | Variabilité stochastique |
| **Coupures UK** | `P(outage) ≈ 0.0016 × peak_stress × winter_stress` | Fiabilité 99.5%, pic en hiver |

Variables générées : `datetime`, `total_load_kw`, `solar_pv_kw`,
`base_load_kw`, `sterilization_kw`, `is_outage`, `grid_available`,
`generators_kw`, `site_code`, `site_name`.

> ⚠️ Le `is_outage` synthétique sur les sites NHS est volontairement très
> peu fréquent (≈ 0.5 %). En pratique, la cible utile pour l'entraînement
> reste celle de Lacor.

---

## 4. Source 3 — NYC LL84 (5 hôpitaux USA)

### Description

**NYC Local Law 84** est une obligation déclarative de la ville de New York
imposant aux propriétaires de bâtiments > 25 000 ft² de publier leur
consommation d'énergie annuelle. Le dataset est public sur OpenData NYC
(`data.cityofnewyork.us`, dataset `5zyy-y8am`, ~120 hôpitaux NYC publiés).

| Attribut | Valeur |
|----------|--------|
| **Source** | OpenData NYC — LL84 Energy & Water Data |
| **Script** | `src/data/ingest_nyc_ll84.py` |
| **Répertoire local** | `data/raw/nyc_ll84/` |
| **Volume** | 8 760 h × 5 sites |

### Hôpitaux référencés (5 sites)

| Site | Code interne | Surface (m²) | Conso annuelle (kWh) |
|------|--------------|--------------|----------------------|
| Bellevue Hospital Center | `nyc_bellevue` | 211 475 | 52 960 248 |
| NYU Langone Tisch Hospital | `nyc_nyu_tisch` | 64 040 | 45 139 152 |
| NewYork-Presbyterian Brooklyn Methodist | `nyc_nyp_brooklyn` | 126 587 | 32 396 762 |
| Elmhurst Hospital Center | `nyc_elmhurst` | 89 366 | 30 507 199 |
| Lincoln Medical Center | `nyc_lincoln` | 110 874 | 31 236 421 |

### Désagrégation horaire

Le script applique un profil horaire NYC-spécifique (climatisation Con
Edison, pic estival) à la consommation annuelle déclarée pour produire
8 760 lignes par hôpital. La météo locale (Open-Meteo) est jointe ensuite
côté app via `load_nyc_features`.

---

## 5. Source 4 — Open-Meteo (Archive + Forecast)

### 5.1. Open-Meteo Archive (météo historique)

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://archive-api.open-meteo.com/v1/archive` |
| **Authentification** | Aucune (clé optionnelle) |
| **Rate limit** | 10 000 requêtes/jour (gratuit) |
| **Script** | `src/data/ingest_meteo.py` |
| **Fichier de sortie** | `data/raw/meteo_<hospital_key>.csv` |

13 variables horaires demandées (`METEO_HOURLY_VARS` dans
`src/utils/config.py`) :

```
temperature_2m, relative_humidity_2m, dew_point_2m,
wind_speed_10m, wind_gusts_10m, precipitation,
surface_pressure, shortwave_radiation, cloud_cover,
visibility, et0_fao_evapotranspiration, cape, weathercode
```

### 5.2. Open-Meteo Forecast (prévisions J+7)

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://api.open-meteo.com/v1/forecast` |
| **Horizon** | `METEO_FORECAST_DAYS = 7` jours |
| **Script** | `src/data/ingest_openmeteo_forecast.py` |
| **Fichier de sortie** | `data/raw/meteo_forecast_<hospital_key>.csv` |
| **Usage** | Onglet « Prévisions J+7 » de l'app Streamlit |

### Sites interrogés

19 sites listés dans `HOSPITAL_LOCATIONS` (lacor + 11 sites Afrique/Asie +
Phoenix + 5 sites NHS UK). Chaque site est requêté indépendamment, ce qui
fait un fichier `meteo_<hospital_key>.csv` par hôpital.

### Fusion avec les données de consommation

Côté preprocessing, la jointure se fait par hôpital sur l'horodatage
exact (résolution horaire alignée). Côté app (mode live / forecast), la
jointure se fait via `pd.merge_asof` avec une tolérance de 24 h.

---

## 6. Source 5 — Open-Meteo Air Quality

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://air-quality-api.open-meteo.com/v1/air-quality` |
| **Authentification** | Aucune |
| **Granularité** | Horaire |
| **Script** | `src/data/ingest_air_quality.py` |
| **Fichier de sortie** | `data/raw/air_quality_<hospital_key>.csv` |

Variables (`AIR_QUALITY_VARS`) :

| Variable | Unité | Description |
|----------|-------|-------------|
| `pm10` | µg/m³ | Particules fines PM10 |
| `pm2_5` | µg/m³ | Particules fines PM2.5 |
| `carbon_monoxide` | µg/m³ | CO |
| `nitrogen_dioxide` | µg/m³ | NO₂ |
| `sulphur_dioxide` | µg/m³ | SO₂ |
| `ozone` | µg/m³ | O₃ |
| `dust` | µg/m³ | Poussières (utile pour Lacor en saison sèche) |
| `uv_index` | — | Indice UV |
| `european_aqi` | — | AQI européen agrégé |

Les colonnes finales sont préfixées `air_*` après preprocessing
(`air_pm2_5`, `air_pm10`, …).

---

## 7. Source 6 — USGS Earthquake Catalog

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://earthquake.usgs.gov/fdsnws/event/1/query` |
| **Authentification** | Aucune |
| **Rayon de recherche** | 500 km autour de l'hôpital (`USGS_SEARCH_RADIUS_KM`) |
| **Magnitude minimale** | 3.0 (`USGS_MIN_MAGNITUDE`) |
| **Script** | `src/data/ingest_usgs_earthquake.py` |
| **Fichier de sortie** | `data/raw/usgs_earthquake_<hospital_key>.csv` |

Toutes les zones de `HOSPITAL_LOCATIONS` sont interrogées. Les colonnes
finales sont préfixées `eq_*` (`eq_stress`, `eq_max_mag_24h`,
`eq_distance_min_km`, `eq_recent_count_24h`, …).

---

## 8. Source 7 — GDACS (catastrophes)

API publique sans clé, gérée conjointement par l'UE (JRC) et l'OCHA.

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH` |
| **Authentification** | Aucune |
| **Couverture** | Mondiale, sites filtrés via `GDACS_FILTERS` (code ISO3) |
| **Script** | `src/data/ingest_gdacs.py` |
| **Fichier de sortie** | `data/raw/gdacs_<hospital_key>.csv` |

Mapping niveau d'alerte → score numérique (`GDACS_ALERT_SCORE`) :

| Niveau | Score |
|--------|-------|
| Green | 1.0 |
| Orange | 2.0 |
| Red | 3.0 |

Types d'événements (`GDACS_EVENT_TYPES`) : flood, cyclone, earthquake,
volcano, drought, wildfire, tsunami. Les colonnes finales sont préfixées
`gdacs_*`.

Sites couverts : tous les hôpitaux d'Afrique + Asie + Phoenix + NHS UK
(les NHS n'ont en pratique que peu d'alertes Orange/Red).

---

## 9. Source 8 — GDELT DOC 2.0 (signal médiatique)

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://api.gdeltproject.org/api/v2/doc/doc` |
| **Authentification** | Aucune |
| **Granularité** | Quotidien (puis interpolé en horaire) |
| **Script** | `src/data/ingest_gdelt.py` |
| **Fichier de sortie** | `data/raw/gdelt_<hospital_key>.csv` |

### Sites configurés (`GDELT_QUERIES`)

Seuls **2 hôpitaux** ont des requêtes thématiques codées : **Lacor**
(Ouganda) et **Phoenix** (Arizona). Pour les autres sites, les colonnes
GDELT n'existent pas dans le dataset final.

### 4 thèmes par site

| Thème | Description | Exemple Lacor |
|-------|-------------|----------------|
| `power` | Coupures, délestage, problèmes réseau | `(uganda OR gulu) (blackout OR "power outage" OR UMEME OR "load shedding")` |
| `weather` | Événements météo extrêmes | `(uganda OR gulu) (storm OR flood OR "heavy rain" OR lightning)` |
| `health` | Surcharge urgences, crises sanitaires | `(uganda OR gulu) (hospital OR clinic) (emergency OR crisis OR disruption)` |
| `disaster` | Épidémies, catastrophes, déplacés | `(uganda OR gulu) (epidemic OR outbreak OR ebola OR refugee)` |

Les colonnes finales sont préfixées `gdelt_<theme>_*` (`gdelt_power_vol`,
`gdelt_power_tone`, `gdelt_power_vol_7d`, `gdelt_power_anomaly`,
`gdelt_power_stress`, …).

---

## 10. Source 9 — NOAA Storm Events (USA)

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/` |
| **Authentification** | Aucune |
| **Cache local** | `data/raw/noaa_storm/` |
| **Script** | `src/data/ingest_noaa_storm.py` |

### Sites filtrés (`NOAA_STORM_FILTERS`)

Seul **Phoenix (Arizona, comté Maricopa)** est filtré. Tous les autres sites
ne reçoivent pas de signal NOAA (les colonnes `storm_*` sont absentes ou à
0).

### Catégories d'événements (`NOAA_STORM_EVENT_GROUPS`)

| Groupe | Types d'événements |
|--------|---------------------|
| `thunderstorm` | Thunderstorm Wind, Lightning, Tornado, Hail |
| `flood` | Flood, Flash Flood, Heavy Rain |
| `wind` | High Wind, Strong Wind |
| `heat` | Heat, Excessive Heat |
| `winter` | Winter Storm, Ice Storm, Blizzard, Heavy Snow |
| `dust` | Dust Storm, Dust Devil |

Les colonnes finales sont préfixées `storm_*` (`storm_active`,
`storm_event_count`, `storm_active_6h`, `storm_active_24h`,
`storm_count_24h`, `storm_magnitude_max`, etc.).

---

## 11. Source 10 — Electricity Maps (réseau local)

API commerciale (token gratuit pour usage perso/recherche, payant pour
usage pro). Couverture mondiale, granularité horaire.

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://api.electricitymap.org` |
| **Token** | Variable d'env. `ELECTRICITY_MAPS_TOKEN` |
| **Script** | `src/data/ingest_electricitymaps.py` (`run` train + `run_live` live) |
| **Fichier de sortie** | `data/raw/electricitymaps_<hospital_key>.csv` |

### Endpoints exploités

| Endpoint | Usage |
|----------|-------|
| `/v4/zone` | Résolution lat/lon → zone réseau |
| `/v4/total-load/latest` | Charge totale instantanée (MW) |
| `/v4/total-load/history` | Charge horaire des 24 dernières heures |
| `/v4/carbon-intensity/latest` | Intensité carbone (gCO₂/kWh) |
| `/v4/carbon-intensity/history` | Intensité carbone horaire 24 h |
| `/v4/electricity-mix/history` | Mix de production (renouv. / fossile) |

### Mapping zone par hôpital (`HOSPITAL_ELECTRICITY_ZONES`)

| Hôpital | Zone Electricity Maps |
|---------|------------------------|
| Lacor | `UG` |
| Phoenix | `US-SW-AZPS` (Arizona Public Service) |
| Kenyatta | `KE` |
| Tikur Anbessa | `ET` |
| Groote Schuur | `ZA` |
| Dhaka | `BD` |
| Fann | `SN` |
| Parirenyatwa | `ZW` |
| Muhimbili | `TZ` |
| LUTH | `NG` |
| Korle Bu | `GH` |
| Ibn Sina | `MA` |
| Kasr Al Ainy | `EG` |
| CHUK | `RW` |
| NHS UK (5 sites) | `GB` |

> Les sites NYC LL84 ne sont pas dans `HOSPITAL_ELECTRICITY_ZONES` et
> n'ont donc pas de fichier `electricitymaps_*.csv`. L'app gère ce cas en
> affichant un message « Electricity Maps non disponible ».

### Colonnes finales

Préfixées `em_*` :

| Colonne | Description |
|---------|-------------|
| `em_zone` | Code zone (string) |
| `em_total_load_mw` | Charge totale instantanée (MW) |
| `em_carbon_intensity_gco2_kwh` | Intensité carbone (gCO₂/kWh) |
| `em_renewable_pct` | % renouvelable du mix |
| `em_fossil_pct` | % fossile du mix |
| `em_low_carbon_pct` | % bas carbone |

Le bandeau temps réel de l'app (« État réseau local ») affiche les 24 h
glissantes : MW courants, stress vs moyenne 24 h, intensité carbone, mix.
Il propose aussi une **estimation de la conso hôpital** :

```
hospital_load_kw_est = avg_load_kw × (em_total_load_mw_now / em_total_load_mw_avg_24h)
```

---

## 12. Schéma de fusion des données

```
        Bases hospitalières horaires (Lacor + ERIC NHS + NYC LL84)
                                │
                                ▼
                    ┌───────────────────────────────┐
                    │ Enrichissement externe        │
                    │ (par hôpital, par horodatage) │
                    └───────────────┬───────────────┘
                                    │
   ┌──────────────┬─────────────┬───┴───┬─────────────┬──────────────┐
   ▼              ▼             ▼       ▼             ▼              ▼
Open-Meteo   Air Quality      USGS    GDACS        GDELT       Electricity
(historique)                                       (Lacor /        Maps
                                                   Phoenix)
                                                   + NOAA
                                                   (Phoenix)
   └──────────────┴─────────────┴───────┴─────────────┴──────────────┘
                                    │
                                    ▼
                  `data/processed/hospital_merged.csv`
                                    │
                                    ▼
                  `data/features/features_dataset.csv`
                                    │
                                    ▼
                       `src/models/train_baseline.py`
                                    │
                                    ▼
                        `models/calibrated_model.joblib`
```

### Types de jointure

| Jointure | Type | Clé | Tolérance |
|----------|------|-----|-----------|
| Lacor (15 min) → horaire | Resample temporel | `datetime` | — |
| Hôpital + Météo / Air / EM | `merge` exact (puis fallback `merge_asof`) | `(hospital, datetime)` | ±1 h |
| Hôpital + GDELT (quotidien) | Forward-fill jour → heures | `(hospital, date)` | — |
| Hôpital + GDACS / USGS / NOAA | Agrégation événementielle horaire | `(hospital, datetime)` | — |

### Volumétrie indicative

| Source | Lignes | Hôpitaux concernés |
|--------|--------|---------------------|
| Lacor (horaire) | 8 760 | 1 |
| ERIC NHS (horaire) | 87 600 | 10 |
| NYC LL84 (horaire) | 43 800 | 5 |
| **Dataset final fusionné** | **~140 160** | **16 hôpitaux temps réel** |

Le nombre de colonnes du dataset final dépend des sources disponibles au
moment du run (≈ 130-140 colonnes typiquement). Le nombre exact de features
utilisées par le modèle est calculé dynamiquement par `train_baseline.py`
(voir `prepare_data` + `COLS_TO_DROP`).

---

## 13. Dictionnaire des variables

### Variables brutes (après preprocessing)

| Variable | Source | Type | Unité | Description |
|----------|--------|------|-------|-------------|
| `datetime` | toutes | datetime | — | Horodatage horaire |
| `hospital` | toutes | string | — | Clé hôpital (`lacor_uganda`, `st_thomas_nhs`, …) |
| `total_load_kw` | Lacor / ERIC / NYC | float | kW | Consommation totale |
| `solar_pv_kw` | Lacor / ERIC | float | kW | Production solaire |
| `generators_kw` | Lacor | float | kW | Production générateurs |
| `sterilization_kw` | Lacor / ERIC | float | kW | Consommation stérilisation |
| `base_load_kw` | Lacor / ERIC / NYC | float | kW | Charge de base |
| `grid_availability_ratio` | Lacor | float | [0,1] | Fraction de l'heure avec réseau |
| `is_outage` | Lacor (réelle) / ERIC (synthétique) | int | 0/1 | **Variable cible** |
| `temperature_2m` | Open-Meteo | float | °C | Température |
| `relative_humidity_2m` | Open-Meteo | float | % | Humidité relative |
| `dew_point_2m` | Open-Meteo | float | °C | Point de rosée |
| `wind_speed_10m` | Open-Meteo | float | km/h | Vent à 10 m |
| `wind_gusts_10m` | Open-Meteo | float | km/h | Rafales à 10 m |
| `precipitation` | Open-Meteo | float | mm | Précipitations |
| `surface_pressure` | Open-Meteo | float | hPa | Pression au sol |
| `shortwave_radiation` | Open-Meteo | float | W/m² | Rayonnement solaire |
| `cloud_cover` | Open-Meteo | float | % | Couverture nuageuse |
| `visibility` | Open-Meteo | float | m | Visibilité |
| `et0_fao_evapotranspiration` | Open-Meteo | float | mm | Évapotranspiration FAO |
| `cape` | Open-Meteo | float | J/kg | Énergie convective |
| `weathercode` | Open-Meteo | int | — | Code météo WMO |
| `air_pm2_5`, `air_pm10`, `air_ozone`, … | Air Quality | float | µg/m³ | Pollution |
| `air_european_aqi`, `air_uv_index` | Air Quality | float | — | Index agrégés |
| `eq_*` | USGS | float | mixte | Stress sismique (24 h, 7 j, max mag…) |
| `gdacs_*` | GDACS | mixte | — | Score d'alerte, type, durée |
| `gdelt_*` | GDELT | mixte | — | Volume / tonalité / anomalie / stress par thème |
| `storm_*` | NOAA | mixte | — | Tempêtes USA (Phoenix uniquement) |
| `em_*` | Electricity Maps | mixte | — | Charge MW, mix, carbone |

### Variables dérivées (features engineering)

→ Voir [`DOCUMENTATION_MODELE_ET_PREDICTIONS.md`](DOCUMENTATION_MODELE_ET_PREDICTIONS.md) pour le détail des
features réellement utilisées à l'entraînement (liste calculée
dynamiquement via `COLS_TO_DROP` dans `src/models/train_baseline.py`).

---

## 14. Modes train / live et fenêtres temporelles

`run_pipeline.py` expose deux modes :

### Mode `train` (défaut)

```bash
python run_pipeline.py
# ou explicite
python run_pipeline.py --mode train
```

- Météo Open-Meteo : année 2022 entière
- Air Quality : année 2022 entière
- USGS / GDACS / GDELT / NOAA : année 2022
- Electricity Maps : ingestion historique complète (`run`)

### Mode `live`

```bash
python run_pipeline.py --mode live --window-days 30
```

- Météo Open-Meteo : `[today − window_days, today]`
- Air Quality : idem
- USGS / GDACS : `[today − window_days, today]` (datetime UTC)
- GDELT / NOAA : année courante (`datetime.now().year`)
- Electricity Maps : appel `run_live(window_hours = window_days × 24)`
- L'ingestion de consommation Lacor reste sur le fichier 2022 (pas de
  flux temps réel public pour cet hôpital).

### Note importante

Le système n'est **pas** un flux streaming strict. Il fonctionne en :

- **train** : données historiques (principalement 2022)
- **live** : fenêtre glissante récente avec rafraîchissement par appels API

On parle donc de **quasi temps réel / near real-time** : données récentes
agrégées par pas horaire, pas de streaming continu seconde par seconde.

---

## 15. Sources historiques retirées du pipeline

Trois sources mentionnées dans les premières versions du projet
**ne sont plus actives** dans la run courante :

| Source | Statut | Raison |
|--------|--------|--------|
| **OMS / WHO GHO** (`HCF_REL_ELECTRICITY`) | ❌ Plus ingérée par `run_pipeline.py` | La fiabilité OMS est désormais utilisée uniquement comme paramètre statique de l'app (`adjust_for_hospital_profile`) |
| **Eskom / EskomSePush** | ❌ Plus ingérée | Couverture limitée à l'Afrique du Sud, remplacée par Electricity Maps + GDACS qui couvrent toutes les zones |
| **Phoenix Hospital** (Excel github) | ❌ Plus ingéré comme source de consommation | Aucune cible `is_outage`, profils trop éloignés des sites de production. Phoenix reste référencé dans `HOSPITAL_LOCATIONS` pour les ingestions géo (Open-Meteo, USGS, GDACS, GDELT, NOAA) |
| **Kaggle Hospital Energy** | ❌ Jamais branché en production | Reste disponible localement à titre exploratoire (`data/raw/kaggle_hospital/`) |

Les fichiers historiques (`data/raw/who_reliability.csv`,
`data/raw/sa_electricity/*`) restent sur disque à titre de traçabilité,
mais ne sont plus chargés par `preprocessing.py`.
