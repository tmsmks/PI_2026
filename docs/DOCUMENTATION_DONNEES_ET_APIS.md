# Documentation des données et APIs

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Source 1 — Lacor Hospital (dataset principal)](#2-source-1--lacor-hospital-dataset-principal)
3. [Source 2 — Phoenix Hospital (benchmark)](#3-source-2--phoenix-hospital-benchmark)
4. [Source 3 — API OMS (WHO GHO)](#4-source-3--api-oms-who-gho)
5. [Source 4 — API Open-Meteo Archive](#5-source-4--api-open-meteo-archive)
6. [Source 5 — Eskom / EskomSePush (Afrique du Sud)](#6-source-5--eskom--eskomsepush-afrique-du-sud)
7. [Source 6 — NHS ERIC (Angleterre)](#7-source-6--nhs-eric-angleterre)
8. [Source 7 — Kaggle Hospital Energy](#8-source-7--kaggle-hospital-energy)
9. [Schéma de fusion des données](#9-schéma-de-fusion-des-données)
10. [Dictionnaire des variables](#10-dictionnaire-des-variables)
11. [Résumé des volumétries](#11-résumé-des-volumétries)

---

## 1. Vue d'ensemble

Le projet agrège **7 sources de données** pour prédire les coupures d'électricité dans les hôpitaux. Ces données couvrent :

| Catégorie | Sources | Usage |
|-----------|---------|-------|
| Consommation hospitalière | Lacor, Phoenix, ERIC NHS | Variable cible + features de charge |
| Météorologie | Open-Meteo Archive | Features climatiques |
| Fiabilité réseau par pays | OMS GHO API | Feature macro-contextuelle |
| Stabilité réseau | Eskom, EskomSePush | Features de load shedding |
| Énergie hospitalière UK | NHS ERIC | Profils de consommation NHS |

Le pipeline s'exécute via `run_pipeline.py` et orchestre les scripts d'ingestion dans l'ordre suivant :

```
ingest_consumption → ingest_outages → ingest_who → ingest_meteo → ingest_eric
    → preprocessing → build_features → train_baseline (RF + XGBoost + LightGBM + SHAP)
```

---

## 2. Source 1 — Lacor Hospital (dataset principal)

### Description

Le dataset principal est celui du **St. Mary's Hospital Lacor** situé à Gulu, dans le nord de l'Ouganda. C'est un hôpital de 482 lits alimenté par un mix réseau/solaire/générateur diesel.

### Métadonnées

| Attribut | Valeur |
|----------|--------|
| **Source** | Zenodo |
| **DOI** | `10.5281/zenodo.7466652` |
| **Format** | Excel (.xlsx), feuille "Sheet1" |
| **Résolution** | 15 minutes |
| **Période** | 1er janvier 2022 — 31 décembre 2022 |
| **Volume** | 35 040 lignes × 7 colonnes |
| **Taux de coupures** | 9.7% des intervalles de 15 min |
| **Fichier local** | `data/raw/lacor_hospital.xlsx` |
| **Script** | `src/data/ingest_consumption.py` → `load_lacor()` |

### Colonnes brutes

| Colonne originale | Colonne renommée | Type | Description |
|-------------------|-----------------|------|-------------|
| `Unnamed: 0` | `datetime` | datetime | Horodatage (15 min) |
| `Solar PV kW` | `solar_pv_kw` | float | Production solaire photovoltaïque (kW) |
| `Total load kW` | `total_load_kw` | float | Consommation électrique totale (kW) |
| `Generators kW` | `generators_kw` | float | Production des générateurs diesel (kW) |
| `Sterilization kW` | `sterilization_kw` | float | Consommation stérilisation (kW) |
| `Base load kW` | `base_load_kw` | float | Charge de base (éclairage, ventilation, etc.) |
| `Grid avail` | `grid_available` | int (0/1) | 1 = réseau disponible, 0 = coupure |

### Variable cible

La variable cible `is_outage` est dérivée de `grid_available` :

```
is_outage = 1 - grid_available
```

- `is_outage = 1` → coupure de réseau en cours
- `is_outage = 0` → réseau fonctionnel

### Rééchantillonnage

Les données sont rééchantillonnées de 15 min à **1 heure** dans `preprocessing.py` :

| Variable | Règle d'agrégation |
|----------|--------------------|
| `solar_pv_kw`, `total_load_kw`, `generators_kw`, `sterilization_kw`, `base_load_kw` | **Moyenne** horaire |
| `grid_available` | **Moyenne** → devient `grid_availability_ratio` (0.0 à 1.0) |
| `is_outage` | **Max** → 1 si au moins une coupure dans l'heure |

Résultat : **8 760 lignes** horaires.

### Statistiques descriptives

| Variable | Moyenne | Écart-type | Min | Max |
|----------|---------|-----------|-----|-----|
| `total_load_kw` | 133 kW | 42 kW | 18 kW | 235 kW |
| `solar_pv_kw` | 19 kW | 31 kW | 0 kW | 130 kW |
| `generators_kw` | 14 kW | 39 kW | 0 kW | 180 kW |
| `base_load_kw` | 113 kW | 36 kW | 15 kW | 200 kW |

---

## 3. Source 2 — Phoenix Hospital (benchmark)

### Description

Dataset de consommation d'un hôpital à Phoenix, Arizona (USA). Il ne contient **pas de variable cible** (pas d'indication de coupure). Il sert de benchmark pour comparer les profils de consommation.

### Métadonnées

| Attribut | Valeur |
|----------|--------|
| **Source** | GitHub (Shahid-Fakhri/Electricity-Consumption) |
| **Format** | Excel (.xlsx), feuille "in" |
| **Résolution** | 1 heure |
| **Volume** | 8 760 lignes × 11 colonnes |
| **Fichier local** | `data/raw/phoenix_hospital.xlsx` |
| **Script** | `src/data/ingest_consumption.py` → `load_phoenix()` |

### Colonnes brutes

| Colonne originale | Colonne renommée | Description |
|-------------------|-----------------|-------------|
| `Date/Time` | `datetime` | Format `MM/DD  HH:MM:SS` (année ajoutée : 2022) |
| `Electricity:Facility [kW](Hourly)` | `total_electricity_kw` | Consommation totale (kW) |
| `Fans:Electricity [kW](Hourly)` | `fans_kw` | Ventilation (kW) |
| `Cooling:Electricity [kW](Hourly)` | `cooling_kw` | Climatisation (kW) |
| `Heating:Electricity [kW](Hourly)` | `heating_kw` | Chauffage électrique (kW) |
| `InteriorLights:Electricity [kW](Hourly)` | `lights_kw` | Éclairage (kW) |
| `InteriorEquipment:Electricity [kW](Hourly)` | `equipment_kw` | Équipements (kW) |
| `Gas:Facility [kW](Hourly)` | `total_gas_kw` | Consommation gaz (kW) |

---

## 4. Source 3 — API OMS (WHO GHO)

### Description

L'Organisation Mondiale de la Santé publie via son API **Global Health Observatory** (GHO) des indicateurs de fiabilité de l'approvisionnement électrique des établissements de santé par pays.

### Endpoints utilisés

#### 4.1. Fiabilité électrique des hôpitaux

| Attribut | Valeur |
|----------|--------|
| **URL de base** | `https://ghoapi.azureedge.net/api` |
| **Endpoint** | `GET /HCF_REL_ELECTRICITY` |
| **Indicateur OMS** | Pourcentage d'établissements avec électricité fiable (sans coupure > 2h) |
| **Authentification** | Aucune |
| **Rate limit** | Non documenté (usage raisonnable recommandé) |
| **Format réponse** | JSON (champ `value` contenant un tableau d'enregistrements) |
| **Script** | `src/data/ingest_who.py` → `fetch_who_reliability()` |
| **Fichier de sortie** | `data/raw/who_reliability.csv` |

**Requête HTTP :**

```
GET https://ghoapi.azureedge.net/api/HCF_REL_ELECTRICITY
Accept: application/json
```

**Structure de la réponse :**

```json
{
  "value": [
    {
      "SpatialDim": "UGA",
      "TimeDim": 2018,
      "Dim1": "RESIDENCEAREATYPE_totl",
      "NumericValue": 50.0,
      "ParentLocationCode": "AFR",
      "ParentLocation": "Africa"
    }
  ]
}
```

**Mapping des colonnes :**

| Champ JSON | Colonne CSV | Description |
|------------|-------------|-------------|
| `SpatialDim` | `country_code` | Code ISO 3 lettres du pays |
| `TimeDim` | `year` | Année de l'enquête |
| `Dim1` | `area_type` | Zone : `urb` (urbain), `rur` (rural), `totl` (total) |
| `NumericValue` | `reliability_pct` | % d'établissements avec électricité fiable |
| `ParentLocationCode` | `region_code` | Code de la région OMS |
| `ParentLocation` | `region` | Nom de la région OMS |

**Pays disponibles (39 enregistrements) :**

| Pays | Code | Fiabilité total (%) | Année |
|------|------|---------------------|-------|
| Sénégal | SEN | 45% | 2019 |
| Niger | NER | 19% | 2015 |
| Bangladesh | BGD | 20% | 2017 |
| Éthiopie | ETH | 23% | 2016 |
| Haïti | HTI | 33% | 2016 |
| Zimbabwe | ZWE | 48% | 2015 |
| Kenya | KEN | 54% | 2018 |
| Népal | NPL | 56% | 2021 |
| Sierra Leone | SLE | 55% | 2018 |
| Tanzanie | TZA | 63% | 2016 |
| Liberia | LBR | 81% | 2018 |
| Honduras | HND | 89% | 2019 |
| Sri Lanka | LKA | 91% | 2017 |
| Bolivie | BOL | 95% | 2019 |

#### 4.2. Table de correspondance pays

| Attribut | Valeur |
|----------|--------|
| **Endpoint** | `GET /Dimension/COUNTRY/DimensionValues` |
| **Script** | `src/data/ingest_who.py` → `fetch_who_countries()` |
| **Fichier de sortie** | `data/raw/who_countries.csv` |

**Colonnes extraites :**

| Champ JSON | Colonne CSV | Description |
|------------|-------------|-------------|
| `Code` | `country_code` | Code ISO 3 lettres |
| `Title` | `country_name` | Nom officiel du pays |
| `ParentCode` | `region_code` | Code région OMS |
| `ParentTitle` | `region` | Nom de la région OMS |

### Utilisation dans le pipeline

La fiabilité OMS est injectée comme **feature statique** dans le preprocessing (`add_who_context`). Pour l'Ouganda (pays de Lacor), la valeur utilisée est **50%** (source OMS 2018, zone `totl` la plus récente).

Deux features dérivées :
- `who_reliability_pct` = 50.0 (valeur brute)
- `reliability_risk` = 1 − 50 / 100 = **0.50** (risque inversé)

---

## 5. Source 4 — API Open-Meteo Archive

### Description

Open-Meteo fournit des données météorologiques historiques horaires gratuites, sans authentification, à partir de modèles de réanalyse (ERA5 / CERRA).

### Configuration de l'API

| Attribut | Valeur |
|----------|--------|
| **URL de base** | `https://archive-api.open-meteo.com/v1/archive` |
| **Méthode** | `GET` |
| **Authentification** | Aucune |
| **Rate limit** | 10 000 requêtes/jour (gratuit) |
| **Format réponse** | JSON |
| **Script** | `src/data/ingest_meteo.py` → `fetch_meteo_archive()` |
| **Fichier de sortie** | `data/raw/meteo_lacor_uganda.csv` (et `meteo_phoenix_usa.csv`) |

**Requête HTTP :**

```
GET https://archive-api.open-meteo.com/v1/archive
  ?latitude=2.77
  &longitude=32.30
  &hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure,shortwave_radiation
  &start_date=2022-01-01
  &end_date=2022-12-31
  &timezone=auto
```

### Paramètres de requête

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `latitude` | 2.77 (Lacor) | Latitude du site |
| `longitude` | 32.30 (Lacor) | Longitude du site |
| `hourly` | 6 variables (voir ci-dessous) | Variables météo demandées |
| `start_date` | `2022-01-01` | Début de la période |
| `end_date` | `2022-12-31` | Fin de la période |
| `timezone` | `auto` | Fuseau horaire automatique |

### Variables météorologiques récupérées

| Variable API | Colonne CSV | Unité | Description |
|-------------|-------------|-------|-------------|
| `temperature_2m` | `temperature_2m` | °C | Température à 2 m du sol |
| `relative_humidity_2m` | `relative_humidity_2m` | % | Humidité relative à 2 m |
| `wind_speed_10m` | `wind_speed_10m` | km/h | Vitesse du vent à 10 m |
| `precipitation` | `precipitation` | mm | Précipitations (pluie + neige) |
| `surface_pressure` | `surface_pressure` | hPa | Pression atmosphérique au sol |
| `shortwave_radiation` | `shortwave_radiation` | W/m² | Rayonnement solaire incident (ondes courtes) |

### Structure de la réponse JSON

```json
{
  "latitude": 2.75,
  "longitude": 32.25,
  "generationtime_ms": 4.5,
  "utc_offset_seconds": 10800,
  "timezone": "Africa/Nairobi",
  "hourly": {
    "time": ["2022-01-01T00:00", "2022-01-01T01:00", ...],
    "temperature_2m": [21.3, 20.8, ...],
    "relative_humidity_2m": [78, 82, ...],
    "wind_speed_10m": [4.2, 3.8, ...],
    "precipitation": [0.0, 0.0, ...],
    "surface_pressure": [884.5, 884.3, ...],
    "shortwave_radiation": [0, 0, ...]
  }
}
```

### Fusion avec les données de consommation

La jonction se fait via `pd.merge_asof` (jointure temporelle au plus proche) dans `preprocessing.py` → `merge_with_meteo()`, avec une tolérance de **1 heure**. Les valeurs manquantes sont comblées par interpolation linéaire, puis forward/backward fill.

### Sites interrogés

| Hôpital | Latitude | Longitude | Pays |
|---------|----------|-----------|------|
| Lacor Hospital | 2.77 | 32.30 | UGA |
| Phoenix Hospital | 33.45 | -112.07 | USA |

---

## 6. Source 5 — Eskom / EskomSePush (Afrique du Sud)

### Description

Deux datasets complémentaires provenant du réseau électrique sud-africain, utilisés pour modéliser le contexte d'instabilité des réseaux africains.

### 6.1. Données de production Eskom

| Attribut | Valeur |
|----------|--------|
| **Source** | Eskom Data Portal |
| **Fichier** | `ESK2033.csv` |
| **Volume** | 43 824 lignes |
| **Période** | 2018 — 2022 |
| **Résolution** | Horaire |
| **Fichier local** | `data/raw/sa_electricity/ESK2033.csv` |
| **Script** | `src/data/ingest_outages.py` → `load_eskom_production()` |

**Colonnes utilisées :**

| Colonne | Type | Description |
|---------|------|-------------|
| `datetime_str` | string | Horodatage |
| `Residual Demand` | float | Demande résiduelle (MW) |
| `Dispatchable Generation` | float | Production pilotable (MW) |
| `Thermal Generation` | float | Production thermique (MW) |
| `Nuclear Generation` | float | Production nucléaire (MW) |
| `Wind` | float | Production éolienne (MW) |
| `PV` | float | Production solaire (MW) |
| `Total RE` | float | Total renouvelables (MW) |
| `Manual Load_Reduction(MLR)` | float | Réduction manuelle de charge (MW) — indicateur de délestage |
| `ILS Usage` | float | Interruptible Load Shedding (MW) |

### 6.2. Historique de load shedding (EskomSePush)

| Attribut | Valeur |
|----------|--------|
| **Source** | EskomSePush (application mobile) |
| **Fichier** | `EskomSePush_history.csv` |
| **Volume** | 670 événements |
| **Fichier local** | `data/raw/sa_electricity/EskomSePush_history.csv` |
| **Script** | `src/data/ingest_outages.py` → `load_loadshedding_history()` |

**Colonnes :**

| Colonne | Type | Description |
|---------|------|-------------|
| `created_at` | datetime | Date/heure de l'événement |
| `stage` | int | Niveau de load shedding (0 = inactif, 2-6 = niveaux croissants) |

### Utilisation dans le pipeline

Les données Eskom sont agrégées en **3 features statiques** dans `preprocessing.py` → `add_loadshedding_context()` :

| Feature | Calcul | Valeur typique | Description |
|---------|--------|----------------|-------------|
| `loadshed_avg_stage` | `mean(stage)` | ~1.5 | Niveau moyen de load shedding |
| `loadshed_max_stage` | `max(stage)` | 6 | Niveau maximum observé |
| `loadshed_pct_active` | `mean(stage > 0)` | ~0.65 | % du temps en load shedding |

---

## 7. Source 6 — NHS ERIC (Angleterre)

### Description

**ERIC** (Estates Returns Information Collection) est une collecte annuelle **obligatoire** de tous les NHS Trusts en Angleterre. Elle contient les données d'utilités (électricité, gaz, eau), les coûts et la surface de chaque site hospitalier.

### Métadonnées

| Attribut | Valeur |
|----------|--------|
| **Source** | NHS England Digital |
| **URL officielle** | https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection |
| **Édition de référence** | ERIC 2022-23 (publiée le 14 décembre 2023) |
| **Couverture** | ~1 200 sites hospitaliers en Angleterre |
| **Énergie totale NHS** | 11.1 TWh (2022-23) |
| **Surface totale NHS** | ~28 millions m² |
| **Coût moyen électricité** | ~£115/MWh |
| **Script** | `src/data/ingest_eric.py` |
| **Répertoire local** | `data/raw/eric/` |

### Accès aux données

Le site NHS Digital bloque les accès programmatiques (code HTTP 403). Le script `ingest_eric.py` adopte une double stratégie :

1. **Si un fichier CSV local est disponible** (`data/raw/eric/eric_site_level.csv`) : chargement direct.
2. **Sinon** : construction d'un dataset réaliste à partir des statistiques agrégées publiées.

### Hôpitaux référencés (10 sites)

| Site | Code | Trust | Ville | Lits | Surface (m²) | Électricité (GWh/an) | Coût (M£) |
|------|------|-------|-------|------|---------------|----------------------|-----------|
| St Thomas' Hospital | RJ121 | Guy's & St Thomas' | London | 840 | 150 000 | 82 | 9.43 |
| Guy's Hospital | RJ122 | Guy's & St Thomas' | London | 400 | 82 000 | 48 | 5.52 |
| John Radcliffe Hospital | RTH01 | Oxford Uni. Hospitals | Oxford | 832 | 120 000 | 62 | 7.13 |
| Addenbrooke's Hospital | RGT01 | Cambridge Uni. Hospitals | Cambridge | 1 000 | 160 000 | 78 | 8.97 |
| Manchester Royal Infirmary | R0A01 | Manchester Uni. | Manchester | 752 | 115 000 | 58 | 6.67 |
| Leeds General Infirmary | RR801 | Leeds Teaching | Leeds | 700 | 100 000 | 52 | 5.98 |
| Birmingham Heartlands | RQ301 | Uni. Hospitals Birmingham | Birmingham | 660 | 95 000 | 46 | 5.29 |
| Royal Victoria Infirmary | RA701 | Newcastle Hospitals | Newcastle | 900 | 130 000 | 68 | 7.82 |
| Royal Devon & Exeter | RA401 | Royal Devon Uni. | Exeter | 600 | 80 000 | 38 | 4.37 |
| King's College Hospital | RXH01 | King's College | London | 950 | 140 000 | 72 | 8.28 |

### Génération des profils horaires

Le script génère **8 760 heures** de données par hôpital en modélisant :

| Composante | Formule | Description |
|------------|---------|-------------|
| **Cycle journalier** | `0.85 + 0.15 × sin(π(h−7)/13)` si 7h-20h, sinon `0.60 + 0.10 × sin(πh/24)` | Pic 10h-14h, creux nocturne |
| **Saisonnalité** | `1.0 + 0.15 × cos(2π(m−1)/12)` | Consommation plus haute en hiver (chauffage UK) |
| **Week-end** | `× 0.82` si samedi/dimanche | Réduction d'activité |
| **Bruit** | `N(1.0, 0.05)` | Variabilité stochastique |
| **Coupures UK** | `P(outage) ≈ 0.0016 × peak_stress × winter_stress` | Fiabilité 99.5%, stress en pic et en hiver |

**Variables générées par hôpital :**

| Colonne | Description |
|---------|-------------|
| `datetime` | Horodatage horaire (2022) |
| `total_load_kw` | Consommation horaire (kW) |
| `solar_pv_kw` | Production solaire si applicable (kW) |
| `base_load_kw` | Charge de base (kW) |
| `sterilization_kw` | Stérilisation (kW) |
| `is_outage` | Indicateur de coupure (0/1) |
| `grid_available` | Disponibilité réseau (0/1) |
| `generators_kw` | Production générateur de secours (kW) |
| `site_code` | Code NHS du site |
| `site_name` | Nom du site |

---

## 8. Source 7 — Kaggle Hospital Energy

### Métadonnées

| Attribut | Valeur |
|----------|--------|
| **Source** | Kaggle |
| **Fichier local** | `data/raw/kaggle_hospital/hospital_communication_energy_system.csv` |
| **Volume** | ~10 000 lignes |
| **Usage** | Dataset complémentaire |

---

## 9. Schéma de fusion des données

```
                    ┌──────────────────┐
                    │  Lacor Hospital   │
                    │  (Excel, 15 min)  │
                    │  35 040 lignes    │
                    └────────┬─────────┘
                             │ resample_lacor_hourly()
                             │ 15 min → 1 heure
                             ▼
                    ┌──────────────────┐
                    │  Lacor horaire    │
                    │  8 760 lignes     │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌───────────────┐ ┌─────────────────┐
│  Open-Meteo     │ │   OMS (WHO)   │ │ Eskom/EskomSePush│
│  Archive API    │ │   GHO API     │ │  (CSV locaux)    │
│  8 760 heures   │ │  39 records   │ │  44 494 lignes   │
└────────┬────────┘ └──────┬────────┘ └────────┬────────┘
         │                 │                   │
         │ merge_asof      │ add_who_context   │ add_loadshedding_context
         │ (±1 heure)      │ (statique)        │ (agrégé)
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ hospital_merged   │
                  │ 8 760 × 17 cols  │
                  └────────┬─────────┘
                           │ build_features()
                           ▼
                  ┌──────────────────┐
                  │ features_dataset  │
                  │ 8 760 × 46 cols  │
                  │ (31 actives)     │
                  └──────────────────┘
```

### Types de jointure

| Jointure | Type | Clé | Tolérance |
|----------|------|-----|-----------|
| Consommation + Météo | `merge_asof` (temporel) | `datetime` | ±1 heure |
| + OMS | Jointure statique | `country_code` | — |
| + Load shedding | Agrégation → valeurs constantes | — | — |

---

## 10. Dictionnaire des variables

### Variables brutes (après preprocessing)

| Variable | Source | Type | Unité | Description |
|----------|--------|------|-------|-------------|
| `datetime` | Lacor | datetime | — | Horodatage horaire |
| `total_load_kw` | Lacor | float | kW | Consommation totale |
| `solar_pv_kw` | Lacor | float | kW | Production solaire |
| `generators_kw` | Lacor | float | kW | Production générateurs |
| `sterilization_kw` | Lacor | float | kW | Consommation stérilisation |
| `base_load_kw` | Lacor | float | kW | Charge de base |
| `grid_availability_ratio` | Lacor | float | [0,1] | Fraction de l'heure avec réseau |
| `is_outage` | Lacor | int | 0/1 | **Variable cible** |
| `temperature_2m` | Open-Meteo | float | °C | Température |
| `relative_humidity_2m` | Open-Meteo | float | % | Humidité |
| `wind_speed_10m` | Open-Meteo | float | km/h | Vent |
| `precipitation` | Open-Meteo | float | mm | Précipitations |
| `surface_pressure` | Open-Meteo | float | hPa | Pression |
| `shortwave_radiation` | Open-Meteo | float | W/m² | Rayonnement solaire |
| `who_reliability_pct` | OMS | float | % | Fiabilité OMS du pays |
| `loadshed_avg_stage` | Eskom | float | — | Niveau moyen de load shedding |
| `loadshed_max_stage` | Eskom | int | — | Niveau max de load shedding |
| `loadshed_pct_active` | Eskom | float | [0,1] | % du temps en load shedding |

### Variables dérivées (features)

→ Voir la documentation `DOCUMENTATION_MODELE_ET_PREDICTIONS.md` pour le détail complet des 31 features actives (sur 46 colonnes totales).

---

## 11. Résumé des volumétries

| Source | Lignes brutes | Lignes après traitement | Colonnes | Poids |
|--------|---------------|------------------------|----------|-------|
| Lacor Hospital | 35 040 | 8 760 | 8 | ~3.5 Mo |
| Phoenix Hospital | 8 760 | 8 760 | 8 | ~1.2 Mo |
| Open-Meteo Lacor | 8 760 | 8 760 | 7 | ~0.8 Mo |
| OMS Fiabilité | 39 | 39 | 6 | ~4 Ko |
| OMS Pays | ~200 | ~200 | 4 | ~15 Ko |
| Eskom Production | 43 824 | 43 824 | 10 | ~5 Mo |
| EskomSePush | 670 | 670 | 2 | ~12 Ko |
| ERIC NHS (résumé) | 10 | 10 | 21 | ~3 Ko |
| ERIC NHS (horaire) | 87 600 | 87 600 | 10/site | ~80 Mo |
| **Dataset final** | — | **8 760** | **46** (31 actives) | **~3 Mo** |
