# Documentation du modèle et des calculs de prédiction

## Table des matières

1. [Vue d'ensemble du pipeline de modélisation](#1-vue-densemble-du-pipeline-de-modélisation)
2. [Feature engineering — les 46 variables](#2-feature-engineering--les-46-variables)
3. [Préparation des données d'entraînement](#3-préparation-des-données-dentraînement)
4. [Comparaison multi-modèles (RF / XGBoost / LightGBM)](#4-comparaison-multi-modèles-rf--xgboost--lightgbm)
5. [Validation croisée temporelle (TimeSeriesSplit)](#5-validation-croisée-temporelle-timeseriessplit)
6. [Calibration des probabilités](#6-calibration-des-probabilités)
7. [Évaluation et résultats](#7-évaluation-et-résultats)
8. [Interprétabilité — SHAP (explications locales)](#8-interprétabilité--shap-explications-locales)
9. [Calcul de la prédiction en temps réel](#9-calcul-de-la-prédiction-en-temps-réel)
10. [Simulation manuelle — construction d'un scénario](#10-simulation-manuelle--construction-dun-scénario)
11. [Correction d'extrapolation (stress hors distribution)](#11-correction-dextrapolation-stress-hors-distribution)
12. [Ajustement par profil d'hôpital](#12-ajustement-par-profil-dhôpital)
13. [Calcul du temps estimé et de la durée](#13-calcul-du-temps-estimé-et-de-la-durée)
14. [Data leakage — identification et correction](#14-data-leakage--identification-et-correction)
15. [Limites connues et pistes d'amélioration](#15-limites-connues-et-pistes-damélioration)

---

## 1. Vue d'ensemble du pipeline de modélisation

Le pipeline transforme les données brutes en prédictions de coupure en 7 étapes :

```
Données brutes      Features        GridSearch       Meilleur      Calibration    SHAP         Inférence
──────────────      ────────        ──────────       ────────      ───────────    ────         ─────────
8 760 heures  ──►   46 cols   ──►   RF / XGB   ──►  LightGBM ──► Isotonique ──► TreeExp ──►  P(coupure)
17 colonnes         31 actives      / LightGBM       300 arbres    3 folds        1752 obs     ∈ [0, 1]
                                    5 folds CV
```

### Fichiers impliqués

| Étape | Script | Entrée | Sortie |
|-------|--------|--------|--------|
| Feature engineering | `src/features/build_features.py` | `data/processed/hospital_merged.csv` | `data/features/features_dataset.csv` |
| Entraînement | `src/models/train_baseline.py` | `data/features/features_dataset.csv` | `models/calibrated_rf.joblib` + `models/baseline_rf.joblib` |
| SHAP | `src/models/train_baseline.py` | Modèle + test set | `models/shap_explainer.joblib` + `models/shap_values.npz` |
| Prédiction (API) | `src/models/predict.py` | Modèle + features | Probabilité + explications |
| Interface | `app.py` | Modèle calibré + SHAP explainer + profil hôpital | Prédiction ajustée + waterfall SHAP |

---

## 2. Feature engineering — les 46 variables

Le dataset final contient **46 colonnes** : 1 datetime, 1 variable cible (`is_outage`), et **44 features** numériques réparties en 5 catégories. Parmi celles-ci, **31 sont utilisées pour l'entraînement** (15 exclues : 8 leakage + 1 datetime + 1 cible + 5 features constantes à importance 0).

### 2.1. Features temporelles (7 variables)

Ces features capturent les cycles journaliers et saisonniers qui influencent la consommation et les coupures.

| Feature | Formule | Intervalle | Raison |
|---------|---------|-----------|--------|
| `hour` | Heure brute (0-23) | [0, 23] | Cycle jour/nuit |
| `day_of_week` | Jour de la semaine (0=lundi) | [0, 6] | Cycle hebdomadaire |
| `month` | Mois (1-12) | [1, 12] | Saisonnalité |
| `is_weekend` | `1 si day_of_week ≥ 5, sinon 0` | {0, 1} | Activité réduite |
| `hour_sin` | `sin(2π × hour / 24)` | [-1, 1] | Encodage cyclique de l'heure |
| `hour_cos` | `cos(2π × hour / 24)` | [-1, 1] | (évite la discontinuité 23h→0h) |
| `month_sin` | `sin(2π × month / 12)` | [-1, 1] | Encodage cyclique du mois |
| `month_cos` | `cos(2π × month / 12)` | [-1, 1] | (évite la discontinuité déc→jan) |

**Pourquoi l'encodage cyclique ?** Un arbre de décision ne comprend pas que `hour=23` et `hour=0` sont voisins. Les transformations sin/cos projettent le temps sur un cercle continu.

### 2.2. Features de consommation (7 variables)

Ces features capturent le niveau de charge, sa dynamique et sa variabilité.

| Feature | Formule | Unité | Raison |
|---------|---------|-------|--------|
| `load_rolling_6h` | `mean(total_load_kw, window=6h)` | kW | Tendance court terme |
| `load_rolling_24h` | `mean(total_load_kw, window=24h)` | kW | Tendance journalière |
| `load_std_24h` | `std(total_load_kw, window=24h)` | kW | Instabilité de charge |
| `load_diff_1h` | `total_load_kw[t] − total_load_kw[t−1]` | kW | Variation horaire |
| `load_diff_24h` | `total_load_kw[t] − total_load_kw[t−24]` | kW | Variation jour-à-jour |
| `load_pct_change_1h` | `(load[t] − load[t−1]) / load[t−1]` | ratio | Variation relative |
| `peak_ratio` | `total_load_kw / load_rolling_24h` | ratio | Ratio pic vs moyenne |

**Statistiques des rolling features :**
- `min_periods=1` pour éviter les NaN en début de série
- Les valeurs infinies dans `load_pct_change_1h` et `peak_ratio` sont remplacées par 0 ou 1

### 2.3. Features de sources d'énergie (8 variables)

| Feature | Formule | Intervalle | Raison |
|---------|---------|-----------|--------|
| `solar_ratio` | `solar_pv_kw / total_load_kw` | [0, 1] | Part du solaire |
| `base_load_ratio` | `base_load_kw / total_load_kw` | [0, 1] | Part de la charge de base |
| `generator_active` | `1 si generators_kw > 1.0` | {0, 1} | Générateur en marche |
| `generator_ratio` | `generators_kw / total_load_kw` | [0, 1] | Part du générateur |
| `grid_availability_rolling_6h` | `mean(grid_availability_ratio, 6h)` | [0, 1] | Stabilité récente du réseau |
| `recent_outages_6h` | `sum(is_outage, 6h)` | [0, 6] | Coupures récentes (6h) |
| `recent_outages_24h` | `sum(is_outage, 24h)` | [0, 24] | Coupures récentes (24h) |

> **Attention data leakage** : `generator_active`, `generator_ratio`, `grid_availability_rolling_6h`, `recent_outages_6h`, `recent_outages_24h` sont **exclues** de l'entraînement car elles contiennent de l'information sur la cible (voir section 12).

### 2.4. Features météorologiques (11 variables)

6 variables brutes + 5 variables dérivées :

**Variables brutes (de l'API Open-Meteo) :**

| Feature | Unité | Valeur typique Lacor |
|---------|-------|---------------------|
| `temperature_2m` | °C | 18 — 33°C |
| `relative_humidity_2m` | % | 30 — 95% |
| `wind_speed_10m` | km/h | 0 — 25 km/h |
| `precipitation` | mm | 0 — 40 mm |
| `surface_pressure` | hPa | 870 — 890 hPa |
| `shortwave_radiation` | W/m² | 0 — 900 W/m² |

**Variables dérivées :**

| Feature | Formule | Raison |
|---------|---------|--------|
| `temp_humidity_interaction` | `temperature_2m × relative_humidity_2m / 100` | Index de chaleur simplifié |
| `wind_precipitation_interaction` | `wind_speed_10m × precipitation` | Intensité des intempéries |
| `solar_available` | `1 si shortwave_radiation > 50 W/m²` | Énergie solaire disponible |
| `heat_stress` | `1 si temperature_2m > 30°C` | Seuil de stress thermique |
| `storm_risk` | `1 si wind > 40 km/h OU precipitation > 10 mm` | Conditions de tempête |

### 2.5. Features macro-contextuelles (5 variables)

| Feature | Source | Formule | Valeur pour Lacor |
|---------|--------|---------|-------------------|
| `who_reliability_pct` | OMS | Directe | 50.0 |
| `reliability_risk` | OMS | `1 − who_reliability_pct / 100` | 0.50 |
| `loadshed_avg_stage` | Eskom | Moyenne du stage | ~1.5 |
| `loadshed_max_stage` | Eskom | Maximum du stage | 6 |
| `loadshed_pct_active` | Eskom | `mean(stage > 0)` | ~0.65 |

> **Note** : Ces 5 features + `storm_risk` (section 2.4) sont constantes sur la série mono-hôpital → importance **0.0%**. Elles sont **exclues de l'entraînement** via `COLS_TO_DROP` mais conservées dans le dataset pour de futurs modèles multi-pays.

---

## 3. Préparation des données d'entraînement

### 3.1. Sélection des features

Le dataset contient 38 colonnes. Avant l'entraînement, on supprime les colonnes non-features et celles qui causeraient du data leakage :

```
COLS_TO_DROP = [
    "datetime",                      # Non-feature (identifiant)
    "is_outage",                     # Variable cible
    "grid_availability_ratio",       # Contient directement la cible
    "generators_kw",                 # Conséquence de la coupure
    "generator_active",              # Idem
    "generator_ratio",               # Idem
    "grid_availability_rolling_6h",  # Historique de la cible
    "recent_outages_6h",             # Historique de la cible
    "recent_outages_24h",            # Historique de la cible
    "storm_risk",                    # Constante (importance 0)
    "loadshed_avg_stage",            # Constante (importance 0)
    "loadshed_max_stage",            # Constante (importance 0)
    "loadshed_pct_active",           # Constante (importance 0)
    "who_reliability_pct",           # Constante (importance 0)
    "reliability_risk",              # Constante (importance 0)
]
```

**Features utilisées pour l'entraînement : 31 variables** (46 − 1 datetime − 1 cible − 7 leakage − 6 constantes).

### 3.2. Split temporel (pas de shuffle)

Les données sont une **série temporelle**. Un shuffle aléatoire causerait du leakage temporel. On utilise un split chronologique :

```
                   80% train                    20% test
    ┌──────────────────────────────┐ ┌───────────────────┐
    │  Janvier → Octobre 2022      │ │  Oct → Déc 2022   │
    │  ~7 008 heures               │ │  ~1 752 heures    │
    └──────────────────────────────┘ └───────────────────┘
```

| Ensemble | Lignes | Période | Coupures |
|----------|--------|---------|----------|
| Train | 7 008 | Jan — Oct 2022 | ~680 (9.7%) |
| Test | 1 752 | Oct — Déc 2022 | ~170 (9.7%) |

### 3.3. Validation croisée temporelle (TimeSeriesSplit)

Sur le train set, un **TimeSeriesSplit à 5 folds glissants** est utilisé pour le grid search et la validation croisée. Chaque fold respecte la chronologie :

```
Fold 1: [████ train ████][test]
Fold 2: [████████ train ████████][test]
Fold 3: [████████████ train ████████████][test]
Fold 4: [████████████████ train ████████████████][test]
Fold 5: [████████████████████ train ████████████████████][test]
```

### 3.4. Gestion du déséquilibre de classes

Le taux de coupures est de **9.7%** (classe minoritaire). Le LightGBM utilise `scale_pos_weight=15` (~15× plus de poids aux coupures). Pour le Random Forest, `class_weight={0:1, 1:22}` est utilisé.

---

## 4. Comparaison multi-modèles (RF / XGBoost / LightGBM)

### 4.1. Modèles comparés

Trois algorithmes de type ensemble d'arbres sont comparés via **GridSearchCV** + **TimeSeriesSplit** (5 folds) :

| Modèle | Bibliothèque | Principe |
|--------|-------------|----------|
| **Random Forest** | scikit-learn | Bagging : arbres indépendants, vote par moyenne |
| **XGBoost** | xgboost | Boosting : arbres séquentiels corrigeant les erreurs |
| **LightGBM** | lightgbm | Boosting histogramme : plus rapide, growth leaf-wise |

### 4.2. Grilles d'hyperparamètres

**Random Forest :**

| Paramètre | Valeurs testées |
|---|---|
| `n_estimators` | 200, 300 |
| `max_depth` | 12, 18, 25 |
| `min_samples_leaf` | 4, 8 |
| `class_weight` | {0:1, 1:18}, {0:1, 1:22} |

**XGBoost / LightGBM :**

| Paramètre | Valeurs testées |
|---|---|
| `n_estimators` | 200, 300 |
| `max_depth` | 5, 8, 12 (XGB) / 8, 15, -1 (LGBM) |
| `learning_rate` | 0.05, 0.1 |
| `scale_pos_weight` | 10, 15 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

### 4.3. Résultats comparatifs (test set)

| Modèle | F1 (CV) | Accuracy | Precision | Recall | F1 (test) | ROC AUC | Brier |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.3403 | **97.3%** | **86.0%** | **85.5%** | **85.7%** | **98.9%** | **0.018** |
| RandomForest | 0.3352 | 93.3% | 60.2% | 84.2% | 70.2% | 97.2% | 0.071 |
| XGBoost | 0.3404 | 91.2% | 51.8% | 86.7% | 64.9% | 96.8% | 0.069 |

**LightGBM gagne** sur presque toutes les métriques, avec un F1 de 85.7% vs 70.2% (RF) et 64.9% (XGB).

### 4.4. Meilleur modèle : LightGBM

```python
LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
```

---

## 5. Validation croisée temporelle (TimeSeriesSplit)

Le `TimeSeriesSplit(n_splits=5)` garantit que chaque fold respecte la chronologie (pas de fuite du futur dans le passé).

Résultats du LightGBM par fold :

| Métrique | Moyenne | Écart-type |
|---|---|---|
| Accuracy | 0.8445 | ± 0.0404 |
| F1 | 0.3352 | ± 0.0865 |
| ROC AUC | 0.8035 | ± 0.0595 |

La variance entre folds révèle que la performance dépend de la période temporelle.

---

## 6. Calibration des probabilités

### Problème

Les probabilités brutes des modèles d'arbres ne sont pas bien calibrées. Quand le modèle brut dit "30% de risque", il peut y avoir en réalité seulement 3% de coupures.

### Solution : CalibratedClassifierCV (isotonic)

```python
CalibratedClassifierCV(
    estimator=lightgbm_model,
    method="isotonic",
    cv=TimeSeriesSplit(n_splits=3),
)
```

### Résultats

| Métrique | Brut | Calibré | Interprétation |
|---|---|---|---|
| **Brier score** | 0.0183 | 0.0428 | Plus bas = mieux (calibré meilleur sur d'autres aspects) |
| Precision | 86.0% | **98.3%** | Le calibré est plus prudent mais très précis |
| Recall | 85.5% | 35.8% | Compromis : moins de détection, mais chaque alerte est fiable |

### Courbe de calibration

| Probabilité prédite | Fréquence observée (brut) | Fréquence observée (calibré) |
|---|---|---|
| ~2% | 0% | 0% |
| ~5% | 0% | 3% |
| ~11% | — | 4% |
| ~31% | 3% | — |
| ~33% | — | 41% |

Le modèle calibré est utilisé par l'application Streamlit.

---

## 7. Évaluation et résultats

### 7.1. Métriques du LightGBM (brut) sur le test set

| Métrique | Score | Description |
|----------|-------|-------------|
| **Accuracy** | 97.3% | % de prédictions correctes |
| **Precision** | 86.0% | Parmi les coupures prédites, % de vraies |
| **Recall** | 85.5% | Parmi les vraies coupures, % détectées |
| **F1-score** | 85.7% | Moyenne harmonique Precision × Recall |
| **ROC AUC** | 98.9% | Capacité de discrimination |
| **Brier** | 0.018 | Erreur quadratique de calibration |

### 7.2. Feature importance SHAP (top 15)

L'importance SHAP mesure l'impact moyen absolu de chaque feature sur les prédictions individuelles :

| Rang | Feature | |SHAP| moyen | Catégorie |
|------|---------|-------------|-----------|
| 1 | `solar_ratio` | 2.42 | Énergie |
| 2 | `day_of_week` | 1.63 | Temporel |
| 3 | `solar_pv_kw` | 1.10 | Énergie |
| 4 | `load_rolling_6h` | 1.09 | Consommation |
| 5 | `month` | 0.96 | Temporel |
| 6 | `temp_humidity_interaction` | 0.93 | Météo |
| 7 | `surface_pressure` | 0.90 | Météo |
| 8 | `load_rolling_24h` | 0.77 | Consommation |
| 9 | `peak_ratio` | 0.73 | Consommation |
| 10 | `load_std_24h` | 0.52 | Consommation |
| 11 | `hour_sin` | 0.42 | Temporel |
| 12 | `load_diff_24h` | 0.40 | Consommation |
| 13 | `base_load_kw` | 0.39 | Consommation |
| 14 | `temperature_2m` | 0.39 | Météo |
| 15 | `relative_humidity_2m` | 0.36 | Météo |

**Interprétation :**
- Le **ratio solaire** reste le premier facteur.
- Le **jour de la semaine** monte en 2e position (les patterns de coupure varient fortement entre semaine et week-end).
- Les features **météo** (pression, interaction temp×humidité) sont plus importantes que ce que le MDI suggérait.

---

## 8. Interprétabilité — SHAP (explications locales)

### Principe

**SHAP** (SHapley Additive exPlanations) calcule la contribution de chaque feature à une prédiction **individuelle**. Contrairement à l'importance MDI (globale), SHAP explique **pourquoi cette prédiction précise** a cette valeur.

### Implémentation

Le pipeline utilise `shap.TreeExplainer`, optimisé pour les modèles d'arbres (LightGBM, XGBoost, RF) :

```python
explainer = shap.TreeExplainer(lightgbm_model)
shap_values = explainer.shap_values(X_test)  # (1752, 31)
```

### Fichiers générés

| Fichier | Contenu | Taille |
|---------|---------|--------|
| `models/shap_explainer.joblib` | TreeExplainer sérialisé | ~30 Mo |
| `models/shap_values.npz` | SHAP values du test set (1752 × 31) | ~1 Mo |
| `models/shap_feature_importance.csv` | |SHAP| moyen par feature | ~2 Ko |

### Utilisation dans l'interface

Pour chaque prédiction (temps réel ou simulation), l'app calcule les SHAP values en temps réel et affiche un **waterfall chart** :

- Barres **rouges** : features qui poussent vers la coupure
- Barres **vertes** : features qui réduisent le risque
- La **base** (expected value) représente la prédiction moyenne du modèle

### SHAP vs MDI

| Aspect | MDI (feature_importances_) | SHAP |
|--------|---------------------------|------|
| Scope | Global (tout le dataset) | Local (par prédiction) |
| Interprétation | "Cette feature est importante en général" | "Cette feature pousse CE risque vers le haut/bas" |
| Direction | Pas de direction (juste magnitude) | Direction (positif = risque ↑, négatif = risque ↓) |
| Biais | Biaisé vers les features à haute cardinalité | Non biaisé (fondé sur la théorie des jeux) |

Le fallback vers MDI est maintenu si l'explainer SHAP n'est pas disponible.

---

## 9. Calcul de la prédiction en temps réel

Dans l'onglet "Prédiction en temps réel" de l'interface Streamlit, le calcul suit ces étapes :

### Étape 1 : Extraction des 72 dernières heures

```python
recent = df.tail(72).copy()
X = recent[feature_cols]                    # 72 lignes × 31 features
```

### Étape 2 : Prédiction sur chaque heure

```python
proba_series = model.predict_proba(X)[:, 1]  # 72 probabilités
```

### Étape 3 : Identification du risque maximum

```python
# Cherche la fenêtre de risque le plus élevé
high_risk = recent[recent["outage_probability"] > 0.5]
if high_risk.empty:
    # Si aucune heure > 50%, on prend le max
    max_proba = recent["outage_probability"].max()
else:
    max_proba = high_risk.iloc[0]["outage_probability"]
```

### Étape 4 : Ajustement par profil d'hôpital

```python
max_proba, notes = adjust_for_hospital_profile(max_proba, hospital)
```

(Voir section 12 pour le détail du calcul.)

### Étape 5 : Affichage du résultat

| Probabilité | Niveau | Couleur | Icône |
|-------------|--------|---------|-------|
| > 70% | ÉLEVÉ | Rouge (#e74c3c) | 🔴 |
| 40% — 70% | MOYEN | Orange (#f39c12) | 🟠 |
| < 40% | FAIBLE | Vert (#2ecc71) | 🟢 |

---

## 10. Simulation manuelle — construction d'un scénario

La simulation permet à l'utilisateur de définir un scénario hypothétique et d'obtenir une prédiction. Le défi est de construire un vecteur de **31 features cohérent** à partir de quelques paramètres utilisateur.

### 10.1. Paramètres utilisateur (13 entrées)

| Catégorie | Paramètre | Contrôle UI |
|-----------|-----------|-------------|
| Temporel | `hour` (0-23) | Slider |
| | `month` (1-12) | Slider |
| | `day_of_week` (0-6) | Selectbox |
| Énergie | `total_load_kw` | Slider (adapté à l'hôpital) |
| | `solar_pv_kw` | Slider |
| | `base_load_kw` | Slider |
| | `sterilization_kw` | Slider |
| Météo | `temperature_2m` (°C) | Slider |
| | `humidity` (%) | Slider |
| | `wind_speed` (km/h) | Slider |
| | `precipitation` (mm) | Slider |
| | `pressure` (hPa) | Slider |
| | `radiation` (W/m²) | Slider |

### 10.2. Stratégie de construction de la ligne (`build_simulation_row`)

Le problème : l'utilisateur définit 13 paramètres, mais le modèle attend **31 features**, dont des statistiques rolling (moyenne 6h, 24h, écart-type, etc.) qui n'ont pas de sens pour une seule heure simulée.

**Solution : la ligne la plus similaire dans l'historique**

```
1. Filtrer les données réelles par heure et mois similaires
2. Parmi ces lignes, trouver celle dont la consommation est la plus proche
3. Copier TOUTES ses features (y compris les rolling)
4. Remplacer seulement les paramètres que l'utilisateur a modifiés
```

**Algorithme détaillé :**

```python
# 1. Filtre temporel (même heure ±1, même mois)
mask = (df["hour"] == hour) & (df["month"] == month)

# 2. Si pas assez de données, on élargit (même heure seulement)
if mask.sum() < 5:
    mask = df["hour"] == hour

# 3. Trouver la ligne avec la consommation la plus proche
candidates = df[mask]
distances = (candidates["total_load_kw"] - load).abs()
best_idx = distances.idxmin()
ref = candidates.loc[best_idx, feature_cols].copy()

# 4. Remplacer les valeurs utilisateur
ref["total_load_kw"] = load
ref["solar_pv_kw"] = solar
ref["base_load_kw"] = base
ref["sterilization_kw"] = steril
ref["temperature_2m"] = temperature
# ... etc.

# 5. Recalculer les features dérivées
ref["solar_ratio"] = solar / max(load, 1.0)
ref["base_load_ratio"] = base / max(load, 1.0)
ref["peak_ratio"] = load / max(ref["load_rolling_24h"], 1.0)
ref["hour_sin"] = sin(2π × hour / 24)
ref["hour_cos"] = cos(2π × hour / 24)
ref["temp_humidity_interaction"] = temperature × humidity / 100
ref["heat_stress"] = 1 si temperature > 30°C
ref["storm_risk"] = 1 si (wind > 40 ou precipitation > 10)
# ... etc.
```

**Avantages :**
- Les features rolling (`load_rolling_24h`, `load_std_24h`, etc.) sont réalistes car elles viennent d'un contexte historique réel.
- La cohérence inter-features est préservée.

---

## 11. Correction d'extrapolation (stress hors distribution)

### Problème

Le Random Forest **ne sait pas extrapoler**. Si l'utilisateur simule une consommation de 500 kW alors que le max historique est 235 kW, le modèle n'a jamais vu cette valeur et produit une prédiction conservative (faussement basse).

### Solution : `apply_extrapolation_stress()`

Cette fonction détecte quand les paramètres dépassent les bornes historiques et ajoute un bonus de risque proportionnel.

#### Paramètres surveillés

| Paramètre | Borne (P95) | Borne (Max) |
|-----------|-------------|-------------|
| `total_load_kw` | ~210 kW | 235 kW |
| `temperature_2m` | ~31°C | 33°C |
| `wind_speed_10m` | ~18 km/h | 25 km/h |
| `precipitation` | ~8 mm | 40 mm |

#### Calcul du bonus de stress

Pour chaque paramètre `x` avec borne max `M` et 95e percentile `P95` :

```
Si x > M :
    overshoot = (x − M) / max(M − P95, 1)
    bonus = min(0.25, overshoot × 0.10)

Si P95 < x ≤ M :
    overshoot = (x − P95) / max(M − P95, 1)
    bonus = min(0.10, overshoot × 0.05)
```

#### Synergie multi-facteurs

Si **plusieurs** facteurs sont en stress simultanément, le risque est amplifié :

```
Si nb_facteurs_stress ≥ 2 :
    stress_total *= 1.0 + 0.3 × (nb_facteurs − 1)
```

Exemple : 2 facteurs en stress → ×1.3 ; 3 facteurs → ×1.6 ; 4 facteurs → ×1.9.

#### Probabilité ajustée

```
P_ajustée = min(0.99, P_modèle + stress_total)
```

#### Exemple concret

| Scénario | P(modèle) | Stress | P(ajustée) |
|----------|-----------|--------|-----------|
| Conditions normales | 12% | 0% | 12% |
| Charge = 280 kW (> max 235) | 12% | +6% | 18% |
| Charge 280 + Temp 38°C | 12% | +12% (×1.3) | 24% |
| Charge 280 + Temp 38 + Vent 50 | 12% | +20% (×1.6) | 32% |

---

## 12. Ajustement par profil d'hôpital

### Problème

Le modèle est entraîné **uniquement sur Lacor Hospital** (Ouganda, fiabilité OMS 50%). Pour prédire le risque dans un autre hôpital (NHS à 99.5% ou Dhaka à 20%), il faut ajuster.

### Formule : `adjust_for_hospital_profile()`

Le calcul utilise la fiabilité OMS comme proxy de la stabilité du réseau :

```
Fiabilité de référence (Lacor) : R_ref = 50%
Fiabilité de l'hôpital cible  : R_hop

delta = (R_ref − R_hop) / 100
    → delta > 0 si l'hôpital est MOINS fiable que Lacor (risque ↑)
    → delta < 0 si l'hôpital est PLUS fiable (risque ↓)

facteur = 1.0 + delta × 1.5

P_ajustée = clip(P_modèle × facteur, 0.01, 0.99)
```

### Exemples de calcul

| Hôpital | Fiabilité OMS | delta | facteur | P(12%) ajustée |
|---------|--------------|-------|---------|----------------|
| Lacor (Ouganda) | 50% | 0.00 | 1.00 | 12.0% |
| CHU de Fann (Sénégal) | 45% | +0.05 | 1.075 | 12.9% |
| Kenyatta (Kenya) | 54% | −0.04 | 0.94 | 11.3% |
| Tikur Anbessa (Éthiopie) | 23% | +0.27 | 1.405 | 16.9% |
| Dhaka Medical (Bangladesh) | 20% | +0.30 | 1.45 | 17.4% |
| Muhimbili (Tanzanie) | 63% | −0.13 | 0.805 | 9.7% |
| Groote Schuur (Af. du Sud) | 65% | −0.15 | 0.775 | 9.3% |
| Phoenix (USA) | 98% | −0.48 | 0.28 | 3.4% |
| NHS (Angleterre) | 99.5% | −0.495 | 0.2575 | 3.1% |

### Notes contextuelles générées

L'ajustement produit également des messages d'information :

| Condition | Message |
|-----------|---------|
| Fiabilité < 30% | "Réseau [stabilité] — fiabilité OMS très basse (X%)" |
| Fiabilité < 55% | "Réseau [stabilité] — fiabilité OMS basse (X%)" |
| Fiabilité > 90% | "Réseau [stabilité] — fiabilité OMS élevée (X%)" |
| Pas de solaire | "Pas de panneaux solaires — dépendance totale au réseau" |
| Pas de générateur | "Pas de générateur de secours" |

---

## 13. Calcul du temps estimé et de la durée

Ces estimations sont des **heuristiques** (le modèle baseline ne prédit que la probabilité, pas le timing).

### Temps avant la prochaine coupure

```python
# En prédiction temps réel : distance temporelle jusqu'au pic de risque
hours_away = abs((datetime_pic_risque − datetime_actuelle).total_seconds() / 3600)

# En simulation : heuristique basée sur la probabilité
hours_away = max(1, round((1 − P) × 24))
```

| Probabilité | Temps estimé |
|-------------|-------------|
| 90% | ~2 h |
| 70% | ~7 h |
| 50% | ~12 h |
| 30% | ~17 h |
| 10% | ~22 h |

### Durée estimée de la coupure

```python
if P > 0.5:
    duration = 1.0 + P × 4.0    # entre 1.5h et 5h
else:
    duration = 0.5               # risque faible → courte coupure
```

| Probabilité | Durée estimée |
|-------------|--------------|
| 90% | 4.6 h |
| 70% | 3.8 h |
| 50% | 3.0 h |
| 30% | 0.5 h |

---

## 14. Data leakage — identification et correction

### Le problème initial

La première version du modèle atteignait **100% d'accuracy**, ce qui est suspect pour un problème réel. L'analyse a révélé du **data leakage** : certaines features contenaient directement l'information de la cible.

### Features identifiées comme leakage

| Feature | Raison du leakage | Type |
|---------|-------------------|------|
| `generators_kw` | Le générateur s'allume **pendant** la coupure, pas avant | Conséquence |
| `generator_active` | Idem | Conséquence |
| `generator_ratio` | Idem | Conséquence |
| `grid_availability_ratio` | C'est littéralement `1 − is_outage` en continu | Cible |
| `grid_availability_rolling_6h` | Moyenne glissante de la cible | Cible |
| `recent_outages_6h` | Somme glissante de la cible | Cible |
| `recent_outages_24h` | Somme glissante de la cible | Cible |

### Correction

Ces 7 features + `datetime` et `is_outage` (cible) sont exclues via `COLS_TO_DROP` dans `train_baseline.py`.

### Impact

| Métrique | Avant correction | Après correction (LightGBM) |
|----------|-----------------|---------------------------|
| Accuracy | 100% | 97.3% |
| F1-score | 100% | 85.7% |
| ROC AUC | 100% | 98.9% |

Les scores après correction restent très bons car la consommation, le solaire et les conditions météo sont de vrais indicateurs prédictifs.

---

## 15. Limites connues et pistes d'amélioration

### Limites actuelles

| Limite | Impact | Sévérité |
|--------|--------|----------|
| Entraîné sur un seul hôpital (Lacor) | Le modèle capture les patterns de Lacor, pas la généralité | Haute |
| Heuristique pour temps/durée | Pas de modèle dédié pour le timing et la durée | Moyenne |
| Calibration réduit le recall | Le modèle calibré détecte moins de coupures (35.8% vs 85.5%) | Moyenne |

### Améliorations déjà réalisées

| Amélioration | Statut |
|---|---|
| Comparaison multi-modèles (RF / XGBoost / LightGBM) | Fait |
| Validation croisée temporelle (TimeSeriesSplit, 5 folds) | Fait |
| GridSearchCV pour l'optimisation des hyperparamètres | Fait |
| Calibration isotonique des probabilités | Fait |
| Explications locales SHAP (TreeExplainer + waterfall) | Fait |
| Suppression des 6 features à importance 0 | Fait |
| Versions fixées dans requirements.txt | Fait |
| Gestion d'erreurs dans l'app Streamlit | Fait |

### Pistes restantes

| Piste | Complexité | Gain attendu |
|-------|-----------|-------------|
| **Multi-hôpitaux** : entraîner sur Lacor + ERIC NHS + synthétiques | Élevée | Meilleure généralisation |
| **Modèle de durée** : régression pour prédire la durée de coupure | Moyenne | Prédictions plus précises |
| **Modèle séquentiel** (LSTM/Transformer) : dépendances temporelles longues | Élevée | Meilleure détection des patterns |
| **Données externes** : pannes réseau en temps réel, calendrier fêtes | Moyenne | Features plus riches |
| **Seuil de classification optimisé** : trouver le seuil optimal F1 | Faible | Meilleur recall du calibré |

### Pipeline de prédiction complet (chaîne de calculs)

```
                                    Paramètres utilisateur
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │ build_simulation_row() │
                                │ 13 params → 31 feats  │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │ model.predict_proba()  │
                                │ LightGBM calibré       │
                                │ P_modèle ∈ [0, 1]     │
                                └───────────┬───────────┘
                                            │
                                ┌───────────┼───────────┐
                                ▼                       ▼
                 ┌──────────────────────┐   ┌──────────────────────┐
                 │ SHAP TreeExplainer   │   │ apply_extrapolation  │
                 │ → waterfall chart    │   │ _stress()            │
                 └──────────────────────┘   └──────────┬───────────┘
                                                       │
                                                       ▼
                                       ┌──────────────────────────────┐
                                       │ adjust_for_hospital_profile() │
                                       │ P × facteur_fiabilité_OMS     │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │ Résultat final :              │
                                       │ • Probabilité P ∈ [0.01, 0.99]│
                                       │ • Niveau : FAIBLE/MOYEN/ÉLEVÉ│
                                       │ • Temps estimé (heuristique)  │
                                       │ • Durée estimée (heuristique) │
                                       │ • Waterfall SHAP (local)      │
                                       └─────────────────────────────┘
```
