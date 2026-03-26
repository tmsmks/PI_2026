# Cahier des charges - Projet de prediction de coupures electriques en hopitaux

## 1) Contexte et finalite

Ce projet vise a concevoir une solution data/IA capable de predire les coupures electriques dans un contexte hospitalier, afin d'anticiper les risques operationnels et de renforcer la continuite des soins.

Le systeme s'appuie sur des donnees reelles (Lacor Hospital), des donnees de benchmark (Phoenix, NHS ERIC), des donnees meteo, et des donnees de contexte reseau/pays (OMS, Eskom).

## 2) Objectifs du projet

### 2.1 Objectif principal

Construire un pipeline complet de prediction de coupure electrique et une interface applicative permettant une interpretation metier des resultats.

### 2.2 Objectifs fonctionnels

- Predire la probabilite de coupure (`is_outage`).
- Estimer le temps avant prochaine coupure (indicateur derive).
- Estimer la duree probable de coupure (indicateur derive).
- Fournir des explications interpretable du resultat (SHAP/importance des variables).
- Permettre une utilisation via une interface Streamlit en mode analyse et simulation.

### 2.3 Objectifs techniques

- Industrialiser un pipeline data reproductible (ingestion -> preprocessing -> features -> entrainement -> evaluation).
- Comparer plusieurs modeles ML (Random Forest, XGBoost, LightGBM).
- Eviter les biais de fuite de donnees (data leakage) via split temporel et selection de features.
- Produire une base documentaire claire et exploitable.

## 3) Perimetre

### 3.1 Inclus dans le perimetre

- Ingestion multi-sources:
  - Lacor (donnees de charge et disponibilite reseau),
  - Phoenix (benchmark),
  - OMS (fiabilite electrique par pays),
  - Open-Meteo (variables meteorologiques),
  - Eskom/EskomSePush (contexte de stabilite reseau),
  - NHS ERIC (profils hospitaliers).
- Nettoyage et fusion des donnees temporelles.
- Feature engineering (variables temporelles, consommation, meteo, contexte).
- Entrainement et comparaison de modeles.
- Calibration des probabilites.
- Generation d'explications locales/globales (SHAP).
- Exposition des resultats dans l'application Streamlit.

### 3.2 Hors perimetre (phase actuelle)

- Deploiement cloud haute disponibilite.
- MLOps complet (CI/CD modele, monitoring production automatise).
- Collecte temps reel connectee a des IoT/SCADA hospitaliers.
- Validation clinique/reglementaire formelle.

## 4) Parties prenantes

- **Porteur du projet**: etudiant/chef de projet data.
- **Utilisateur cible**: responsable technique hopital, energie/maintenance, decideur operationnel.
- **Encadrant/lecteur**: evaluateur academique ou jury.

## 5) Exigences fonctionnelles

### 5.1 Pipeline de donnees

- Le pipeline doit pouvoir etre execute de bout en bout via `run_pipeline.py`.
- Chaque etape doit produire des artefacts lisibles et versionnes (CSV, joblib, JSON).
- Les erreurs de chargement doivent etre explicites (messages clairs).

### 5.2 Modelisation

- Le systeme doit entrainer plusieurs modeles et selectionner le meilleur selon des metriques definies.
- Le split entrainement/test doit respecter la chronologie.
- Les probabilites doivent etre calibrables.

### 5.3 Application

- L'utilisateur doit pouvoir:
  - choisir un profil d'hopital,
  - consulter une probabilite de coupure,
  - visualiser les facteurs explicatifs,
  - simuler un scenario manuel.

## 6) Exigences non fonctionnelles

- **Performance**: inferer une prediction en quelques secondes sur poste local.
- **Fiabilite**: execution reproductible avec dependances figees (`requirements.txt`).
- **Maintenabilite**: code modulaire par couche (`src/data`, `src/features`, `src/models`).
- **Lisibilite**: documentation fonctionnelle et technique a jour.
- **Traçabilite**: artefacts d'entrainement sauvegardes (`models/`, `data/features/`, `training_summary.json`).

## 7) Donnees et architecture cible

### 7.1 Flux de traitement

1. Ingestion des sources heterogenes (Excel/CSV/API).
2. Harmonisation temporelle et nettoyage.
3. Fusion des jeux de donnees.
4. Construction de features derivees.
5. Entrainement/validation/calibration.
6. Explicabilite des predictions.
7. Consommation des resultats dans l'interface.

### 7.2 Artefacts attendus

- Dataset fusionne preprocess (`data/processed/...`).
- Dataset features (`data/features/features_dataset.csv`).
- Modeles sauvegardes (`models/baseline_rf.joblib`, `models/calibrated_rf.joblib`).
- Explicabilite (`models/shap_explainer.joblib`, `models/shap_values.npz`).
- Fichiers de synthese (`models/model_comparison.csv`, `models/training_summary.json`).

## 8) Livrables

- **L1 - Pipeline data/ML fonctionnel** (scripts ingestion, preprocessing, features, training).
- **L2 - Modele baseline compare et evalue** (RF/XGB/LGBM + metriques).
- **L3 - Module de prediction** (`src/models/predict.py`).
- **L4 - Interface Streamlit operationnelle** (`app.py`).
- **L5 - Documentation complete** (donnees, APIs, modele, usage).
- **L6 - Cahier des charges final** (ce document).

## 9) Planning previsionnel (mi-mars -> mi-juin)

Periode cible: **du 15 mars au 15 juin** (13 semaines environ).

### Phase 1 - Cadrage et collecte (S1 a S3: 15 mars -> 4 avril)

- Validation du besoin metier et des objectifs.
- Inventaire et qualification des sources de donnees.
- Mise en place de la structure projet et environnement Python.
- Premieres briques d'ingestion API/CSV/Excel.

**Jalon J1 (fin S3)**: sources principales accessibles, scripts d'ingestion initialises.

### Phase 2 - Preparation data et feature engineering (S4 a S6: 5 avril -> 25 avril)

- Nettoyage des donnees, traitement des valeurs manquantes/incoherentes.
- Synchronisation temporelle et fusion des sources.
- Construction des features (temporelles, charge, meteo, reseau).
- Verification de la qualite et de la stabilite des donnees.

**Jalon J2 (fin S6)**: dataset fusionne exploitable et jeu de features finalise.

### Phase 3 - Modelisation et evaluation (S7 a S9: 26 avril -> 16 mai)

- Entrainement baseline et comparaison multi-modeles.
- Validation croisee temporelle et mesure des performances.
- Detection/correction des risques de fuite de donnees.
- Calibration des probabilites et analyse de robustesse.

**Jalon J3 (fin S9)**: meilleur modele retenu, metriques consolidees.

### Phase 4 - Explicabilite et application (S10 a S11: 17 mai -> 30 mai)

- Integration SHAP (global/local).
- Mise en oeuvre des fonctions de prediction inferentielles.
- Integration dans l'interface Streamlit (analyse + simulation).

**Jalon J4 (fin S11)**: application demonstrable de bout en bout.

### Phase 5 - Stabilisation et livraison (S12 a S13: 31 mai -> 15 juin)

- Tests de non-regression et verification fonctionnelle.
- Finalisation de la documentation technique/fonctionnelle.
- Preparation de la soutenance/demo (scenarios, captures, narration).

**Jalon J5 (15 juin)**: livraison complete du projet.

## 10) Criteres d'acceptation

Le projet est considere comme valide si:

- Le pipeline s'execute sans erreur sur l'environnement cible.
- Les donnees sont fusionnees correctement et les artefacts sont produits.
- Un modele de prediction est entraine, sauvegarde, et exploitable.
- L'application fournit une probabilite de coupure et des explications.
- Les metriques minimales sont documentees (precision, recall, F1, ROC AUC, calibration).
- Les limites et hypotheses sont explicitement decrites.

## 11) Risques et plan de mitigation

- **Risque qualite donnees** (donnees manquantes/bruitees):
  - Mitigation: controles qualite, regles de nettoyage explicites, fallbacks.
- **Risque d'incoherence temporelle**:
  - Mitigation: normalisation timezone/frequence, verification des index datetime.
- **Risque de surapprentissage**:
  - Mitigation: split chronologique, TimeSeriesSplit, suivi des metriques train/test.
- **Risque de fuite de donnees**:
  - Mitigation: liste stricte des colonnes exclues, revue des features sensibles.
- **Risque de planning**:
  - Mitigation: jalons hebdomadaires, priorisation livrables critiques.

## 12) Hypotheses et contraintes

- Travail realise en Python avec dependances definies.
- Execution majoritairement locale (pas d'obligation cloud).
- Le dataset principal de reference reste Lacor 2022.
- Les predicteurs de "temps avant coupure" et "duree" sont derives dans cette phase et peuvent etre raffines dans une phase 2.

## 13) Gouvernance et suivi

- Point d'avancement hebdomadaire (etat, blocages, actions).
- Revue de jalon a chaque fin de phase.
- Validation finale sur demonstration complete: pipeline + modele + app + documentation.

---

Version: 1.0  
Periode de reference: mi-mars -> mi-juin  
Statut: document de cadrage projet
