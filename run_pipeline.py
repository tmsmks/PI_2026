"""
Script principal : exécute le pipeline complet de bout en bout.

Étapes :
  1. Ingestion des données réelles (Lacor, Phoenix, Eskom, OMS, météo)
  2. Preprocessing (rééchantillonnage, nettoyage, fusion)
  3. Feature engineering
  4. Entraînement du modèle baseline (Random Forest)
"""

import logging
import sys

from src.utils.io import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("  PIPELINE — Prédiction de coupures d'électricité")
    logger.info("  Données réelles : Lacor Hospital (Ouganda, 2022)")
    logger.info("=" * 60)

    # ── Étape 1 : Ingestion ─────────────────────────────────────────
    logger.info("\n▶ ÉTAPE 1 : Ingestion des données")

    logger.info("  → Chargement des datasets de consommation…")
    from src.data.ingest_consumption import run as ingest_consumption
    ingest_consumption()

    logger.info("  → Chargement des données de pannes (Eskom)…")
    from src.data.ingest_outages import run as ingest_outages
    ingest_outages()

    logger.info("  → Récupération des données OMS…")
    try:
        from src.data.ingest_who import run as ingest_who
        ingest_who()
    except Exception as e:
        logger.warning("  ⚠ Ingestion OMS échouée : %s", e)

    logger.info("  → Récupération de la météo historique 2022…")
    try:
        from src.data.ingest_meteo import run as ingest_meteo
        ingest_meteo()
    except Exception as e:
        logger.warning("  ⚠ Ingestion météo échouée : %s", e)

    logger.info("  → Ingestion données ERIC NHS…")
    try:
        from src.data.ingest_eric import run as ingest_eric
        ingest_eric()
    except Exception as e:
        logger.warning("  ⚠ Ingestion ERIC échouée : %s", e)

    # ── Étape 2 : Preprocessing ─────────────────────────────────────
    logger.info("\n▶ ÉTAPE 2 : Preprocessing (rééchantillonnage + fusion)")
    from src.data.preprocessing import run as preprocess
    preprocess()

    # ── Étape 3 : Feature engineering ───────────────────────────────
    logger.info("\n▶ ÉTAPE 3 : Feature engineering")
    from src.features.build_features import run as build_features
    build_features()

    # ── Étape 4 : Entraînement baseline ─────────────────────────────
    logger.info("\n▶ ÉTAPE 4 : Entraînement du modèle baseline")
    from src.models.train_baseline import run as train_baseline
    train_baseline()

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
