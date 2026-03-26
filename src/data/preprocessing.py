"""
Preprocessing : nettoyage, rééchantillonnage, fusion des datasets.

Stratégie :
  1. Dataset principal = Lacor Hospital (15 min, 2022)
     → rééchantillonné à l'heure pour aligner avec la météo
  2. Météo historique 2022 (Open-Meteo Archive, horaire) → jointure temporelle
  3. Features macro OMS (statique par pays) → jointure par pays
  4. Données Eskom (contexte réseau) → features statistiques agrégées

Le résultat est un DataFrame horaire unique prêt pour le feature engineering.
"""

import logging

import numpy as np
import pandas as pd

from src.utils.config import RAW_DIR, PROCESSED_DIR
from src.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def resample_lacor_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rééchantillonne les données 15 min de Lacor en données horaires.
    Pour les variables continues → moyenne horaire.
    Pour is_outage → 1 si au moins une coupure dans l'heure.
    Pour grid_available → fraction de disponibilité dans l'heure.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    agg_rules = {
        "solar_pv_kw": "mean",
        "total_load_kw": "mean",
        "generators_kw": "mean",
        "sterilization_kw": "mean",
        "base_load_kw": "mean",
        "grid_available": "mean",
        "is_outage": "max",
    }

    hourly = df.resample("1h").agg(agg_rules)
    hourly = hourly.reset_index()

    # grid_available devient la fraction de l'heure avec réseau (0.0 à 1.0)
    hourly = hourly.rename(columns={"grid_available": "grid_availability_ratio"})

    logger.info(
        "Rééchantillonnage Lacor : %d → %d lignes (15 min → horaire)",
        len(df), len(hourly),
    )
    return hourly


def merge_with_meteo(consumption: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """Jointure temporelle consommation + météo sur l'heure la plus proche."""
    meteo = meteo.copy()
    meteo["datetime"] = pd.to_datetime(meteo["datetime"])
    meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
    meteo_subset = meteo[["datetime"] + meteo_cols].copy()

    merged = pd.merge_asof(
        consumption.sort_values("datetime"),
        meteo_subset.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    n_missing = merged[meteo_cols].isnull().sum().sum()
    if n_missing > 0:
        logger.warning("%d cellules météo sans correspondance → interpolation", n_missing)
        merged[meteo_cols] = merged[meteo_cols].interpolate(method="linear").ffill().bfill()

    logger.info("Fusion météo : %d colonnes ajoutées", len(meteo_cols))
    return merged


def add_who_context(df: pd.DataFrame, country_code: str = "UGA") -> pd.DataFrame:
    """Enrichit avec la fiabilité OMS du pays."""
    try:
        who = load_csv(RAW_DIR / "who_reliability.csv")
    except FileNotFoundError:
        logger.warning("Fichier OMS non trouvé — valeur par défaut.")
        df["who_reliability_pct"] = 50.0
        return df

    country_data = who[
        (who["country_code"] == country_code) & (who["area_type"] == "totl")
    ]
    if country_data.empty:
        logger.warning("Pas de données OMS pour %s — valeur par défaut", country_code)
        df["who_reliability_pct"] = 50.0
    else:
        latest = country_data.sort_values("year", ascending=False).iloc[0]
        df["who_reliability_pct"] = latest["reliability_pct"]
        logger.info("OMS %s : fiabilité = %.0f%%", country_code, latest["reliability_pct"])

    return df


def add_loadshedding_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit avec des statistiques agrégées de load shedding.
    Le dataset Eskom est sud-africain mais fournit un pattern
    représentatif de l'instabilité des réseaux africains.
    """
    try:
        loadshed = load_csv(RAW_DIR / "loadshedding_history_clean.csv")
    except FileNotFoundError:
        logger.warning("Historique load shedding non trouvé — ignoré.")
        return df

    loadshed["created_at"] = pd.to_datetime(loadshed["created_at"])

    avg_stage = loadshed["stage"].mean()
    max_stage = loadshed["stage"].max()
    pct_active = (loadshed["stage"] > 0).mean()

    df["loadshed_avg_stage"] = round(avg_stage, 2)
    df["loadshed_max_stage"] = max_stage
    df["loadshed_pct_active"] = round(pct_active, 4)

    logger.info(
        "Load shedding context : avg_stage=%.2f, max=%d, pct_active=%.1f%%",
        avg_stage, max_stage, 100 * pct_active,
    )
    return df


def run() -> None:
    # 1. Charger Lacor et rééchantillonner
    lacor = load_csv(RAW_DIR / "lacor_clean.csv")
    hourly = resample_lacor_hourly(lacor)

    # 2. Fusionner avec la météo
    try:
        meteo = load_csv(RAW_DIR / "meteo_lacor_uganda.csv")
        hourly = merge_with_meteo(hourly, meteo)
    except FileNotFoundError:
        logger.warning("Pas de données météo Lacor — on continue sans.")

    # 3. Ajouter le contexte OMS
    hourly = add_who_context(hourly, country_code="UGA")

    # 4. Ajouter le contexte load shedding
    hourly = add_loadshedding_context(hourly)

    # 5. Ajouter des colonnes temporelles de base
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["day_of_week"] = hourly["datetime"].dt.dayofweek
    hourly["month"] = hourly["datetime"].dt.month

    # 6. Traiter les valeurs manquantes restantes
    numeric_cols = hourly.select_dtypes(include=[np.number]).columns
    n_missing = hourly[numeric_cols].isnull().sum().sum()
    if n_missing > 0:
        logger.warning("%d valeurs manquantes restantes → interpolation", n_missing)
        hourly[numeric_cols] = hourly[numeric_cols].interpolate(method="linear").ffill().bfill()

    save_csv(hourly, PROCESSED_DIR / "hospital_merged.csv")
    logger.info(
        "Preprocessing terminé : %d lignes, %d colonnes, coupures=%.1f%%",
        len(hourly), len(hourly.columns), 100 * hourly["is_outage"].mean(),
    )


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
