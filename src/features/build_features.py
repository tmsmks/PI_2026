"""
Feature Engineering pour la prédiction de coupures (données réelles Lacor).

Variables créées :
  ── Temporelles ──
  - hour, day_of_week, month              (déjà dans le dataset)
  - is_weekend                            (booléen)
  - hour_sin, hour_cos                    (encodage cyclique)
  - month_sin, month_cos                  (encodage cyclique)

  ── Consommation (rolling) ──
  - load_rolling_6h                       (moyenne glissante 6h)
  - load_rolling_24h                      (moyenne glissante 24h)
  - load_std_24h                          (écart-type glissant 24h)
  - load_diff_1h                          (variation heure par heure)
  - load_diff_24h                         (variation jour par jour)
  - load_pct_change_1h                    (variation relative %)
  - peak_ratio                            (ratio charge / moyenne 24h)

  ── Sources d'énergie ──
  - solar_ratio                           (part du solaire dans la charge)
  - generator_active                      (générateur en marche ? 0/1)
  - generator_ratio                       (part du générateur)
  - base_load_ratio                       (ratio base / total)
  - grid_availability_rolling_6h          (stabilité réseau 6h glissant)
  - recent_outages_6h                     (nombre de coupures dans les 6h)
  - recent_outages_24h                    (nombre de coupures dans les 24h)

  ── Météo ──
  - temp_humidity_interaction
  - wind_precipitation_interaction
  - solar_available                       (rayonnement > 50 W/m²)
  - heat_stress                           (température > 30°C en Ouganda)

Features retirées (importance 0, constantes sur la série mono-hôpital) :
  - storm_risk, who_reliability_pct, reliability_risk
  - loadshed_avg_stage, loadshed_max_stage, loadshed_pct_active
"""

import logging

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR, FEATURES_DIR
from src.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    col = "total_load_kw"

    df["load_rolling_6h"] = df[col].rolling(6, min_periods=1).mean()
    df["load_rolling_24h"] = df[col].rolling(24, min_periods=1).mean()
    df["load_std_24h"] = df[col].rolling(24, min_periods=1).std().fillna(0)

    df["load_diff_1h"] = df[col].diff().fillna(0)
    df["load_diff_24h"] = df[col].diff(24).fillna(0)
    df["load_pct_change_1h"] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    df["peak_ratio"] = (df[col] / df["load_rolling_24h"]).fillna(1).replace([np.inf, -np.inf], 1)

    return df


def add_energy_source_features(df: pd.DataFrame) -> pd.DataFrame:
    total = df["total_load_kw"].replace(0, np.nan)

    if "solar_pv_kw" in df.columns:
        df["solar_ratio"] = (df["solar_pv_kw"] / total).fillna(0).clip(0, 1)

    if "generators_kw" in df.columns:
        df["generator_active"] = (df["generators_kw"] > 1.0).astype(int)
        df["generator_ratio"] = (df["generators_kw"] / total).fillna(0).clip(0, 1)

    if "base_load_kw" in df.columns:
        df["base_load_ratio"] = (df["base_load_kw"] / total).fillna(0).clip(0, 1)

    if "grid_availability_ratio" in df.columns:
        df["grid_availability_rolling_6h"] = (
            df["grid_availability_ratio"].rolling(6, min_periods=1).mean()
        )

    # Historique récent de coupures
    if "is_outage" in df.columns:
        df["recent_outages_6h"] = df["is_outage"].rolling(6, min_periods=1).sum()
        df["recent_outages_24h"] = df["is_outage"].rolling(24, min_periods=1).sum()

    return df


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature_2m" not in df.columns:
        logger.info("Pas de colonnes météo — features météo ignorées.")
        return df

    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["wind_precipitation_interaction"] = df["wind_speed_10m"] * df["precipitation"]
    df["solar_available"] = (df["shortwave_radiation"] > 50).astype(int)
    df["heat_stress"] = (df["temperature_2m"] > 30).astype(int)
    df["storm_risk"] = ((df["wind_speed_10m"] > 40) | (df["precipitation"] > 10)).astype(int)

    return df


def add_reliability_risk(df: pd.DataFrame) -> pd.DataFrame:
    if "who_reliability_pct" in df.columns:
        df["reliability_risk"] = 1 - df["who_reliability_pct"] / 100
    return df


def run() -> None:
    df = load_csv(PROCESSED_DIR / "hospital_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    df = add_temporal_features(df)
    df = add_load_features(df)
    df = add_energy_source_features(df)
    df = add_meteo_features(df)
    df = add_reliability_risk(df)

    # Remplacer les NaN restants
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    save_csv(df, FEATURES_DIR / "features_dataset.csv")

    feature_cols = [c for c in df.columns if c not in ("datetime", "is_outage")]
    logger.info("Feature engineering terminé : %d features", len(feature_cols))
    logger.info("Colonnes : %s", list(df.columns))


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
