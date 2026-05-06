"""
Ingestion des données de qualité de l'air via Open-Meteo Air Quality API.

Pourquoi c'est utile pour la prédiction de coupure :
    - Pollution élevée (PM2.5, PM10) ⇒ usage accru de la climatisation et
      de la ventilation hospitalière (surcharge réseau interne).
    - Tempêtes de poussière (Harmattan, haboob) ⇒ dépôt sur lignes/panneaux
      solaires, dégradation de la production PV → bascule vers le réseau.
    - Pics d'ozone et de NO₂ ⇒ afflux respiratoires aux urgences (consommation
      en stérilisation, oxygène, équipements respiratoires).
    - Indice UV élevé ⇒ corrélé aux vagues de chaleur (climatisation).

API publique sans clé, granularité horaire, données disponibles depuis 2013.
Documentation : https://open-meteo.com/en/docs/air-quality-api
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src.utils.config import (
    AIR_QUALITY_BASE,
    AIR_QUALITY_VARS,
    HOSPITAL_LOCATIONS,
    RAW_DIR,
)
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def fetch_air_quality(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Récupère la qualité de l'air horaire pour une position géographique."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AIR_QUALITY_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    logger.info(
        "Open-Meteo Air Quality lat=%.2f lon=%.2f [%s → %s]",
        lat, lon, start_date, end_date,
    )
    resp = requests.get(AIR_QUALITY_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo Air Quality : réponse sans section hourly")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "datetime"})

    rename_map = {col: f"air_{col}" for col in AIR_QUALITY_VARS if col in df.columns}
    df = df.rename(columns=rename_map)
    return df


def run(
    year: int | None = 2022,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    if start_date is None or end_date is None:
        if year is None:
            raise ValueError("Fournir `year` ou (`start_date`, `end_date`).")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    for name, coords in HOSPITAL_LOCATIONS.items():
        try:
            df = fetch_air_quality(
                lat=coords["lat"],
                lon=coords["lon"],
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            logger.warning("Air Quality %s : échec (%s)", name, exc)
            continue

        df["hospital"] = name
        save_csv(df, RAW_DIR / f"air_quality_{name}.csv")

    logger.info(
        "Ingestion qualité de l'air terminée [%s → %s].",
        start_date,
        end_date,
    )


def run_live(window_days: int = 30) -> None:
    """Récupère une fenêtre glissante récente (quasi temps réel)."""
    end = date.today()
    start = end - timedelta(days=window_days)
    run(year=None, start_date=start.isoformat(), end_date=end.isoformat())


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
