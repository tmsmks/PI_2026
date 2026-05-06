"""
Ingestion des données météo historiques via Open-Meteo Archive API.

Récupère l'historique horaire 2022 (année du dataset Lacor)
pour chaque hôpital de référence.
"""

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src.utils.config import METEO_BASE, METEO_HOURLY_VARS, HOSPITAL_LOCATIONS, RAW_DIR
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def fetch_meteo_archive(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Appelle Open-Meteo Archive API pour récupérer la météo historique."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(METEO_HOURLY_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    logger.info("Requête Open-Meteo Archive lat=%.2f lon=%.2f [%s → %s]", lat, lon, start_date, end_date)
    resp = requests.get(METEO_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "datetime"})
    return df


def run(
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
) -> None:
    """Récupère la météo historique sur une plage de dates."""
    for name, coords in HOSPITAL_LOCATIONS.items():
        df = fetch_meteo_archive(
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start_date,
            end_date=end_date,
        )
        df["hospital"] = name
        save_csv(df, RAW_DIR / f"meteo_{name}.csv")

    logger.info(
        "Ingestion météo terminée [%s → %s].",
        start_date,
        end_date,
    )


def run_live(window_days: int = 30) -> None:
    """Récupère une fenêtre glissante récente (quasi temps réel)."""
    end = date.today()
    start = end - timedelta(days=window_days)
    run(start_date=start.isoformat(), end_date=end.isoformat())


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
