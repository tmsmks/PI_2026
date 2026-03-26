"""
Ingestion des vrais datasets de consommation hospitalière.

Dataset principal : LACOR HOSPITAL (Ouganda)
  - Source  : Zenodo (doi:10.5281/zenodo.7466652)
  - Format  : 35 040 lignes × 7 colonnes, toutes les 15 min, année 2022
  - Colonnes clés :
      * Total load kW    → consommation totale
      * Grid avail       → 1 = réseau OK, 0 = coupure  ← VARIABLE CIBLE
      * Solar PV kW      → production solaire
      * Generators kW    → usage du générateur de secours
      * Base load kW     → charge de base

Dataset complémentaire : PHOENIX HOSPITAL (USA)
  - Source  : GitHub (Shahid-Fakhri/Electricity-Consumption)
  - Format  : 8 760 lignes × 11 colonnes, horaires, 1 an
  - Pas de colonne coupure → utilisé pour enrichissement / benchmark
"""

import logging

import pandas as pd

from src.utils.config import LACOR_FILE, PHOENIX_FILE, RAW_DIR
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def load_lacor() -> pd.DataFrame:
    """Charge et nettoie le dataset Lacor Hospital."""
    logger.info("Chargement Lacor Hospital : %s", LACOR_FILE)
    df = pd.read_excel(LACOR_FILE, sheet_name="Sheet1")

    df = df.rename(columns={
        "Unnamed: 0": "datetime",
        "Solar PV kW": "solar_pv_kw",
        "Total load kW": "total_load_kw",
        "Generators kW": "generators_kw",
        "Sterilization kW": "sterilization_kw",
        "Base load kW": "base_load_kw",
        "Grid avail": "grid_available",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Variable cible : is_outage = inverse de grid_available
    df["is_outage"] = (1 - df["grid_available"]).astype(int)

    logger.info(
        "Lacor : %d lignes, plage %s → %s, coupures=%d (%.1f%%)",
        len(df),
        df["datetime"].min().date(),
        df["datetime"].max().date(),
        df["is_outage"].sum(),
        100 * df["is_outage"].mean(),
    )
    return df


def load_phoenix() -> pd.DataFrame:
    """Charge et nettoie le dataset Phoenix Hospital."""
    logger.info("Chargement Phoenix Hospital : %s", PHOENIX_FILE)
    df = pd.read_excel(PHOENIX_FILE, sheet_name="in")

    df = df.rename(columns={
        "Date/Time": "datetime_str",
        "Electricity:Facility [kW](Hourly)": "total_electricity_kw",
        "Fans:Electricity [kW](Hourly)": "fans_kw",
        "Cooling:Electricity [kW](Hourly)": "cooling_kw",
        "Heating:Electricity [kW](Hourly)": "heating_kw",
        "InteriorLights:Electricity [kW](Hourly)": "lights_kw",
        "InteriorEquipment:Electricity [kW](Hourly)": "equipment_kw",
        "Gas:Facility [kW](Hourly)": "total_gas_kw",
    })

    # Le format date Phoenix est "MM/DD  HH:MM:SS" sans année → on ajoute 2022
    df["datetime"] = pd.to_datetime(
        "2022/" + df["datetime_str"].str.strip(), format="%Y/%m/%d  %H:%M:%S"
    )
    df = df.drop(columns=["datetime_str"])
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info("Phoenix : %d lignes, plage %s → %s", len(df), df["datetime"].min(), df["datetime"].max())
    return df


def run() -> None:
    lacor = load_lacor()
    save_csv(lacor, RAW_DIR / "lacor_clean.csv")

    phoenix = load_phoenix()
    save_csv(phoenix, RAW_DIR / "phoenix_clean.csv")

    logger.info("Ingestion consommation terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
