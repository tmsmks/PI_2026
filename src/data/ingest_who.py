"""
Ingestion des données OMS (WHO GHO API).

Récupère :
  - La fiabilité de l'électricité dans les hôpitaux par pays
  - La table de correspondance codes pays ↔ noms
"""

import logging

import pandas as pd
import requests

from src.utils.config import WHO_ENDPOINTS, RAW_DIR
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def fetch_who_reliability() -> pd.DataFrame:
    """
    GET /HCF_REL_ELECTRICITY → pourcentage d'hôpitaux avec électricité
    fiable (sans coupure > 2 h) par pays / année / zone (urbain/rural/total).
    """
    url = WHO_ENDPOINTS["reliability"]
    logger.info("Requête OMS fiabilité : %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("value", [])
    df = pd.DataFrame(records)

    cols_keep = [
        "SpatialDim",
        "TimeDim",
        "Dim1",
        "NumericValue",
        "ParentLocationCode",
        "ParentLocation",
    ]
    df = df[cols_keep].rename(columns={
        "SpatialDim": "country_code",
        "TimeDim": "year",
        "Dim1": "area_type",
        "NumericValue": "reliability_pct",
        "ParentLocationCode": "region_code",
        "ParentLocation": "region",
    })

    # Nettoyer la colonne area_type pour ne garder que le suffixe lisible
    df["area_type"] = (
        df["area_type"]
        .str.replace("RESIDENCEAREATYPE_", "", regex=False)
        .str.lower()
    )
    return df


def fetch_who_countries() -> pd.DataFrame:
    """Table de correspondance code ISO 3 lettres → nom de pays + région."""
    url = WHO_ENDPOINTS["countries"]
    logger.info("Requête OMS pays : %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("value", [])
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "Code": "country_code",
        "Title": "country_name",
        "ParentCode": "region_code",
        "ParentTitle": "region",
    })
    return df[["country_code", "country_name", "region_code", "region"]]


def run() -> None:
    reliability = fetch_who_reliability()
    save_csv(reliability, RAW_DIR / "who_reliability.csv")

    countries = fetch_who_countries()
    save_csv(countries, RAW_DIR / "who_countries.csv")

    logger.info("Ingestion OMS terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
