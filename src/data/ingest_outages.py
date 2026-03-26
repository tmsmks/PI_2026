"""
Ingestion des données de pannes / réseau sud-africain.

Sources :
  - ESK2033.csv : données horaires de production Eskom (43 824 lignes, 2018-2022)
    Colonnes utiles : Manual Load_Reduction(MLR), ILS Usage → indicateurs de délestage
  - EskomSePush_history.csv : historique des niveaux de load shedding (670 lignes)
    Colonnes : created_at, stage (0 = pas de délestage, 2-6 = niveaux)

Ces données servent à créer des features de contexte réseau,
pas comme axe principal (celui-ci est Lacor).
"""

import logging

import pandas as pd

from src.utils.config import SA_ESKOM_FILE, SA_LOADSHED_FILE, RAW_DIR
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def load_eskom_production() -> pd.DataFrame:
    """Charge et nettoie les données de production Eskom."""
    logger.info("Chargement Eskom production : %s", SA_ESKOM_FILE)
    df = pd.read_csv(SA_ESKOM_FILE, low_memory=False)

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "datetime_str"})

    # Les dates sont dans la première colonne mais avec un format particulier
    # On garde les colonnes les plus utiles pour l'analyse du réseau
    cols_keep = [
        "datetime_str",
        "Residual Demand",
        "Dispatchable Generation",
        "Thermal Generation",
        "Nuclear Generation",
        "Wind",
        "PV",
        "Total RE",
        "Manual Load_Reduction(MLR)",
        "ILS Usage",
    ]
    available = [c for c in cols_keep if c in df.columns]
    df = df[available].copy()

    # Convertir les colonnes numériques
    for col in df.columns:
        if col != "datetime_str":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Eskom : %d lignes, %d colonnes", len(df), len(df.columns))
    return df


def load_loadshedding_history() -> pd.DataFrame:
    """Charge l'historique des niveaux de load shedding."""
    logger.info("Chargement EskomSePush history : %s", SA_LOADSHED_FILE)
    df = pd.read_csv(SA_LOADSHED_FILE)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    n_events = len(df)
    max_stage = df["stage"].max()
    logger.info(
        "Load shedding history : %d événements, stage max = %d",
        n_events, max_stage,
    )
    return df


def run() -> None:
    eskom = load_eskom_production()
    save_csv(eskom, RAW_DIR / "eskom_production_clean.csv")

    loadshed = load_loadshedding_history()
    save_csv(loadshed, RAW_DIR / "loadshedding_history_clean.csv")

    logger.info("Ingestion données pannes terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
