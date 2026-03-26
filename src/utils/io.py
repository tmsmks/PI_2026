"""
Fonctions utilitaires d'entrée/sortie (lecture, sauvegarde, logs).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Sauvegardé %s  (%d lignes, %d colonnes)", path.name, len(df), len(df.columns))


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    logger.info("Chargé %s  (%d lignes, %d colonnes)", path.name, len(df), len(df.columns))
    return df
