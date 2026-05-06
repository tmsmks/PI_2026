"""
Ingestion du signal événementiel GDELT DOC 2.0.

GDELT (Global Database of Events, Language and Tone) monitore la presse
mondiale en temps réel. On l'utilise ici pour obtenir un **signal de
contexte** là où on n'a pas d'API réseau électrique officielle :
    - volume d'articles mentionnant des coupures / délestage près de l'hôpital,
    - volume d'articles mentionnant des événements météo extrêmes,
    - tonalité moyenne (proxy de sévérité perçue).

Ce signal se marie bien avec la prédiction de coupures : une hausse du
volume « power outage Uganda » à J-1 indique un stress réseau plausible.

Approche :
    1. Pour chaque thème configuré (power, weather, health), appel à
       l'endpoint `TimelineVolRaw` puis `TimelineTone` (granularité jour).
    2. Les séries quotidiennes sont fusionnées en un seul DataFrame.
    3. On broadcast au pas horaire pour coller à l'index des autres
       features (même valeur pour les 24 heures d'un jour).

Sortie : `data/raw/gdelt_<hospital>.csv`
    datetime, gdelt_<theme>_vol, gdelt_<theme>_tone, hospital
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Literal

import pandas as pd
import requests

from src.utils.config import (
    GDELT_DOC_BASE,
    GDELT_QUERIES,
    HOSPITAL_LOCATIONS,
    RAW_DIR,
)
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


Mode = Literal["TimelineVolRaw", "TimelineTone"]


# ─── Appel API bas niveau ─────────────────────────────────────────────


def _call_gdelt(
    query: str,
    start: datetime,
    end: datetime,
    mode: Mode,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Appel GDELT DOC 2.0 et parsing en DataFrame.

    GDELT attend des timestamps au format YYYYMMDDHHMMSS (UTC).
    En mode `Timeline*`, la granularité est journalière : on reçoit une
    série (date, valeur) qu'on retourne telle quelle.
    """
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "timelinesmooth": 0,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(GDELT_DOC_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                logger.warning("GDELT échec définitif (%s) : %s", mode, exc)
                return pd.DataFrame(columns=["date", "value"])
            wait = 2 ** attempt
            logger.info("GDELT retry %d/%d dans %ds…", attempt, max_retries, wait)
            time.sleep(wait)

    timelines = data.get("timeline", [])
    if not timelines:
        return pd.DataFrame(columns=["date", "value"])

    series = timelines[0].get("data", [])
    if not series:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df = df.dropna(subset=["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df[["date", "value"]]


# ─── Construction de la série par thème ───────────────────────────────


def _fetch_theme(
    query: str,
    start: datetime,
    end: datetime,
    theme: str,
) -> pd.DataFrame:
    """Récupère volume + tonalité pour un thème donné, fusionnés par date."""
    vol = _call_gdelt(query, start, end, mode="TimelineVolRaw")
    vol = vol.rename(columns={"value": f"gdelt_{theme}_vol"})

    # GDELT limite (courtoisie) : on espace les requêtes
    time.sleep(1.0)

    tone = _call_gdelt(query, start, end, mode="TimelineTone")
    tone = tone.rename(columns={"value": f"gdelt_{theme}_tone"})

    if vol.empty and tone.empty:
        logger.warning("Thème %s : aucune donnée GDELT retournée.", theme)
        return pd.DataFrame(columns=["date", f"gdelt_{theme}_vol", f"gdelt_{theme}_tone"])

    merged = pd.merge(vol, tone, on="date", how="outer").fillna(0.0)
    logger.info("Thème %-8s : %d jours ramenés (volume moyen=%.1f)",
                theme, len(merged), merged[f"gdelt_{theme}_vol"].mean())
    return merged


# ─── Passage quotidien → horaire ──────────────────────────────────────


def _broadcast_to_hourly(
    daily: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Broadcast chaque valeur quotidienne sur les 24 heures du jour.

    On produit un index horaire continu entre `start` et `end` puis on
    forward-fill depuis la valeur du jour courant.
    """
    if daily.empty:
        hourly_index = pd.date_range(start, end, freq="h", inclusive="left")
        return pd.DataFrame({"datetime": hourly_index})

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.floor("D")
    daily = daily.set_index("date").sort_index()

    hourly_index = pd.date_range(
        start.replace(hour=0, minute=0, second=0),
        end,
        freq="h",
        inclusive="left",
    )
    hourly = daily.reindex(
        pd.to_datetime(hourly_index.floor("D").unique()),
        method="ffill",
    )
    hourly = hourly.reindex(
        pd.to_datetime(hourly_index.floor("D").unique()),
        method="bfill",
    ) if hourly.isna().all().any() else hourly

    # Construire le DataFrame final horaire en dupliquant la ligne du jour
    map_daily = hourly.copy()
    map_daily.index.name = "day"
    out = pd.DataFrame({"datetime": hourly_index})
    out["day"] = out["datetime"].dt.floor("D")
    out = out.merge(map_daily, left_on="day", right_index=True, how="left")
    out = out.drop(columns=["day"])
    numeric = out.select_dtypes(include="number").columns
    out[numeric] = out[numeric].fillna(0)
    return out


# ─── Pipeline public ──────────────────────────────────────────────────


def fetch_gdelt_signal(
    hospital: str,
    year: int = 2022,
) -> pd.DataFrame:
    """Construit la série horaire GDELT pour un hôpital et une année."""
    if hospital not in GDELT_QUERIES:
        raise ValueError(
            f"Aucune requête GDELT définie pour '{hospital}'. "
            f"Sites disponibles : {list(GDELT_QUERIES)}"
        )

    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)

    themes_df: list[pd.DataFrame] = []
    for theme, query in GDELT_QUERIES[hospital].items():
        logger.info("GDELT %s / %s : %s", hospital, theme, query)
        theme_df = _fetch_theme(query, start, end, theme)
        themes_df.append(theme_df)

    # Fusion de tous les thèmes sur la date
    merged_daily = themes_df[0]
    for extra in themes_df[1:]:
        merged_daily = pd.merge(merged_daily, extra, on="date", how="outer")
    merged_daily = merged_daily.fillna(0).sort_values("date").reset_index(drop=True)

    hourly = _broadcast_to_hourly(merged_daily, start=start, end=end)
    hourly["hospital"] = hospital
    return hourly


def run(year: int = 2022) -> None:
    for hospital in HOSPITAL_LOCATIONS:
        if hospital not in GDELT_QUERIES:
            logger.info("GDELT : pas de requête pour %s, ignoré.", hospital)
            continue
        try:
            df = fetch_gdelt_signal(hospital=hospital, year=year)
            save_csv(df, RAW_DIR / f"gdelt_{hospital}.csv")
        except Exception as exc:
            logger.warning("GDELT %s : échec ingestion (%s)", hospital, exc)

    logger.info("Ingestion GDELT terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
