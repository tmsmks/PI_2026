"""
Ingestion des événements météo extrêmes via NOAA Storm Events.

NOAA Storm Events Database : catalogue officiel américain des phénomènes
météorologiques significatifs (orages, tornades, inondations, canicules,
tempêtes hivernales, etc.), souvent corrélés aux pannes électriques.

Couverture géographique : USA uniquement (Phoenix dans notre cas).
Pour les sites hors USA (Lacor/Ouganda), on utilise d'autres sources
(Open-Meteo pour la météo brute, GDELT pour le signal événementiel).

Le module :
    1. Télécharge (et cache) le fichier « details » d'une année donnée
       depuis le répertoire public NOAA NCEI.
    2. Filtre par état/comté correspondant à chaque hôpital.
    3. Produit une série horaire alignée sur l'index temporel des autres
       features (un événement couvrant N heures est étalé sur ces N heures).
    4. Agrège plusieurs indicateurs : présence d'événement, sévérité,
       dommages matériels, blessés/décès, catégories d'événement.

Sortie : `data/raw/noaa_storm_<hospital>.csv`, colonnes :
    datetime, storm_active, storm_event_count, storm_magnitude_max,
    storm_damage_property_usd, storm_injuries, storm_deaths,
    storm_is_thunderstorm, storm_is_flood, storm_is_wind,
    storm_is_heat, storm_is_winter, storm_is_dust, hospital
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import (
    HOSPITAL_LOCATIONS,
    NOAA_STORM_BASE,
    NOAA_STORM_CACHE_DIR,
    NOAA_STORM_EVENT_GROUPS,
    NOAA_STORM_FILTERS,
    RAW_DIR,
)
from src.utils.io import save_csv

logger = logging.getLogger(__name__)

DETAILS_FILE_PATTERN = re.compile(
    r'StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz',
)


# ─── Téléchargement + cache ────────────────────────────────────────────


def _resolve_details_filename(year: int) -> str:
    """Résout le nom exact du fichier « details » pour une année donnée.

    Le suffixe `c<YYYYMMDD>` change à chaque republication NOAA : on liste
    le répertoire HTML et on prend la version la plus récente de l'année.
    """
    logger.info("Résolution du fichier NOAA pour l'année %d…", year)
    resp = requests.get(NOAA_STORM_BASE, timeout=60)
    resp.raise_for_status()

    candidates = [
        m.group(0)
        for m in DETAILS_FILE_PATTERN.finditer(resp.text)
        if m.group(1) == str(year)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier NOAA Storm Events trouvé pour {year} sur {NOAA_STORM_BASE}"
        )
    latest = sorted(set(candidates))[-1]
    logger.info("Fichier retenu : %s", latest)
    return latest


def _download_details(year: int) -> Path:
    """Télécharge (si absent) le CSV gz NOAA pour une année et retourne
    le chemin vers la version décompressée en cache local."""
    NOAA_STORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cached_csv = NOAA_STORM_CACHE_DIR / f"storm_events_details_{year}.csv"
    if cached_csv.exists():
        logger.info("Cache hit : %s", cached_csv.name)
        return cached_csv

    filename = _resolve_details_filename(year)
    gz_path = NOAA_STORM_CACHE_DIR / filename
    url = f"{NOAA_STORM_BASE}{filename}"

    logger.info("Téléchargement %s …", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    logger.info("Décompression vers %s", cached_csv.name)
    with gzip.open(gz_path, "rb") as src, open(cached_csv, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink(missing_ok=True)

    return cached_csv


# ─── Parsing + filtrage ────────────────────────────────────────────────


def _parse_damage(value: object) -> float:
    """Convertit une chaîne NOAA type « 10.00K » ou « 5.00M » en USD.

    Formats connus : "", "0", "10.00K", "1.50M", "2.00B".
    """
    if value is None:
        return 0.0
    s = str(value).strip().upper()
    if not s or s == "NAN":
        return 0.0
    multiplier = 1.0
    if s.endswith("K"):
        multiplier, s = 1_000.0, s[:-1]
    elif s.endswith("M"):
        multiplier, s = 1_000_000.0, s[:-1]
    elif s.endswith("B"):
        multiplier, s = 1_000_000_000.0, s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return 0.0


def _load_and_filter(
    csv_path: Path,
    state: str,
    county: str,
) -> pd.DataFrame:
    """Charge le CSV brut et filtre sur état + comté cible."""
    usecols = [
        "BEGIN_DATE_TIME",
        "END_DATE_TIME",
        "STATE",
        "CZ_NAME",
        "EVENT_TYPE",
        "MAGNITUDE",
        "DAMAGE_PROPERTY",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=str, low_memory=False)

    df["STATE"] = df["STATE"].str.upper().str.strip()
    df["CZ_NAME"] = df["CZ_NAME"].str.upper().str.strip()
    mask = (df["STATE"] == state.upper()) & (df["CZ_NAME"] == county.upper())
    filtered = df[mask].copy()

    logger.info(
        "Filtrage %s / %s : %d événements retenus sur %d bruts",
        state, county, len(filtered), len(df),
    )
    return filtered


# ─── Agrégation horaire ────────────────────────────────────────────────


def _classify_event_type(event_type: str) -> dict[str, int]:
    """Retourne un one-hot des catégories définies dans la config."""
    flags = {f"storm_is_{grp}": 0 for grp in NOAA_STORM_EVENT_GROUPS}
    for group, labels in NOAA_STORM_EVENT_GROUPS.items():
        if event_type in labels:
            flags[f"storm_is_{group}"] = 1
    return flags


def _to_hourly_series(events: pd.DataFrame, year: int) -> pd.DataFrame:
    """Étale chaque événement sur les heures qu'il couvre puis agrège.

    NOAA stocke BEGIN/END en heure locale ; on parse au format
    « DD-MMM-YY HH:MM:SS » (ex : « 15-JUN-22 14:30:00 »).
    """
    if events.empty:
        hourly_index = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq="h",
        )
        return pd.DataFrame(
            {"datetime": hourly_index}
            | {"storm_active": 0, "storm_event_count": 0}
            | {"storm_magnitude_max": 0.0, "storm_damage_property_usd": 0.0}
            | {"storm_injuries": 0, "storm_deaths": 0}
            | {f"storm_is_{g}": 0 for g in NOAA_STORM_EVENT_GROUPS}
        )

    dt_format = "%d-%b-%y %H:%M:%S"
    events = events.copy()
    events["begin"] = pd.to_datetime(
        events["BEGIN_DATE_TIME"], format=dt_format, errors="coerce",
    )
    events["end"] = pd.to_datetime(
        events["END_DATE_TIME"], format=dt_format, errors="coerce",
    )
    events = events.dropna(subset=["begin", "end"])
    events["begin"] = events["begin"].dt.floor("h")
    events["end"] = events["end"].dt.ceil("h")

    events["magnitude"] = pd.to_numeric(events["MAGNITUDE"], errors="coerce").fillna(0.0)
    events["damage_usd"] = events["DAMAGE_PROPERTY"].map(_parse_damage)
    for col in ("INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"):
        events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0).astype(int)
    events["injuries"] = events["INJURIES_DIRECT"] + events["INJURIES_INDIRECT"]
    events["deaths"] = events["DEATHS_DIRECT"] + events["DEATHS_INDIRECT"]

    rows: list[dict] = []
    for _, ev in events.iterrows():
        hours = pd.date_range(ev["begin"], ev["end"], freq="h", inclusive="left")
        flags = _classify_event_type(str(ev["EVENT_TYPE"]).strip())
        per_hour_damage = ev["damage_usd"] / max(len(hours), 1)
        per_hour_injuries = ev["injuries"] / max(len(hours), 1)
        per_hour_deaths = ev["deaths"] / max(len(hours), 1)
        for h in hours:
            rows.append({
                "datetime": h,
                "storm_event_count": 1,
                "storm_magnitude_max": ev["magnitude"],
                "storm_damage_property_usd": per_hour_damage,
                "storm_injuries": per_hour_injuries,
                "storm_deaths": per_hour_deaths,
                **flags,
            })

    if not rows:
        return _to_hourly_series(pd.DataFrame(), year)

    expanded = pd.DataFrame(rows)
    agg = {
        "storm_event_count": "sum",
        "storm_magnitude_max": "max",
        "storm_damage_property_usd": "sum",
        "storm_injuries": "sum",
        "storm_deaths": "sum",
        **{f"storm_is_{g}": "max" for g in NOAA_STORM_EVENT_GROUPS},
    }
    hourly = expanded.groupby("datetime", as_index=False).agg(agg)

    full_index = pd.date_range(
        f"{year}-01-01", f"{year}-12-31 23:00", freq="h",
    )
    hourly = (
        hourly.set_index("datetime")
        .reindex(full_index, fill_value=0)
        .rename_axis("datetime")
        .reset_index()
    )
    hourly["storm_active"] = (hourly["storm_event_count"] > 0).astype(int)
    cols = [
        "datetime", "storm_active", "storm_event_count",
        "storm_magnitude_max", "storm_damage_property_usd",
        "storm_injuries", "storm_deaths",
        *[f"storm_is_{g}" for g in NOAA_STORM_EVENT_GROUPS],
    ]
    return hourly[cols]


# ─── Pipeline public ───────────────────────────────────────────────────


def fetch_storm_events(
    hospital: str,
    year: int = 2022,
) -> pd.DataFrame:
    """Point d'entrée unitaire : retourne la série horaire NOAA pour un hôpital."""
    if hospital not in NOAA_STORM_FILTERS:
        raise ValueError(
            f"Pas de filtre NOAA défini pour '{hospital}'. "
            f"Sites supportés : {list(NOAA_STORM_FILTERS)}"
        )

    filt = NOAA_STORM_FILTERS[hospital]
    csv_path = _download_details(year)
    events = _load_and_filter(csv_path, state=filt["state"], county=filt["county"])
    hourly = _to_hourly_series(events, year=year)
    hourly["hospital"] = hospital
    return hourly


def run(year: int = 2022) -> None:
    """Ingère les événements NOAA pour tous les hôpitaux US et sauvegarde."""
    for hospital in HOSPITAL_LOCATIONS:
        if hospital not in NOAA_STORM_FILTERS:
            logger.info("NOAA : site non-US ignoré (%s).", hospital)
            continue
        df = fetch_storm_events(hospital=hospital, year=year)
        save_csv(df, RAW_DIR / f"noaa_storm_{hospital}.csv")

    logger.info("Ingestion NOAA Storm Events terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
