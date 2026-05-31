"""
Ingestion des séismes proches via USGS Earthquake Catalog.

Pourquoi c'est utile pour la prédiction de coupure :
    - Un séisme même modéré peut endommager les transformateurs, sous-stations
      et lignes aériennes desservant l'hôpital → coupure persistante.
    - L'Ouganda est traversé par le Rift Albertin (zone sismique active),
      Phoenix est dans une zone sismique secondaire mais l'Ouest US a des
      séismes fréquents qui se propagent.
    - Un séisme déclenche un afflux massif aux urgences (traumatologie,
      stérilisation, blocs opératoires) → pics de consommation post-événement.

API publique gratuite, sans clé : https://earthquake.usgs.gov/fdsnws/event/1/

Approche :
    1. Requête sur la fenêtre temporelle (1 an) avec rayon configurable
       autour de chaque hôpital.
    2. Pour chaque séisme, on calcule un "stress sismique" qui décroît
       exponentiellement après l'événement (impact ressenti pendant 24 h,
       résiduel pendant 7 jours).
    3. Agrégation horaire produisant des features prêtes à fusionner.

Sortie : `data/raw/usgs_earthquake_<hospital>.csv`
    datetime, eq_recent_count_24h, eq_max_mag_24h, eq_stress, eq_distance_min_km
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from src.utils.config import (
    HOSPITAL_LOCATIONS,
    RAW_DIR,
    USGS_EARTHQUAKE_BASE,
    USGS_MIN_MAGNITUDE,
    USGS_SEARCH_RADIUS_KM,
)
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Distance Haversine en kilomètres entre un point fixe et une série."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def fetch_earthquakes(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    radius_km: float = USGS_SEARCH_RADIUS_KM,
    min_magnitude: float = USGS_MIN_MAGNITUDE,
) -> pd.DataFrame:
    """Liste les séismes proches d'un point dans une fenêtre temporelle."""
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    logger.info(
        "USGS lat=%.2f lon=%.2f rayon=%dkm M≥%.1f [%s → %s]",
        lat, lon, radius_km, min_magnitude,
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
    )
    resp = requests.get(USGS_EARTHQUAKE_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None, None])
        ts_ms = props.get("time")
        if ts_ms is None or coords[0] is None:
            continue
        rows.append({
            "datetime": pd.to_datetime(ts_ms, unit="ms", utc=True),
            "magnitude": props.get("mag", 0.0),
            "depth_km": coords[2] or 0.0,
            "place": props.get("place", ""),
            "lon": coords[0],
            "lat": coords[1],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["distance_km"] = _haversine_km(lat, lon, df["lat"], df["lon"])
    return df


def _build_hourly_stress(
    events: pd.DataFrame,
    hospital_lat: float,
    hospital_lon: float,
    start: datetime,
    end: datetime,
    decay_hours: float = 24.0,
) -> pd.DataFrame:
    """Convertit la liste d'événements ponctuels en série horaire continue.

    Le « stress sismique » à l'instant t pour un séisme survenu en t₀ :
        stress(t) = magnitude × exp(-(t-t₀) / decay) × (1 - distance/radius)
    Pour chaque heure on somme les stress résiduels de tous les séismes passés.

    Implémentation vectorisée : on matérialise la matrice (heures × events)
    en numpy via broadcasting plutôt qu'une boucle Python sur 8 760 itérations.
    Pour ~500 événements c'est ~100× plus rapide.
    """
    hourly_index = pd.date_range(
        start.replace(minute=0, second=0, microsecond=0),
        end,
        freq="h",
        inclusive="left",
    )
    out = pd.DataFrame({"datetime": hourly_index})
    out["eq_stress"] = 0.0
    out["eq_recent_count_24h"] = 0
    out["eq_max_mag_24h"] = 0.0
    out["eq_distance_min_km"] = float(USGS_SEARCH_RADIUS_KM)

    if events.empty:
        return out

    events = events.copy()
    events["datetime"] = pd.to_datetime(events["datetime"]).dt.tz_localize(None)
    events = events[(events["datetime"] >= start) & (events["datetime"] < end)]
    if events.empty:
        return out

    proximity = (1.0 - (events["distance_km"] / USGS_SEARCH_RADIUS_KM)).clip(0, 1)
    events["proximity"] = proximity

    # Matrice delta_h (H heures × E événements) en heures depuis le séisme.
    hours_np = out["datetime"].to_numpy(dtype="datetime64[ns]")
    ev_np = events["datetime"].to_numpy(dtype="datetime64[ns]")
    delta_h = (hours_np[:, None] - ev_np[None, :]).astype("timedelta64[s]").astype(np.float64) / 3600.0

    mag_np = events["magnitude"].to_numpy(dtype=np.float64)
    prox_np = events["proximity"].to_numpy(dtype=np.float64)
    dist_np = events["distance_km"].to_numpy(dtype=np.float64)

    active = delta_h >= 0
    decay = np.where(active, np.exp(-np.clip(delta_h, 0, None) / decay_hours), 0.0)
    stress_matrix = decay * mag_np[None, :] * prox_np[None, :]
    out["eq_stress"] = stress_matrix.sum(axis=1)

    recent = active & (delta_h <= 24)
    has_recent = recent.any(axis=1)
    out["eq_recent_count_24h"] = recent.sum(axis=1).astype(int)
    if has_recent.any():
        # max(magnitude) et min(distance) sur la fenêtre 24 h, par broadcasting.
        mag_masked = np.where(recent, mag_np[None, :], -np.inf)
        out.loc[has_recent, "eq_max_mag_24h"] = mag_masked[has_recent].max(axis=1)
        dist_masked = np.where(recent, dist_np[None, :], np.inf)
        out.loc[has_recent, "eq_distance_min_km"] = dist_masked[has_recent].min(axis=1)

    return out


def run(
    year: int | None = 2022,
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    if start is None or end is None:
        if year is None:
            raise ValueError("Fournir `year` ou (`start`, `end`).")
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)

    for name, coords in HOSPITAL_LOCATIONS.items():
        try:
            events = fetch_earthquakes(
                lat=coords["lat"],
                lon=coords["lon"],
                start=start,
                end=end,
            )
        except Exception as exc:
            logger.warning("USGS %s : échec (%s)", name, exc)
            continue

        if not events.empty:
            top_mag = events["magnitude"].max()
            logger.info(
                "USGS %s : %d séismes (M≥%.1f), max=%.1f",
                name, len(events), USGS_MIN_MAGNITUDE, top_mag,
            )
        else:
            logger.info("USGS %s : aucun séisme dans la fenêtre.", name)

        hourly = _build_hourly_stress(
            events,
            hospital_lat=coords["lat"],
            hospital_lon=coords["lon"],
            start=start,
            end=end,
        )
        hourly["hospital"] = name
        save_csv(hourly, RAW_DIR / f"usgs_earthquake_{name}.csv")

    logger.info(
        "Ingestion USGS terminée [%s → %s].",
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


def run_live(window_days: int = 30) -> None:
    """Récupère une fenêtre glissante récente (quasi temps réel)."""
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=window_days)
    run(year=None, start=start, end=end)


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
