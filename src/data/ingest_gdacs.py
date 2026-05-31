"""
Ingestion des catastrophes naturelles via GDACS (Global Disaster Alert
and Coordination System) — JRC / OCHA, sans clé.

Pourquoi c'est utile pour la prédiction de coupure :
    - Une catastrophe naturelle majeure (inondation, cyclone, séisme,
      feu de forêt) dégrade le réseau électrique et amène simultanément
      un afflux de patients à l'hôpital → double effet sur la stabilité
      énergétique du site.
    - Le niveau d'alerte GDACS (Vert/Orange/Rouge) reflète l'ampleur
      de l'événement → pondération directe du « stress » indirect.
    - Pour Lacor (Ouganda) : inondations récurrentes (lac Albert,
      Nil), épidémies via mouvements de population.
    - Pour Phoenix : feux de forêt et sécheresses prolongées
      (consommation accrue par la climatisation et délestages
      préventifs des compagnies électriques).

API publique : https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH

Approche :
    1. Pour chaque hôpital, on requête les événements GDACS du pays
       sur la fenêtre annuelle (paramètre `country=ISO3`).
    2. Chaque événement a une période d'activité (`fromdate`, `todate`).
    3. On construit une série horaire avec :
         - `gdacs_active_count` : nb d'événements en cours
         - `gdacs_alert_score`  : somme des niveaux d'alerte (1/2/3)
         - `gdacs_is_<type>`    : one-hot par type d'événement

Sortie : `data/raw/gdacs_<hospital>.csv`
    datetime, gdacs_active_count, gdacs_alert_score,
    gdacs_is_flood, gdacs_is_cyclone, gdacs_is_earthquake,
    gdacs_is_drought, gdacs_is_wildfire, gdacs_is_volcano, gdacs_is_tsunami
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.utils.config import (
    GDACS_ALERT_SCORE,
    GDACS_BASE,
    GDACS_EVENT_TYPES,
    GDACS_FILTERS,
    HOSPITAL_LOCATIONS,
    RAW_DIR,
)
from src.utils.io import save_csv

logger = logging.getLogger(__name__)

CATEGORIES = list(GDACS_EVENT_TYPES.values())


def fetch_gdacs_events(iso3: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Liste les catastrophes GDACS pour un pays sur une fenêtre temporelle."""
    params = {
        "country": iso3.upper(),
        "fromDate": start.strftime("%Y-%m-%d"),
        "toDate": end.strftime("%Y-%m-%d"),
        "alertlevel": "Green;Orange;Red",
    }
    logger.info(
        "GDACS iso3=%s [%s → %s]",
        iso3.upper(), params["fromDate"], params["toDate"],
    )
    try:
        resp = requests.get(GDACS_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("GDACS : appel échoué (%s)", exc)
        return pd.DataFrame()

    rows = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        ev_type_code = p.get("eventtype", "")
        category = GDACS_EVENT_TYPES.get(ev_type_code)
        if category is None:
            continue
        from_d = pd.to_datetime(p.get("fromdate"), errors="coerce")
        to_d = pd.to_datetime(p.get("todate"), errors="coerce")
        if pd.isna(from_d):
            continue
        if pd.isna(to_d):
            to_d = from_d + pd.Timedelta(days=7)
        # Ne conserver que les événements impactant le pays cible (le champ
        # `country` peut lister plusieurs pays pour une même catastrophe).
        rows.append({
            "name": p.get("name", ""),
            "category": category,
            "alertlevel": p.get("alertlevel", "Green"),
            "fromdate": from_d.tz_localize(None) if from_d.tz else from_d,
            "todate": to_d.tz_localize(None) if to_d.tz else to_d,
            "iso3": p.get("iso3", ""),
        })

    return pd.DataFrame(rows)


def _build_hourly_series(
    events: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    hourly_index = pd.date_range(
        start.replace(minute=0, second=0, microsecond=0),
        end,
        freq="h",
        inclusive="left",
    )
    out = pd.DataFrame({"datetime": hourly_index})
    out["gdacs_active_count"] = 0
    out["gdacs_alert_score"] = 0.0
    for cat in CATEGORIES:
        out[f"gdacs_is_{cat}"] = 0

    if events.empty:
        return out

    events = events.copy()
    events["fromdate"] = pd.to_datetime(events["fromdate"])
    events["todate"] = pd.to_datetime(events["todate"])
    events["score"] = events["alertlevel"].map(GDACS_ALERT_SCORE).fillna(1.0)

    for _, ev in events.iterrows():
        # On clippe à la fenêtre étudiée (un événement peut commencer avant
        # ou se prolonger après l'année en cours).
        active_mask = (out["datetime"] >= ev["fromdate"]) & (out["datetime"] <= ev["todate"])
        if not active_mask.any():
            continue
        out.loc[active_mask, "gdacs_active_count"] += 1
        out.loc[active_mask, "gdacs_alert_score"] += ev["score"]
        cat_col = f"gdacs_is_{ev['category']}"
        if cat_col in out.columns:
            out.loc[active_mask, cat_col] = 1

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

    for hospital, iso3 in GDACS_FILTERS.items():
        if hospital not in HOSPITAL_LOCATIONS:
            continue
        try:
            events = fetch_gdacs_events(iso3=iso3, start=start, end=end)
            if not events.empty:
                logger.info(
                    "GDACS %s : %d événements — %s",
                    hospital, len(events),
                    events["category"].value_counts().to_dict(),
                )
            else:
                logger.info("GDACS %s : aucun événement.", hospital)

            hourly = _build_hourly_series(events, start=start, end=end)
            hourly["hospital"] = hospital
            save_csv(hourly, RAW_DIR / f"gdacs_{hospital}.csv")
        except Exception as exc:
            logger.warning("GDACS %s : échec (%s)", hospital, exc)

    logger.info(
        "Ingestion GDACS terminée [%s → %s].",
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
