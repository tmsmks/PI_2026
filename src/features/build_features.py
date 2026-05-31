"""
Feature Engineering pour la prédiction de coupures (données réelles Lacor).

Variables créées :
  ── Temporelles ──
  - hour, day_of_week, month              (déjà dans le dataset)
  - is_weekend                            (booléen)
  - hour_sin, hour_cos                    (encodage cyclique)
  - month_sin, month_cos                  (encodage cyclique)
  - is_public_holiday                     (jours fériés ougandais)

  ── Consommation (rolling) ──
  - load_rolling_6h                       (moyenne glissante 6h)
  - load_rolling_24h                      (moyenne glissante 24h)
  - load_std_24h                          (écart-type glissant 24h)
  - load_diff_1h                          (variation heure par heure)
  - load_diff_24h                         (variation jour par jour)
  - load_pct_change_1h                    (variation relative %)
  - peak_ratio                            (ratio charge / moyenne 24h)

  ── Sources d'énergie ──
  - solar_ratio                           (part du solaire dans la charge)
  - generator_active                      (générateur en marche ? 0/1)
  - generator_ratio                       (part du générateur)
  - base_load_ratio                       (ratio base / total)
  - grid_availability_rolling_6h          (stabilité réseau 6h glissant)
  - recent_outages_6h                     (nombre de coupures dans les 6h)
  - recent_outages_24h                    (nombre de coupures dans les 24h)

  ── Historique coupures ──
  - hours_since_last_outage               (heures depuis dernière coupure)
  - last_outage_duration_h                (durée de la dernière coupure en heures)
  - avg_outage_duration_7d                (durée moyenne des coupures sur 7 jours)
  - outage_frequency_7d                   (nombre de coupures sur 7 jours glissants)
  - outage_trend_7d                       (tendance coupures : 7d récents vs 7d précédents)

  ── Météo ──
  - temp_humidity_interaction
  - wind_precipitation_interaction
  - solar_available                       (rayonnement > 50 W/m²)
  - heat_stress                           (température > 30°C en Ouganda)

  ── Météo avancée ──
  - cloud_cover_pct                       (couverture nuageuse %)
  - dew_point_2m                          (point de rosée)
  - visibility_m                          (visibilité en mètres)
  - evapotranspiration                    (évapotranspiration)
  - rain_intensity                        (intensité pluie = précip × vent)
  - thermal_amplitude_24h                 (amplitude thermique sur 24h)
  - humidity_change_3h                    (variation humidité sur 3h)
  - pressure_change_3h                    (variation pression sur 3h)

Features retirées (importance 0, constantes sur la série mono-hôpital) :
  - storm_risk
"""

import logging
from time import perf_counter

import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURES_DIR,
    PROCESSED_DIR,
    UGANDA_PUBLIC_HOLIDAYS_2022,
)
from src.utils.io import load_csv, load_table, save_table

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    holidays = pd.to_datetime(UGANDA_PUBLIC_HOLIDAYS_2022)
    df["is_public_holiday"] = df["datetime"].dt.normalize().isin(holidays).astype(int)

    return df


def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    col = "total_load_kw"
    by_hospital = df.groupby("hospital", sort=False)[col]

    df["load_rolling_6h"] = by_hospital.transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["load_rolling_24h"] = by_hospital.transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["load_std_24h"] = by_hospital.transform(lambda s: s.rolling(24, min_periods=1).std()).fillna(0)

    df["load_diff_1h"] = by_hospital.diff().fillna(0)
    df["load_diff_24h"] = by_hospital.diff(24).fillna(0)
    df["load_pct_change_1h"] = by_hospital.pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    df["peak_ratio"] = (df[col] / df["load_rolling_24h"]).fillna(1).replace([np.inf, -np.inf], 1)

    return df


def add_energy_source_features(df: pd.DataFrame) -> pd.DataFrame:
    total = df["total_load_kw"].replace(0, np.nan)

    if "solar_pv_kw" in df.columns:
        df["solar_ratio"] = (df["solar_pv_kw"] / total).fillna(0).clip(0, 1)

    if "generators_kw" in df.columns:
        df["generator_active"] = (df["generators_kw"] > 1.0).astype(int)
        df["generator_ratio"] = (df["generators_kw"] / total).fillna(0).clip(0, 1)

    if "base_load_kw" in df.columns:
        df["base_load_ratio"] = (df["base_load_kw"] / total).fillna(0).clip(0, 1)

    if "grid_availability_ratio" in df.columns:
        df["grid_availability_rolling_6h"] = (
            df.groupby("hospital", sort=False)["grid_availability_ratio"]
            .transform(lambda s: s.rolling(6, min_periods=1).mean())
        )

    # Historique récent de coupures : on shifte de 1 heure pour éviter
    # d'inclure l'observation courante (sinon la cible fuite directement
    # dans la feature). Ces colonnes restent listées dans COLS_TO_DROP
    # côté training, mais le calcul propre les rend réutilisables pour
    # l'inspection / l'analyse historique sans risque.
    if "is_outage" in df.columns:
        shifted_outage = (
            df.groupby("hospital", sort=False)["is_outage"]
            .shift(1)
            .fillna(0)
        )
        df["recent_outages_6h"] = (
            shifted_outage.groupby(df["hospital"], sort=False)
            .transform(lambda s: s.rolling(6, min_periods=1).sum())
        )
        df["recent_outages_24h"] = (
            shifted_outage.groupby(df["hospital"], sort=False)
            .transform(lambda s: s.rolling(24, min_periods=1).sum())
        )

    return df


def add_outage_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées de l'historique des coupures (sans fuite de données)."""
    if "is_outage" not in df.columns:
        logger.info("Pas de colonne is_outage — features historiques ignorées.")
        return df

    shifted = (
        df.groupby("hospital", sort=False)["is_outage"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    df["_shifted_outage"] = shifted

    def _per_hospital_outage_features(grp: pd.DataFrame) -> pd.DataFrame:
        s = grp["_shifted_outage"]
        groups = (s != s.shift(1)).cumsum()
        grp["hours_since_last_outage"] = (s == 0).groupby(groups).cumsum().fillna(0)

        outage_starts = (s == 1) & (s.shift(1) == 0)
        outage_ends = (s == 0) & (s.shift(1) == 1)
        durations = s.groupby(outage_starts.cumsum()).transform("sum")
        grp["last_outage_duration_h"] = durations.where(outage_ends).ffill().fillna(0)

        grp["outage_frequency_7d"] = s.rolling(168, min_periods=1).sum()
        outage_hours_7d = s.rolling(168, min_periods=1).sum()
        # shift(fill_value=False) garde le dtype booléen (pas de NaN), ce qui
        # évite le FutureWarning pandas de downcasting object → bool sur fillna.
        outage_events_7d = outage_starts.shift(1, fill_value=False).rolling(168, min_periods=1).sum()
        grp["avg_outage_duration_7d"] = (
            outage_hours_7d / outage_events_7d.replace(0, np.nan)
        ).fillna(0)

        recent_7d = s.rolling(168, min_periods=1).sum()
        prev_7d = s.shift(168).rolling(168, min_periods=1).sum()
        grp["outage_trend_7d"] = (recent_7d / prev_7d.replace(0, np.nan)).fillna(1.0).clip(0, 10)
        return grp

    # Boucle explicite par hôpital plutôt que `groupby.apply` : évite le
    # FutureWarning « apply operated on the grouping columns » (pandas
    # exclura bientôt la colonne de groupe) et garde un comportement stable.
    # `reindex(original_index)` restitue l'ordre exact des lignes en entrée.
    original_index = df.index
    parts = [
        _per_hospital_outage_features(grp.copy())
        for _, grp in df.groupby("hospital", sort=False)
    ]
    df = pd.concat(parts).reindex(original_index)
    df = df.drop(columns=["_shifted_outage"])

    return df


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature_2m" not in df.columns:
        logger.info("Pas de colonnes météo — features météo ignorées.")
        return df

    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["wind_precipitation_interaction"] = df["wind_speed_10m"] * df["precipitation"]
    df["solar_available"] = (df["shortwave_radiation"] > 50).astype(int)
    df["heat_stress"] = (df["temperature_2m"] > 30).astype(int)
    df["storm_risk"] = ((df["wind_speed_10m"] > 40) | (df["precipitation"] > 10)).astype(int)

    return df


def add_advanced_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features météo avancées dérivées des variables Open-Meteo."""
    if "temperature_2m" not in df.columns:
        logger.info("Pas de colonnes météo — features météo avancées ignorées.")
        return df

    # Couverture nuageuse (si disponible via Open-Meteo étendu)
    if "cloud_cover" in df.columns:
        df["cloud_cover_pct"] = df["cloud_cover"]
    else:
        # Proxy : inverse du ratio rayonnement solaire / max théorique.
        # Restriction aux heures de jour : la nuit, shortwave_radiation=0
        # systématiquement → l'ancien proxy renvoyait 100% (ciel "couvert")
        # même par nuit claire. On garde une valeur neutre (50%) hors plage.
        max_solar = (
            df.groupby("hospital", sort=False)["shortwave_radiation"]
            .transform(lambda s: s.rolling(24 * 30, min_periods=24).max())
        )
        cloud_day = (
            (1 - df["shortwave_radiation"] / max_solar.replace(0, np.nan))
            .clip(0, 1) * 100
        )
        is_day = (df.get("hour", 12) >= 7) & (df.get("hour", 12) <= 19)
        df["cloud_cover_pct"] = cloud_day.where(is_day, 50.0).fillna(50.0)

    # Point de rosée (si disponible, sinon approximation Magnus)
    if "dew_point_2m" not in df.columns:
        t = df["temperature_2m"]
        rh = df["relative_humidity_2m"]
        a, b = 17.27, 237.7
        gamma = (a * t / (b + t)) + np.log(rh / 100 + 1e-10)
        df["dew_point_2m"] = (b * gamma / (a - gamma))

    # Visibilité (si disponible depuis Open-Meteo)
    if "visibility" in df.columns:
        df["visibility_m"] = df["visibility"]

    # Evapotranspiration (si disponible)
    if "et0_fao_evapotranspiration" in df.columns:
        df["evapotranspiration"] = df["et0_fao_evapotranspiration"]

    # Intensité de la pluie (proxy : précipitation × vitesse du vent)
    df["rain_intensity"] = df["precipitation"] * df["wind_speed_10m"]

    # Amplitude thermique sur 24h
    temp_by_hospital = df.groupby("hospital", sort=False)["temperature_2m"]
    df["thermal_amplitude_24h"] = (
        temp_by_hospital.transform(lambda s: s.rolling(24, min_periods=1).max())
        - temp_by_hospital.transform(lambda s: s.rolling(24, min_periods=1).min())
    )

    # Variation d'humidité sur 3h
    df["humidity_change_3h"] = df.groupby("hospital", sort=False)["relative_humidity_2m"].diff(3).fillna(0)

    # Variation de pression sur 3h (chute = front orageux)
    df["pressure_change_3h"] = df.groupby("hospital", sort=False)["surface_pressure"].diff(3).fillna(0)

    return df


def add_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées du signal GDELT (volumes et tonalité presse).

    - Moyenne glissante 7 jours sur chaque volume (lissage bruit quotidien).
    - Anomalie vs moyenne 30 jours (pic médiatique = signal de stress).
    - Ratio tonalité négative × volume (intensité perçue).
    """
    themes = [c.replace("gdelt_", "").replace("_vol", "")
              for c in df.columns if c.startswith("gdelt_") and c.endswith("_vol")]

    if not themes:
        logger.info("Pas de colonnes GDELT — features GDELT ignorées.")
        return df

    win_7d = 24 * 7
    win_30d = 24 * 30
    for theme in themes:
        vol = f"gdelt_{theme}_vol"
        tone = f"gdelt_{theme}_tone"
        df[f"gdelt_{theme}_vol_7d"] = (
            df.groupby("hospital", sort=False)[vol]
            .transform(lambda s: s.rolling(win_7d, min_periods=1).mean())
        )
        base_30d = (
            df.groupby("hospital", sort=False)[vol]
            .transform(lambda s: s.rolling(win_30d, min_periods=1).mean())
        )
        df[f"gdelt_{theme}_anomaly"] = (
            (df[vol] - base_30d) / base_30d.replace(0, np.nan)
        ).fillna(0).clip(-5, 5)
        if tone in df.columns:
            df[f"gdelt_{theme}_stress"] = (
                df[vol] * np.maximum(-df[tone], 0)
            ).clip(lower=0)

    logger.info("Features GDELT ajoutées (%d thèmes).", len(themes))
    return df


def add_noaa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées des événements NOAA Storm Events (USA)."""
    storm_cols = [c for c in df.columns if c.startswith("storm_")]
    if not storm_cols:
        logger.info("Pas de colonnes NOAA — features NOAA ignorées.")
        return df

    if "storm_active" in df.columns:
        by_hospital = df.groupby("hospital", sort=False)["storm_active"]
        df["storm_active_6h"] = by_hospital.transform(lambda s: s.rolling(6, min_periods=1).max())
        df["storm_active_24h"] = by_hospital.transform(lambda s: s.rolling(24, min_periods=1).max())
    if "storm_event_count" in df.columns:
        df["storm_count_24h"] = (
            df.groupby("hospital", sort=False)["storm_event_count"]
            .transform(lambda s: s.rolling(24, min_periods=1).sum())
        )
    logger.info("Features NOAA ajoutées.")
    return df


def add_air_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées de la qualité de l'air.

    Hypothèses physiques :
        - Une pollution forte ⇒ ventilation hospitalière sollicitée.
        - Un pic de poussière ⇒ usure des filtres + chute de production PV.
        - Un AQI > 50 sur 24 h ⇒ afflux respiratoire ⇒ surconsommation
          (oxygène, nébulisateurs, climatisation des chambres d'isolement).
    """
    air_cols = [c for c in df.columns if c.startswith("air_")]
    if not air_cols:
        logger.info("Pas de colonnes air_quality — features ignorées.")
        return df

    # Moyennes glissantes 6 h et 24 h pour PM2.5 et dust
    for col in ("air_pm2_5", "air_pm10", "air_dust", "air_european_aqi"):
        if col in df.columns:
            by_hospital = df.groupby("hospital", sort=False)[col]
            df[f"{col}_6h"] = by_hospital.transform(lambda s: s.rolling(6, min_periods=1).mean())
            df[f"{col}_24h"] = by_hospital.transform(lambda s: s.rolling(24, min_periods=1).mean())

    # Indicateur "pic de pollution" (AQI > 50 = mauvais selon échelle EU)
    if "air_european_aqi" in df.columns:
        df["air_pollution_high"] = (df["air_european_aqi"] > 50).astype(int)

    # Indicateur "tempête de poussière" (dust > 100 µg/m³ = sévère)
    if "air_dust" in df.columns:
        df["air_dust_storm"] = (df["air_dust"] > 100).astype(int)

    # Stress combiné chaleur + pollution (vague de chaleur polluée
    # ⇒ surcharge climatisation + admissions cardio-respiratoires)
    if "heat_stress" in df.columns and "air_pollution_high" in df.columns:
        df["air_heat_pollution_stress"] = (
            df["heat_stress"] * df["air_pollution_high"]
        ).astype(int)

    logger.info("Features qualité de l'air ajoutées.")
    return df


def add_earthquake_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées du stress sismique USGS."""
    eq_cols = [c for c in df.columns if c.startswith("eq_")]
    if not eq_cols:
        logger.info("Pas de colonnes USGS — features sismiques ignorées.")
        return df

    if "eq_stress" in df.columns:
        # Cumul du stress sur 24 h et 7 jours (réplique post-séisme)
        by_hospital = df.groupby("hospital", sort=False)["eq_stress"]
        df["eq_stress_24h"] = by_hospital.transform(lambda s: s.rolling(24, min_periods=1).sum())
        df["eq_stress_7d"] = by_hospital.transform(lambda s: s.rolling(168, min_periods=1).sum())

    if "eq_max_mag_24h" in df.columns:
        df["eq_major_event"] = (df["eq_max_mag_24h"] >= 5.0).astype(int)

    logger.info("Features USGS ajoutées.")
    return df


def add_gdacs_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées des alertes GDACS (catastrophes naturelles).

    Ces signaux capturent un stress *indirect* sur la consommation :
        - Catastrophe en cours ⇒ afflux de patients ⇒ équipements
          médicaux et stérilisation à plein régime.
        - Niveau d'alerte élevé ⇒ probabilité de coupure réseau accrue.
        - Persistance d'un événement ⇒ effet cumulatif sur la
          consommation hospitalière.
    """
    gdacs_cols = [c for c in df.columns if c.startswith("gdacs_")]
    if not gdacs_cols:
        logger.info("Pas de colonnes GDACS — features ignorées.")
        return df

    if "gdacs_alert_score" in df.columns:
        by_hospital = df.groupby("hospital", sort=False)["gdacs_alert_score"]
        df["gdacs_alert_24h"] = by_hospital.transform(lambda s: s.rolling(24, min_periods=1).max())
        df["gdacs_alert_7d_max"] = by_hospital.transform(lambda s: s.rolling(168, min_periods=1).max())

    if "gdacs_active_count" in df.columns:
        df["gdacs_disaster_active"] = (df["gdacs_active_count"] > 0).astype(int)

    # Catastrophe majeure (Orange ou Red ⇒ score ≥ 2)
    if "gdacs_alert_score" in df.columns:
        df["gdacs_major_disaster"] = (df["gdacs_alert_score"] >= 2.0).astype(int)

    # Combinaison stress climatique + catastrophe : double effet
    if "storm_risk" in df.columns and "gdacs_disaster_active" in df.columns:
        df["gdacs_storm_combo"] = (
            df["storm_risk"] * df["gdacs_disaster_active"]
        ).astype(int)

    logger.info("Features GDACS ajoutées.")
    return df


# ── API publique pour l'app Streamlit ──────────────────────────────
# Réutilise les blocs de FE multi-hôpitaux ci-dessus pour qu'un consommateur
# mono-hôpital (Streamlit) puisse appliquer EXACTEMENT le même calcul
# que le pipeline d'entraînement. Élimine la duplication historique
# `_apply_feature_engineering` dans app.py.


def apply_feature_engineering_single(
    df: pd.DataFrame,
    hospital_key: str = "__single__",
) -> pd.DataFrame:
    """Applique le feature engineering complet à un DataFrame *mono-hôpital*.

    Garantit la cohérence stricte avec `run()` (pipeline d'entraînement) :
      - temporel (hour, day_of_week, month, cyclique, fériés UG)
      - charge (rolling, diff, peak_ratio)
      - sources d'énergie (solar/generator/grid + recent_outages_*h)
      - météo de base + interactions
      - météo avancée (cloud_cover_pct fallback jour-only, dew_point Magnus…)
      - historique coupures (hours_since_last_outage, …, outage_trend_7d)

    Hypothèses sur `df` :
      - colonne `datetime` présente ;
      - éventuelles colonnes consommation (`total_load_kw`, `solar_pv_kw`,
        `base_load_kw`, `generators_kw`, `grid_available[_ratio]`) et météo
        (`temperature_2m`, etc.) ; manquantes ⇒ 0 par défaut.

    Renvoie une copie ; n'ajoute PAS `hospital` à la sortie.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # `hospital` temporaire pour réutiliser les groupby des fonctions
    # ci-dessus (coût négligeable pour 1 groupe).
    had_hospital = "hospital" in df.columns
    if not had_hospital:
        df["hospital"] = hospital_key

    # Pré-remplir colonnes manquantes : l'app peut partir d'un df ERIC/NYC
    # sans météo locale, ou d'un raw Lacor sans certaines colonnes
    # dérivées. On garantit la présence de tout ce qui sera lu.
    for mcol in [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_gusts_10m", "precipitation", "surface_pressure",
        "shortwave_radiation", "cape", "weathercode",
    ]:
        if mcol not in df.columns:
            df[mcol] = 0.0

    # `grid_availability_ratio` est calculé par preprocessing pour Lacor ;
    # quand l'app part d'un raw site mono-hôpital, on le reconstruit depuis
    # `grid_available` (Lacor) si présent.
    if "grid_available" in df.columns and "grid_availability_ratio" not in df.columns:
        df["grid_availability_ratio"] = df["grid_available"]

    df = add_temporal_features(df)
    df = add_load_features(df)
    df = add_energy_source_features(df)
    df = add_meteo_features(df)
    df = add_advanced_meteo_features(df)
    df = add_outage_history_features(df)

    # Defaults pour les colonnes "advanced" dont le pipeline complet
    # remplit certaines valeurs en se basant sur la météo brute, mais
    # qui peuvent rester NaN dans le contexte app.
    if "visibility_m" not in df.columns:
        df["visibility_m"] = 10_000.0
    if "evapotranspiration" not in df.columns:
        df["evapotranspiration"] = 0.0

    if not had_hospital:
        df = df.drop(columns=["hospital"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def run() -> None:
    t0 = perf_counter()
    df = load_table(PROCESSED_DIR / "hospital_merged.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["hospital", "datetime"]).reset_index(drop=True)

    step = perf_counter()
    df = add_temporal_features(df)
    logger.info("Timing add_temporal_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_load_features(df)
    logger.info("Timing add_load_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_energy_source_features(df)
    logger.info("Timing add_energy_source_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_meteo_features(df)
    logger.info("Timing add_meteo_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_advanced_meteo_features(df)
    logger.info("Timing add_advanced_meteo_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_outage_history_features(df)
    logger.info("Timing add_outage_history_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_gdelt_features(df)
    logger.info("Timing add_gdelt_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_noaa_features(df)
    logger.info("Timing add_noaa_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_air_quality_features(df)
    logger.info("Timing add_air_quality_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_earthquake_features(df)
    logger.info("Timing add_earthquake_features: %.2fs", perf_counter() - step)
    step = perf_counter()
    df = add_gdacs_features(df)
    logger.info("Timing add_gdacs_features: %.2fs", perf_counter() - step)

    # Remplacer les NaN restants
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    save_table(df, FEATURES_DIR / "features_dataset.csv")

    feature_cols = [c for c in df.columns if c not in ("datetime", "is_outage")]
    logger.info("Feature engineering terminé : %d features", len(feature_cols))
    logger.info("Colonnes : %s", list(df.columns))
    logger.info("Temps total feature engineering : %.2fs", perf_counter() - t0)


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
