"""
Interface Streamlit — Prédiction de coupures d'électricité en hôpitaux.
Deux modes : Analyse historique + Simulation manuelle.
"""

import sys
from datetime import datetime
from pathlib import Path

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config import FEATURES_DIR, MODELS_DIR

# ── Configuration ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prédiction de coupures",
    page_icon="⚡",
    layout="wide",
)

COLS_TO_DROP = [
    "datetime",
    "is_outage",
    "grid_availability_ratio",
    "generators_kw",
    "generator_active",
    "generator_ratio",
    "grid_availability_rolling_6h",
    "recent_outages_6h",
    "recent_outages_24h",
    "storm_risk",
    "cloud_cover",
    "visibility",
    "et0_fao_evapotranspiration",
]

HOSPITAL_DISPLAY = {
    "lacor_uganda": {
        "name": "Lacor Hospital",
        "location": "Gulu, Ouganda",
        "flag": "🇺🇬",
        "beds": 482,
        "type": "Hôpital général (PNL)",
        "who_reliability": 50.0,
        "lat": 2.77, "lon": 32.30,
        "avg_load_kw": 133, "max_load_kw": 235,
        "has_solar": True, "has_generator": True,
        "grid_stability": "faible",
    },
    # ── Hôpitaux africains (réseau temps réel via Electricity Maps) ──
    # data_source = africa_grid : profil Lacor mis à l'échelle par
    # avg_load_kw, météo Open-Meteo locale, et Electricity Maps live.
    "kenyatta_kenya": {
        "name": "Kenyatta National Hospital",
        "location": "Nairobi, Kenya",
        "flag": "🇰🇪",
        "beds": 1800,
        "type": "Hôpital de référence national",
        "who_reliability": 65.0,
        "lat": -1.30, "lon": 36.81,
        "avg_load_kw": 1900, "max_load_kw": 2700,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
    },
    "tikur_ethiopia": {
        "name": "Tikur Anbessa Specialized Hospital",
        "location": "Addis-Abeba, Éthiopie",
        "flag": "🇪🇹",
        "beds": 800,
        "type": "Hôpital universitaire",
        "who_reliability": 45.0,
        "lat": 9.01, "lon": 38.75,
        "avg_load_kw": 950, "max_load_kw": 1500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "faible",
        "data_source": "africa_grid",
    },
    "groote_schuur_sa": {
        "name": "Groote Schuur Hospital",
        "location": "Le Cap, Afrique du Sud",
        "flag": "🇿🇦",
        "beds": 893,
        "type": "Hôpital universitaire (UCT)",
        "who_reliability": 88.0,
        "lat": -33.94, "lon": 18.46,
        "avg_load_kw": 2400, "max_load_kw": 3300,
        "has_solar": True, "has_generator": True,
        "grid_stability": "instable (Eskom)",
        "data_source": "africa_grid",
    },
    "fann_senegal": {
        "name": "CHU de Fann",
        "location": "Dakar, Sénégal",
        "flag": "🇸🇳",
        "beds": 600,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 60.0,
        "lat": 14.69, "lon": -17.46,
        "avg_load_kw": 800, "max_load_kw": 1200,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
    },
    "parirenyatwa_zimbabwe": {
        "name": "Parirenyatwa Group of Hospitals",
        "location": "Harare, Zimbabwe",
        "flag": "🇿🇼",
        "beds": 1800,
        "type": "Hôpital universitaire de référence",
        "who_reliability": 35.0,
        "lat": -17.79, "lon": 31.05,
        "avg_load_kw": 1600, "max_load_kw": 2400,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très faible",
        "data_source": "africa_grid",
    },
    "muhimbili_tanzania": {
        "name": "Muhimbili National Hospital",
        "location": "Dar es Salaam, Tanzanie",
        "flag": "🇹🇿",
        "beds": 1500,
        "type": "Hôpital national de référence",
        "who_reliability": 58.0,
        "lat": -6.80, "lon": 39.27,
        "avg_load_kw": 1700, "max_load_kw": 2500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
    },
    "luth_nigeria": {
        "name": "Lagos University Teaching Hospital (LUTH)",
        "location": "Lagos, Nigeria",
        "flag": "🇳🇬",
        "beds": 760,
        "type": "Hôpital universitaire",
        "who_reliability": 30.0,
        "lat": 6.515, "lon": 3.358,
        "avg_load_kw": 1400, "max_load_kw": 2200,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très faible",
        "data_source": "africa_grid",
    },
    "korle_bu_ghana": {
        "name": "Korle Bu Teaching Hospital",
        "location": "Accra, Ghana",
        "flag": "🇬🇭",
        "beds": 2000,
        "type": "Hôpital universitaire",
        "who_reliability": 70.0,
        "lat": 5.535, "lon": -0.224,
        "avg_load_kw": 1800, "max_load_kw": 2700,
        "has_solar": False, "has_generator": True,
        "grid_stability": "faible",
        "data_source": "africa_grid",
    },
    "ibn_sina_morocco": {
        "name": "CHU Ibn Sina",
        "location": "Rabat, Maroc",
        "flag": "🇲🇦",
        "beds": 1100,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 92.0,
        "lat": 34.005, "lon": -6.834,
        "avg_load_kw": 1500, "max_load_kw": 2200,
        "has_solar": True, "has_generator": True,
        "grid_stability": "stable",
        "data_source": "africa_grid",
    },
    "kasr_alainy_egypt": {
        "name": "Kasr Al Ainy Hospital (Cairo Univ.)",
        "location": "Le Caire, Égypte",
        "flag": "🇪🇬",
        "beds": 5500,
        "type": "Hôpital universitaire (Cairo Univ.)",
        "who_reliability": 88.0,
        "lat": 30.029, "lon": 31.213,
        "avg_load_kw": 4500, "max_load_kw": 6500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
    },
    "chuk_rwanda": {
        "name": "CHU de Kigali (CHUK)",
        "location": "Kigali, Rwanda",
        "flag": "🇷🇼",
        "beds": 519,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 75.0,
        "lat": -1.954, "lon": 30.057,
        "avg_load_kw": 700, "max_load_kw": 1100,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
    },
    # ── Hôpitaux NHS (source : ERIC 2022-23) ────────────────────────
    "st_thomas_nhs": {
        "name": "St Thomas' Hospital",
        "location": "London, Angleterre",
        "flag": "🇬🇧",
        "beds": 840,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.4988, "lon": -0.1175,
        "avg_load_kw": 9361, "max_load_kw": 11863,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rj121",
        "floor_area_m2": 150_000,
        "annual_electricity_kwh": 82_000_000,
    },
    "addenbrookes_nhs": {
        "name": "Addenbrooke's Hospital",
        "location": "Cambridge, Angleterre",
        "flag": "🇬🇧",
        "beds": 1000,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 52.1753, "lon": 0.1405,
        "avg_load_kw": 8904, "max_load_kw": 11500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rgt01",
        "floor_area_m2": 160_000,
        "annual_electricity_kwh": 78_000_000,
    },
    "manchester_nhs": {
        "name": "Manchester Royal Infirmary",
        "location": "Manchester, Angleterre",
        "flag": "🇬🇧",
        "beds": 752,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 53.4617, "lon": -2.2260,
        "avg_load_kw": 6621, "max_load_kw": 8500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "r0a01",
        "floor_area_m2": 115_000,
        "annual_electricity_kwh": 58_000_000,
    },
    "kings_college_nhs": {
        "name": "King's College Hospital",
        "location": "London, Angleterre",
        "flag": "🇬🇧",
        "beds": 950,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.4685, "lon": -0.0940,
        "avg_load_kw": 8219, "max_load_kw": 10500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rxh01",
        "floor_area_m2": 140_000,
        "annual_electricity_kwh": 72_000_000,
    },
    "john_radcliffe_nhs": {
        "name": "John Radcliffe Hospital",
        "location": "Oxford, Angleterre",
        "flag": "🇬🇧",
        "beds": 832,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.7636, "lon": -1.2200,
        "avg_load_kw": 7078, "max_load_kw": 9000,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rth01",
        "floor_area_m2": 120_000,
        "annual_electricity_kwh": 62_000_000,
    },
    "guys_nhs": {
        "name": "Guy's Hospital",
        "location": "London, Angleterre",
        "flag": "🇬🇧",
        "beds": 400,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.5042, "lon": -0.0871,
        "avg_load_kw": 5479, "max_load_kw": 7000,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rj122",
        "floor_area_m2": 82_000,
        "annual_electricity_kwh": 48_000_000,
    },
    "leeds_general_nhs": {
        "name": "Leeds General Infirmary",
        "location": "Leeds, Angleterre",
        "flag": "🇬🇧",
        "beds": 700,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 53.8018, "lon": -1.5520,
        "avg_load_kw": 5936, "max_load_kw": 7600,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rr801",
        "floor_area_m2": 100_000,
        "annual_electricity_kwh": 52_000_000,
    },
    "birmingham_heartlands_nhs": {
        "name": "Birmingham Heartlands Hospital",
        "location": "Birmingham, Angleterre",
        "flag": "🇬🇧",
        "beds": 660,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 52.4636, "lon": -1.8220,
        "avg_load_kw": 5251, "max_load_kw": 6700,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rq301",
        "floor_area_m2": 95_000,
        "annual_electricity_kwh": 46_000_000,
    },
    "newcastle_rvi_nhs": {
        "name": "Royal Victoria Infirmary",
        "location": "Newcastle, Angleterre",
        "flag": "🇬🇧",
        "beds": 900,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 54.9802, "lon": -1.6196,
        "avg_load_kw": 7763, "max_load_kw": 9900,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "ra701",
        "floor_area_m2": 130_000,
        "annual_electricity_kwh": 68_000_000,
    },
    "royal_devon_nhs": {
        "name": "Royal Devon and Exeter Hospital",
        "location": "Exeter, Angleterre",
        "flag": "🇬🇧",
        "beds": 600,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 50.7157, "lon": -3.5060,
        "avg_load_kw": 4338, "max_load_kw": 5500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "ra401",
        "floor_area_m2": 80_000,
        "annual_electricity_kwh": 38_000_000,
    },
    "nyc_bellevue": {
        "name": "Bellevue Hospital Center",
        "location": "Manhattan, New York",
        "flag": "🇺🇸",
        "beds": 912,
        "type": "Public Acute (NYC H+H)",
        "who_reliability": 99.96,
        "lat": 40.7395, "lon": -73.9766,
        "avg_load_kw": 6046, "max_load_kw": 7800,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_bellevue",
        "floor_area_m2": 211_475,
        "annual_electricity_kwh": 52_960_248,
    },
    "nyc_nyu_tisch": {
        "name": "NYU Langone Tisch Hospital",
        "location": "Manhattan, New York",
        "flag": "🇺🇸",
        "beds": 844,
        "type": "Private Acute (NYU Langone)",
        "who_reliability": 99.96,
        "lat": 40.7426, "lon": -73.9744,
        "avg_load_kw": 5153, "max_load_kw": 6700,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_nyu_tisch",
        "floor_area_m2": 64_040,
        "annual_electricity_kwh": 45_139_152,
    },
    "nyc_nyp_brooklyn": {
        "name": "NewYork-Presbyterian Brooklyn Methodist",
        "location": "Brooklyn, New York",
        "flag": "🇺🇸",
        "beds": 1_001,
        "type": "Private Acute (NYP)",
        "who_reliability": 99.96,
        "lat": 40.6686, "lon": -73.9801,
        "avg_load_kw": 3698, "max_load_kw": 4800,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_nyp_brooklyn",
        "floor_area_m2": 126_587,
        "annual_electricity_kwh": 32_396_762,
    },
    "nyc_elmhurst": {
        "name": "Elmhurst Hospital Center",
        "location": "Queens, New York",
        "flag": "🇺🇸",
        "beds": 545,
        "type": "Public Acute (NYC H+H)",
        "who_reliability": 99.96,
        "lat": 40.7444, "lon": -73.8861,
        "avg_load_kw": 3483, "max_load_kw": 4500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_elmhurst",
        "floor_area_m2": 89_366,
        "annual_electricity_kwh": 30_507_199,
    },
    "nyc_lincoln": {
        "name": "Lincoln Medical Center",
        "location": "Bronx, New York",
        "flag": "🇺🇸",
        "beds": 362,
        "type": "Public Acute (NYC H+H)",
        "who_reliability": 99.96,
        "lat": 40.8177, "lon": -73.9242,
        "avg_load_kw": 3566, "max_load_kw": 4600,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_lincoln",
        "floor_area_m2": 110_874,
        "annual_electricity_kwh": 31_236_421,
    },
}

# N'afficher que les hôpitaux avec données de consommation réelles
# (pas de profil estimé/cloné).
REAL_DATA_SOURCES = {"eric", "nyc_ll84"}
REAL_HOSPITAL_KEYS = [
    k for k, v in HOSPITAL_DISPLAY.items()
    if k == "lacor_uganda" or v.get("data_source") in REAL_DATA_SOURCES
]
ALL_HOSPITAL_KEYS = list(HOSPITAL_DISPLAY.keys())

FEATURE_LABELS = {
    # ── Énergie & consommation ──
    "solar_ratio": "Part du solaire dans la charge",
    "solar_pv_kw": "Production solaire (kW)",
    "solar_available": "Solaire disponible",
    "total_load_kw": "Consommation totale (kW)",
    "sterilization_kw": "Stérilisation (kW)",
    "base_load_kw": "Charge de base (kW)",
    "base_load_ratio": "Ratio charge de base",
    "load_rolling_6h": "Charge moyenne (6h)",
    "load_rolling_24h": "Charge moyenne (24h)",
    "load_std_24h": "Variabilité de la charge (24h)",
    "load_diff_1h": "Variation de charge (1h)",
    "load_diff_24h": "Variation de charge (24h)",
    "load_pct_change_1h": "Variation relative (1h)",
    "peak_ratio": "Ratio pic / moyenne",
    # ── Historique coupures (réseau) ──
    "hours_since_last_outage": "Heures depuis dernière coupure",
    "last_outage_duration_h": "Durée dernière coupure (h)",
    "outage_frequency_7d": "Fréquence coupures (7 jours)",
    "avg_outage_duration_7d": "Durée moy. coupures (7 jours)",
    "outage_trend_7d": "Tendance coupures (7 jours)",
    # ── Temporel ──
    "hour": "Heure de la journée",
    "hour_sin": "Cycle horaire (sin)",
    "hour_cos": "Cycle horaire (cos)",
    "month": "Mois",
    "month_sin": "Cycle mensuel (sin)",
    "month_cos": "Cycle mensuel (cos)",
    "day_of_week": "Jour de la semaine",
    "is_weekend": "Week-end",
    "is_public_holiday": "Jour férié",
    # ── Météo ──
    "temperature_2m": "Température (°C)",
    "relative_humidity_2m": "Humidité relative (%)",
    "dew_point_2m": "Point de rosée (°C)",
    "wind_speed_10m": "Vitesse du vent (km/h)",
    "wind_gusts_10m": "Rafales de vent (km/h)",
    "precipitation": "Précipitations (mm)",
    "surface_pressure": "Pression (hPa)",
    "shortwave_radiation": "Rayonnement solaire (W/m²)",
    "cape": "Énergie convective (CAPE)",
    "weathercode": "Code météo",
    "temp_humidity_interaction": "Interaction temp × humidité",
    "wind_precipitation_interaction": "Interaction vent × pluie",
    "heat_stress": "Stress thermique",
    "cloud_cover_pct": "Couverture nuageuse (%)",
    "visibility_m": "Visibilité (m)",
    "evapotranspiration": "Évapotranspiration",
    "rain_intensity": "Intensité de la pluie",
    "thermal_amplitude_24h": "Amplitude thermique (24h)",
    "humidity_change_3h": "Variation humidité (3h)",
    "pressure_change_3h": "Variation pression (3h)",
    # ── Qualité de l'air (Open-Meteo Air Quality) ──
    "air_pm10": "PM10 (µg/m³)",
    "air_pm2_5": "PM2.5 (µg/m³)",
    "air_pm2_5_6h": "PM2.5 — moyenne 6h",
    "air_pm2_5_24h": "PM2.5 — moyenne 24h",
    "air_pm10_6h": "PM10 — moyenne 6h",
    "air_pm10_24h": "PM10 — moyenne 24h",
    "air_carbon_monoxide": "Monoxyde de carbone (µg/m³)",
    "air_nitrogen_dioxide": "NO₂ (µg/m³)",
    "air_sulphur_dioxide": "SO₂ (µg/m³)",
    "air_ozone": "Ozone (µg/m³)",
    "air_dust": "Poussière (µg/m³)",
    "air_dust_6h": "Poussière — moyenne 6h",
    "air_dust_24h": "Poussière — moyenne 24h",
    "air_uv_index": "Indice UV",
    "air_european_aqi": "AQI européen",
    "air_european_aqi_6h": "AQI européen — moyenne 6h",
    "air_european_aqi_24h": "AQI européen — moyenne 24h",
    "air_pollution_high": "Pollution élevée (AQI>50)",
    "air_dust_storm": "Tempête de poussière",
    "air_heat_pollution_stress": "Stress chaleur × pollution",
    # ── Signal médiatique GDELT ──
    "gdelt_power_vol": "GDELT : volume « coupure »",
    "gdelt_power_tone": "GDELT : tonalité « coupure »",
    "gdelt_power_vol_7d": "GDELT : volume coupure (7j)",
    "gdelt_power_anomaly": "GDELT : anomalie coupure",
    "gdelt_power_stress": "GDELT : stress médiatique coupure",
    "gdelt_weather_vol": "GDELT : volume météo extrême",
    "gdelt_weather_tone": "GDELT : tonalité météo extrême",
    "gdelt_weather_vol_7d": "GDELT : volume météo (7j)",
    "gdelt_weather_anomaly": "GDELT : anomalie météo",
    "gdelt_weather_stress": "GDELT : stress météo",
    "gdelt_health_vol": "GDELT : volume santé/urgences",
    "gdelt_health_tone": "GDELT : tonalité santé",
    "gdelt_health_vol_7d": "GDELT : volume santé (7j)",
    "gdelt_health_anomaly": "GDELT : anomalie santé",
    "gdelt_health_stress": "GDELT : stress santé",
    "gdelt_disaster_vol": "GDELT : volume catastrophes",
    "gdelt_disaster_tone": "GDELT : tonalité catastrophes",
    "gdelt_disaster_vol_7d": "GDELT : volume catastrophes (7j)",
    "gdelt_disaster_anomaly": "GDELT : anomalie catastrophes",
    "gdelt_disaster_stress": "GDELT : stress catastrophes",
    # ── Catastrophes GDACS ──
    "gdacs_active_count": "Catastrophes actives (GDACS)",
    "gdacs_alert_score": "Score d'alerte GDACS",
    "gdacs_alert_24h": "Alerte GDACS — max 24h",
    "gdacs_alert_7d_max": "Alerte GDACS — max 7j",
    "gdacs_disaster_active": "Catastrophe en cours",
    "gdacs_major_disaster": "Catastrophe majeure (Orange/Rouge)",
    "gdacs_storm_combo": "Tempête × catastrophe",
    "gdacs_is_flood": "Inondation active",
    "gdacs_is_cyclone": "Cyclone actif",
    "gdacs_is_earthquake": "Séisme actif",
    "gdacs_is_volcano": "Activité volcanique",
    "gdacs_is_drought": "Sécheresse en cours",
    "gdacs_is_wildfire": "Feu de forêt actif",
    "gdacs_is_tsunami": "Alerte tsunami",
    # ── Sismique USGS ──
    "eq_stress": "Stress sismique (instantané)",
    "eq_stress_24h": "Stress sismique cumulé 24h",
    "eq_stress_7d": "Stress sismique cumulé 7j",
    "eq_recent_count_24h": "Séismes 24h",
    "eq_max_mag_24h": "Magnitude max 24h",
    "eq_distance_min_km": "Distance min séisme (km)",
    "eq_major_event": "Séisme majeur (M≥5)",
    # ── NOAA Storm Events (USA) ──
    "storm_active": "Tempête NOAA active",
    "storm_event_count": "Nombre tempêtes (heure)",
    "storm_active_6h": "Tempête NOAA — 6h",
    "storm_active_24h": "Tempête NOAA — 24h",
    "storm_count_24h": "Tempêtes cumulées 24h",
    "storm_magnitude_max": "Magnitude tempête max",
    "storm_damage_property_usd": "Dommages matériels (USD)",
    "storm_injuries": "Blessés tempête",
    "storm_deaths": "Décès tempête",
    "storm_is_thunderstorm": "Orage actif",
    "storm_is_flood": "Inondation NOAA",
    "storm_is_wind": "Vent fort NOAA",
    "storm_is_heat": "Vague de chaleur NOAA",
    "storm_is_winter": "Tempête hivernale",
    "storm_is_dust": "Tempête de poussière NOAA",
    "storm_risk": "Risque orageux (météo)",
    # ── Contexte ──
    "grid_availability_ratio": "Disponibilité réseau",
    "grid_availability_rolling_6h": "Disponibilité réseau (6h)",
    "recent_outages_6h": "Coupures récentes (6h)",
    "recent_outages_24h": "Coupures récentes (24h)",
    "generators_kw": "Générateur (kW)",
    "generator_active": "Générateur actif",
    "generator_ratio": "Part du générateur",
    "cloud_cover": "Couverture nuageuse",
    "visibility": "Visibilité",
    "et0_fao_evapotranspiration": "Évapotranspiration FAO",
}


# Catégorisation des features pour affichage groupé / coloration.
# Chaque catégorie : (nom affiché, emoji, couleur hex, ordre)
FEATURE_CATEGORIES = {
    "energy":   {"label": "Énergie & consommation", "emoji": "🔋", "color": "#3498db"},
    "outage":   {"label": "Historique coupures",    "emoji": "⚡", "color": "#c0392b"},
    "time":     {"label": "Temporel",                "emoji": "🕐", "color": "#9b59b6"},
    "meteo":    {"label": "Météo",                   "emoji": "🌤️", "color": "#e67e22"},
    "air":      {"label": "Qualité de l'air",        "emoji": "🌫️", "color": "#1abc9c"},
    "gdelt":    {"label": "Signal médiatique (GDELT)", "emoji": "📰", "color": "#e84393"},
    "gdacs":    {"label": "Catastrophes (GDACS)",    "emoji": "🚨", "color": "#f39c12"},
    "usgs":     {"label": "Sismique (USGS)",         "emoji": "🌍", "color": "#8b6f47"},
    "noaa":     {"label": "Tempêtes NOAA (USA)",     "emoji": "🌩️", "color": "#34495e"},
    "other":    {"label": "Autre",                   "emoji": "▫️", "color": "#95a5a6"},
}


def get_feature_category(feat: str) -> str:
    """Retourne la clé de catégorie d'une feature à partir de son nom."""
    if feat.startswith("gdelt_"):
        return "gdelt"
    if feat.startswith("gdacs_"):
        return "gdacs"
    if feat.startswith("eq_"):
        return "usgs"
    if feat.startswith("storm_") and feat != "storm_risk":
        return "noaa"
    if feat.startswith("air_"):
        return "air"
    if feat in {
        "hours_since_last_outage", "last_outage_duration_h",
        "outage_frequency_7d", "avg_outage_duration_7d", "outage_trend_7d",
        "recent_outages_6h", "recent_outages_24h", "grid_availability_ratio",
        "grid_availability_rolling_6h",
    }:
        return "outage"
    if feat in {
        "hour", "hour_sin", "hour_cos", "month", "month_sin", "month_cos",
        "day_of_week", "is_weekend", "is_public_holiday",
    }:
        return "time"
    if feat in {
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_gusts_10m", "precipitation",
        "surface_pressure", "shortwave_radiation", "cape", "weathercode",
        "temp_humidity_interaction", "wind_precipitation_interaction",
        "heat_stress", "cloud_cover_pct", "cloud_cover", "visibility_m",
        "visibility", "evapotranspiration", "et0_fao_evapotranspiration",
        "rain_intensity", "thermal_amplitude_24h",
        "humidity_change_3h", "pressure_change_3h", "storm_risk",
        "solar_available",
    }:
        return "meteo"
    if feat in {
        "solar_ratio", "solar_pv_kw", "total_load_kw", "sterilization_kw",
        "base_load_kw", "base_load_ratio", "load_rolling_6h",
        "load_rolling_24h", "load_std_24h", "load_diff_1h", "load_diff_24h",
        "load_pct_change_1h", "peak_ratio", "generators_kw",
        "generator_active", "generator_ratio",
    }:
        return "energy"
    return "other"


def feature_label(feat: str) -> str:
    """Retourne le label humain d'une feature, fallback = nom brut."""
    return FEATURE_LABELS.get(feat, feat)


# Sources de données utilisées (affichage dans le bandeau "modèle")
DATA_SOURCES = [
    {"name": "Lacor Hospital — consommation 2022 (terrain)",
     "icon": "🏥", "type": "Hospitalier",
     "desc": "35 040 mesures à 15 min agrégées en horaire", "key": True},
    {"name": "ERIC NHS (UK)",
     "icon": "🇬🇧", "type": "Hospitalier",
     "desc": "10 hôpitaux NHS, consommation annuelle + profil horaire"},
    {"name": "NYC LL84 (USA)",
     "icon": "🇺🇸", "type": "Hospitalier",
     "desc": "5 hôpitaux NYC, consommation annuelle + profil horaire"},
    {"name": "Open-Meteo Archive",
     "icon": "🌦️", "type": "Météo historique",
     "desc": "13 variables horaires : température, vent, pluie, pression…"},
    {"name": "Open-Meteo Forecast",
     "icon": "🔮", "type": "Météo prévision",
     "desc": "7 jours de prévisions horaires pour chaque hôpital"},
    {"name": "Open-Meteo Air Quality",
     "icon": "🌫️", "type": "Pollution",
     "desc": "PM2.5, PM10, ozone, NO₂, dust, AQI"},
    {"name": "Electricity Maps API",
     "icon": "⚡", "type": "Réseau local temps réel",
     "desc": "Zone locale, charge réseau (MW), intensité carbone, mix"},
    {"name": "GDELT DOC 2.0",
     "icon": "📰", "type": "Signal médiatique",
     "desc": "4 thèmes : coupure, météo, santé, catastrophes"},
    {"name": "GDACS (UE/OCHA)",
     "icon": "🚨", "type": "Catastrophes",
     "desc": "Inondations, cyclones, séismes, feux, sécheresses"},
    {"name": "USGS Earthquake",
     "icon": "🌍", "type": "Sismique",
     "desc": "Séismes M≥3 dans un rayon de 500 km"},
    {"name": "NOAA Storm Events",
     "icon": "🌩️", "type": "Tempêtes (USA)",
     "desc": "Orages, tornades, vagues de chaleur, haboobs"},
]


# ── Chargement ───────────────────────────────────────────────────────

ERIC_DIR = ROOT / "data" / "raw" / "eric"


def _model_file_mtime() -> float:
    """Retourne le mtime du modèle pour invalider le cache quand le fichier change."""
    for p in [MODELS_DIR / "calibrated_rf.joblib", MODELS_DIR / "baseline_rf.joblib"]:
        if p.exists():
            return p.stat().st_mtime
    return 0.0


@st.cache_resource
def load_model(_mtime: float = 0.0):
    calibrated_path = MODELS_DIR / "calibrated_rf.joblib"
    baseline_path = MODELS_DIR / "baseline_rf.joblib"
    summary_path = MODELS_DIR / "training_summary.json"

    winner_name = "?"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                winner_name = json.load(f).get("winner", "?")
        except Exception:
            pass

    if calibrated_path.exists():
        try:
            model = joblib.load(calibrated_path)
            st.sidebar.success(f"Modèle : **{winner_name}** (calibré)")
            return model
        except Exception as e:
            st.sidebar.warning(f"Échec du modèle calibré : {e} — fallback sur le brut")

    if baseline_path.exists():
        try:
            model = joblib.load(baseline_path)
            st.sidebar.info(f"Modèle : **{winner_name}** (brut)")
            return model
        except Exception as e:
            st.error(f"**Erreur au chargement du modèle** : {e}")
            st.stop()

    st.error(
        "**Aucun modèle trouvé.**\n\n"
        "Exécutez d'abord le pipeline d'entraînement :\n"
        "```bash\npython run_pipeline.py\n```"
    )
    st.stop()


@st.cache_resource
def load_shap_explainer(_mtime: float = 0.0):
    explainer_path = MODELS_DIR / "shap_explainer.joblib"
    if not explainer_path.exists():
        return None
    try:
        return joblib.load(explainer_path)
    except Exception:
        return None


def _features_file_mtime() -> float:
    p = FEATURES_DIR / "features_dataset.csv"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_lacor_features(_mtime: float = 0.0):
    csv_path = FEATURES_DIR / "features_dataset.csv"
    if not csv_path.exists():
        st.error(
            f"**Données Lacor introuvables** : `{csv_path}`\n\n"
            "Exécutez d'abord le pipeline de preprocessing :\n"
            "```bash\npython run_pipeline.py\n```"
        )
        st.stop()
    try:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception as e:
        st.error(f"**Erreur au chargement des données Lacor** : {e}")
        st.stop()


UGANDA_PUBLIC_HOLIDAYS_2022 = [
    "2022-01-01", "2022-01-26", "2022-02-16", "2022-03-08",
    "2022-04-15", "2022-04-18", "2022-05-01", "2022-05-02",
    "2022-06-03", "2022-06-09", "2022-07-09", "2022-10-09",
    "2022-12-25", "2022-12-26",
]


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le feature engineering complet sur un DataFrame brut hospitalier."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    holidays = pd.to_datetime(UGANDA_PUBLIC_HOLIDAYS_2022)
    df["is_public_holiday"] = df["datetime"].dt.normalize().isin(holidays).astype(int)

    col = "total_load_kw"
    df["load_rolling_6h"] = df[col].rolling(6, min_periods=1).mean()
    df["load_rolling_24h"] = df[col].rolling(24, min_periods=1).mean()
    df["load_std_24h"] = df[col].rolling(24, min_periods=1).std().fillna(0)
    df["load_diff_1h"] = df[col].diff().fillna(0)
    df["load_diff_24h"] = df[col].diff(24).fillna(0)
    df["load_pct_change_1h"] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    df["peak_ratio"] = (df[col] / df["load_rolling_24h"]).fillna(1).replace([np.inf, -np.inf], 1)

    total = df["total_load_kw"].replace(0, np.nan)
    if "solar_pv_kw" in df.columns:
        df["solar_ratio"] = (df["solar_pv_kw"] / total).fillna(0).clip(0, 1)
    else:
        df["solar_ratio"] = 0.0
    if "base_load_kw" in df.columns:
        df["base_load_ratio"] = (df["base_load_kw"] / total).fillna(0).clip(0, 1)
    else:
        df["base_load_ratio"] = 0.0
    if "generators_kw" in df.columns:
        df["generator_active"] = (df["generators_kw"] > 1.0).astype(int)
        df["generator_ratio"] = (df["generators_kw"] / total).fillna(0).clip(0, 1)
    else:
        df["generator_active"] = 0
        df["generator_ratio"] = 0.0
    if "grid_available" in df.columns and "grid_availability_ratio" not in df.columns:
        df["grid_availability_ratio"] = df["grid_available"]
    if "grid_availability_ratio" in df.columns:
        df["grid_availability_rolling_6h"] = df["grid_availability_ratio"].rolling(6, min_periods=1).mean()
    else:
        df["grid_availability_rolling_6h"] = 1.0
    if "is_outage" in df.columns:
        df["recent_outages_6h"] = df["is_outage"].rolling(6, min_periods=1).sum()
        df["recent_outages_24h"] = df["is_outage"].rolling(24, min_periods=1).sum()
    else:
        df["recent_outages_6h"] = 0
        df["recent_outages_24h"] = 0

    # Colonnes météo de base (fallback à 0 si absentes)
    for mcol in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                  "wind_gusts_10m", "precipitation", "surface_pressure",
                  "shortwave_radiation", "cape", "weathercode"]:
        if mcol not in df.columns:
            df[mcol] = 0.0

    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["wind_precipitation_interaction"] = df["wind_speed_10m"] * df["precipitation"]
    df["solar_available"] = (df["shortwave_radiation"] > 50).astype(int)
    df["heat_stress"] = (df["temperature_2m"] > 30).astype(int)

    # ── Météo avancée ──
    if "cloud_cover" in df.columns:
        df["cloud_cover_pct"] = df["cloud_cover"]
    else:
        max_solar = df["shortwave_radiation"].rolling(24 * 30, min_periods=24).max()
        df["cloud_cover_pct"] = (
            (1 - df["shortwave_radiation"] / max_solar.replace(0, np.nan))
            .fillna(0).clip(0, 1) * 100
        )

    if "dew_point_2m" not in df.columns:
        t = df["temperature_2m"]
        rh = df["relative_humidity_2m"]
        a, b = 17.27, 237.7
        gamma = (a * t / (b + t)) + np.log(rh / 100 + 1e-10)
        df["dew_point_2m"] = (b * gamma / (a - gamma))

    if "visibility" in df.columns:
        df["visibility_m"] = df["visibility"]
    elif "visibility_m" not in df.columns:
        df["visibility_m"] = 10000.0

    if "et0_fao_evapotranspiration" in df.columns:
        df["evapotranspiration"] = df["et0_fao_evapotranspiration"]
    elif "evapotranspiration" not in df.columns:
        df["evapotranspiration"] = 0.0

    df["rain_intensity"] = df["precipitation"] * df["wind_speed_10m"]
    df["thermal_amplitude_24h"] = (
        df["temperature_2m"].rolling(24, min_periods=1).max()
        - df["temperature_2m"].rolling(24, min_periods=1).min()
    )
    df["humidity_change_3h"] = df["relative_humidity_2m"].diff(3).fillna(0)
    df["pressure_change_3h"] = df["surface_pressure"].diff(3).fillna(0)

    # ── Historique coupures ──
    if "is_outage" in df.columns:
        outage = df["is_outage"].astype(int).shift(1).fillna(0)
        groups = (outage != outage.shift(1)).cumsum()
        non_outage_mask = outage == 0
        df["hours_since_last_outage"] = non_outage_mask.groupby(groups).cumsum().fillna(0)

        outage_starts = (outage == 1) & (outage.shift(1) == 0)
        outage_ends = (outage == 0) & (outage.shift(1) == 1)
        durations = outage.groupby(outage_starts.cumsum()).transform("sum")
        df["last_outage_duration_h"] = durations.where(outage_ends).ffill().fillna(0)

        df["outage_frequency_7d"] = outage.rolling(168, min_periods=1).sum()

        outage_hours_7d = outage.rolling(168, min_periods=1).sum()
        outage_events_7d = (
            outage_starts.shift(1)
            .astype("boolean")
            .fillna(False)
            .astype(int)
            .rolling(168, min_periods=1)
            .sum()
        )
        df["avg_outage_duration_7d"] = (
            (outage_hours_7d / outage_events_7d.replace(0, np.nan)).fillna(0)
        )

        recent_7d = outage.rolling(168, min_periods=1).sum()
        prev_7d = outage.shift(168).rolling(168, min_periods=1).sum()
        df["outage_trend_7d"] = (
            (recent_7d / prev_7d.replace(0, np.nan)).fillna(1.0).clip(0, 10)
        )
    else:
        df["hours_since_last_outage"] = 168.0
        df["last_outage_duration_h"] = 0.0
        df["outage_frequency_7d"] = 0.0
        df["avg_outage_duration_7d"] = 0.0
        df["outage_trend_7d"] = 1.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def _forecast_file_mtime(hospital_key: str) -> float:
    p = ROOT / "data" / "raw" / f"meteo_forecast_{hospital_key}.csv"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_meteo_forecast(hospital_key: str, _mtime: float = 0.0) -> pd.DataFrame | None:
    """Charge les prévisions Open-Meteo pour un hôpital (fichier généré par
    `ingest_openmeteo_forecast.run()`)."""
    path = ROOT / "data" / "raw" / f"meteo_forecast_{hospital_key}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception:
        return None


def _match_similar_historical_row(
    df: pd.DataFrame,
    target_hour: int,
    target_month: int,
    target_dow: int,
    target_temp: float,
) -> pd.Series:
    """Retourne la ligne historique la plus proche des conditions visées
    (même heure, même mois, jour-type similaire, température proche)."""
    cand = df.copy()
    cand["_h"] = (cand["hour"] - target_hour).abs()
    cand["_m"] = (cand["month"] - target_month).abs()
    cand["_d"] = (cand["day_of_week"] - target_dow).abs()
    cand["_t"] = (cand["temperature_2m"] - target_temp).abs()
    cand["_s"] = cand["_h"] * 3 + cand["_m"] * 2 + cand["_d"] + cand["_t"] * 0.1
    return df.loc[cand["_s"].idxmin()]


def build_forecast_predictions(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    hospital_info: dict,
    feature_cols: list[str],
    model,
) -> pd.DataFrame:
    """Pour chaque heure future du CSV prévisions, construit une ligne de
    features (consommation empruntée à l'heure historique similaire, météo
    remplacée par les prévisions) et prédit la probabilité de coupure.

    Retourne un DataFrame avec colonnes : datetime, outage_probability,
    temperature_2m, precipitation, wind_speed_10m, shortwave_radiation,
    + colonnes utiles pour affichage.
    """
    predictions = []
    meteo_cols_forecast = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_gusts_10m", "precipitation",
        "surface_pressure", "shortwave_radiation", "cloud_cover",
        "visibility", "et0_fao_evapotranspiration", "cape", "weathercode",
    ]

    for _, row in forecast_df.iterrows():
        ts = row["datetime"]
        hour = ts.hour
        month = ts.month
        dow = ts.dayofweek
        temp = float(row.get("temperature_2m", 25.0))

        ref = _match_similar_historical_row(hist_df, hour, month, dow, temp)
        feat = ref[feature_cols].copy()

        for mcol in meteo_cols_forecast:
            if mcol in row.index and mcol in feat.index:
                feat[mcol] = float(row[mcol])

        if "temperature_2m" in feat and "relative_humidity_2m" in feat:
            feat["temp_humidity_interaction"] = feat["temperature_2m"] * feat["relative_humidity_2m"] / 100
        if "wind_speed_10m" in feat and "precipitation" in feat:
            feat["wind_precipitation_interaction"] = feat["wind_speed_10m"] * feat["precipitation"]
            feat["rain_intensity"] = feat["precipitation"] * feat["wind_speed_10m"]
        if "shortwave_radiation" in feat:
            feat["solar_available"] = 1 if feat["shortwave_radiation"] > 50 else 0
        if "temperature_2m" in feat:
            feat["heat_stress"] = 1 if feat["temperature_2m"] > 30 else 0
        if "cloud_cover" in feat and "cloud_cover_pct" in feat:
            feat["cloud_cover_pct"] = feat["cloud_cover"]
        if "visibility" in feat and "visibility_m" in feat:
            feat["visibility_m"] = feat["visibility"]
        if "et0_fao_evapotranspiration" in feat and "evapotranspiration" in feat:
            feat["evapotranspiration"] = feat["et0_fao_evapotranspiration"]

        feat["hour"] = hour
        feat["month"] = month
        feat["day_of_week"] = dow
        feat["is_weekend"] = 1 if dow >= 5 else 0
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feat["month_sin"] = np.sin(2 * np.pi * month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * month / 12)

        row_df = ensure_numeric_feature_frame(pd.DataFrame([feat]), feature_cols)
        proba = float(model.predict_proba(row_df)[0][1])
        proba_adj, _ = adjust_for_hospital_profile(proba, hospital_info)

        predictions.append({
            "datetime": ts,
            "outage_probability": proba_adj,
            "temperature_2m": float(row.get("temperature_2m", 0.0)),
            "precipitation": float(row.get("precipitation", 0.0)),
            "wind_speed_10m": float(row.get("wind_speed_10m", 0.0)),
            "shortwave_radiation": float(row.get("shortwave_radiation", 0.0)),
        })

    return pd.DataFrame(predictions)


@st.cache_data
def load_eric_features(eric_code: str, hospital_info: dict) -> pd.DataFrame | None:
    csv_path = ERIC_DIR / f"eric_{eric_code}_hourly.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Impossible de lire les données ERIC `{eric_code}` : {e}")
        return None

    # Récupérer la météo : on cherche d'abord un fichier propre à
    # l'hôpital (vraie météo locale Open-Meteo), sinon fallback sur Lacor
    # avec offset de température lié à la latitude.
    hospital_key = next(
        (k for k, v in HOSPITAL_DISPLAY.items()
         if v.get("eric_code") == eric_code),
        None,
    )
    local_meteo = (
        ROOT / "data" / "raw" / f"meteo_{hospital_key}.csv"
        if hospital_key else None
    )

    df["datetime"] = pd.to_datetime(df["datetime"])

    if local_meteo and local_meteo.exists():
        meteo = pd.read_csv(local_meteo)
        meteo["datetime"] = pd.to_datetime(meteo["datetime"])
        meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
        meteo = (
            meteo.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
        )
        # Alignement robuste par timestamp (évite les erreurs si longueurs différentes,
        # ex. météo partielle 744 lignes vs ERIC 8760 lignes).
        df = df.merge(meteo[["datetime", *meteo_cols]], on="datetime", how="left")
    else:
        lacor_meteo = ROOT / "data" / "raw" / "meteo_lacor_uganda.csv"
        if lacor_meteo.exists():
            meteo = pd.read_csv(lacor_meteo)
            meteo["datetime"] = pd.to_datetime(meteo["datetime"])
            lat = hospital_info.get("lat", 51.5)
            temp_offset = (51.5 - lat) * 0.15
            meteo["temperature_2m"] = meteo["temperature_2m"] - temp_offset
            meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
            meteo = (
                meteo.sort_values("datetime")
                .drop_duplicates(subset=["datetime"], keep="last")
            )
            df = df.merge(meteo[["datetime", *meteo_cols]], on="datetime", how="left")

    df = _apply_feature_engineering(df)
    return df


@st.cache_data
def load_africa_grid_features(hospital_key: str, hospital_info: dict) -> pd.DataFrame | None:
    """Charge un profil hospitalier africain en clonant Lacor puis en
    re-scaling sur `avg_load_kw`, en injectant la météo Open-Meteo locale
    et le signal Electricity Maps (charge réseau temps réel) propres au
    pays. Les autres signaux externes (GDELT/GDACS/USGS/AirQuality) sont
    laissés au loader appelant qui les neutralise.

    Justification : on n'a pas de relevé interne de consommation pour ces
    hôpitaux. Le profil temporel reste celui de Lacor (variations
    horaires/journalières/saisonnières d'un hôpital régional africain),
    mais l'amplitude est mise à l'échelle de l'établissement et le contexte
    météo + réseau local est injecté pour que la prédiction soit cohérente.
    """
    base = load_lacor_features(_features_file_mtime())
    if base is None or base.empty:
        return None
    df = base.copy()
    # Rebase temporel en "quasi temps réel" :
    # on conserve le profil de consommation de Lacor mais on remplace
    # les timestamps par une grille horaire se terminant maintenant.
    # Ainsi les graphes/analyses ne restent pas bloqués en 2022.
    now_h = pd.Timestamp.utcnow().floor("h").tz_localize(None)
    df["datetime"] = pd.date_range(end=now_h, periods=len(df), freq="h")

    target_avg = float(hospital_info.get("avg_load_kw", 133))
    lacor_avg = 133.0
    scale = target_avg / lacor_avg if lacor_avg > 0 else 1.0

    consumption_cols = [
        "total_load_kw", "solar_pv_kw", "base_load_kw",
        "generators_kw", "sterilization_kw",
    ]
    for col in consumption_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * scale

    if not hospital_info.get("has_solar") and "solar_pv_kw" in df.columns:
        df["solar_pv_kw"] = 0.0
    if not hospital_info.get("has_generator"):
        for col in ("generators_kw",):
            if col in df.columns:
                df[col] = 0.0

    cols_to_zero = [
        c for c in df.columns
        if any(c.startswith(p) for p in EXTERNAL_SIGNAL_PREFIXES)
    ]
    for c in cols_to_zero:
        df[c] = 0

    local_meteo = ROOT / "data" / "raw" / f"meteo_{hospital_key}.csv"
    if local_meteo.exists():
        try:
            meteo = pd.read_csv(local_meteo)
            meteo["datetime"] = pd.to_datetime(meteo["datetime"])
            meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
            n = min(len(df), len(meteo))
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in meteo_cols:
                if col in meteo.columns:
                    df.loc[df.index[:n], col] = meteo[col].values[:n]
        except Exception as e:
            st.warning(f"Météo locale {hospital_key} illisible : {e}")

    em_path = ROOT / "data" / "raw" / f"electricitymaps_{hospital_key}.csv"
    if em_path.exists():
        try:
            em = pd.read_csv(em_path)
            em["datetime"] = pd.to_datetime(em["datetime"], errors="coerce")
            em = em.dropna(subset=["datetime"]).sort_values("datetime")
            if not em.empty:
                df["datetime"] = pd.to_datetime(df["datetime"])
                em_cols = [c for c in em.columns if c.startswith("em_")]
                merged = pd.merge_asof(
                    df.sort_values("datetime"),
                    em[["datetime"] + em_cols].sort_values("datetime"),
                    on="datetime",
                    direction="nearest",
                    tolerance=pd.Timedelta("24h"),
                    suffixes=("", "_local"),
                )
                for col in em_cols:
                    local_col = f"{col}_local"
                    if local_col in merged.columns:
                        merged[col] = np.where(
                            merged[local_col].notna(),
                            merged[local_col],
                            merged[col],
                        )
                        merged = merged.drop(columns=[local_col])
                df = merged
        except Exception as e:
            st.warning(f"Electricity Maps {hospital_key} illisible : {e}")

    df = _apply_feature_engineering(df)
    return df


@st.cache_data
def load_nyc_features(nyc_code: str, hospital_info: dict) -> pd.DataFrame | None:
    """Charge les profils horaires NYC LL84 + météo locale Open-Meteo."""
    nyc_dir = ROOT / "data" / "raw" / "nyc_ll84"
    csv_path = nyc_dir / f"{nyc_code}_hourly.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Impossible de lire les données NYC LL84 `{nyc_code}` : {e}")
        return None

    hospital_key = next(
        (k for k, v in HOSPITAL_DISPLAY.items()
         if v.get("nyc_code") == nyc_code),
        None,
    )
    local_meteo = (
        ROOT / "data" / "raw" / f"meteo_{hospital_key}.csv"
        if hospital_key else None
    )
    df["datetime"] = pd.to_datetime(df["datetime"])

    if local_meteo and local_meteo.exists():
        meteo = pd.read_csv(local_meteo)
        meteo["datetime"] = pd.to_datetime(meteo["datetime"])
        meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
        meteo = (
            meteo.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
        )
        df = df.merge(meteo[["datetime", *meteo_cols]], on="datetime", how="left")

    df = _apply_feature_engineering(df)
    return df


# ── Détection des sources de données disponibles par hôpital ───────
# Permet d'afficher dans l'UI exactement de quelles sources chaque hôpital
# bénéficie. Les fichiers sont regardés sur disque, donc ça reflète l'état
# réel du projet.

_RAW_DIR = ROOT / "data" / "raw"


def detect_hospital_data_sources(hospital_key: str, hospital_info: dict) -> list[dict]:
    """Renvoie la liste des sources de données réellement disponibles pour
    cet hôpital, sous forme de dicts {label, emoji, color, status, detail}.
    `status` ∈ {"primary", "available", "synthetic", "missing"}.
    """
    sources: list[dict] = []

    # ── 1. Consommation électrique ─────────────────────────────────
    if hospital_info.get("data_source") == "eric":
        eric_code = hospital_info.get("eric_code", "")
        eric_path = ROOT / "data" / "raw" / "eric" / f"eric_{eric_code}_hourly.csv"
        if eric_path.exists():
            sources.append({
                "label": "Consommation NHS ERIC",
                "emoji": "📊", "color": "#2ecc71", "status": "primary",
                "detail": f"Données réelles · {eric_path.name}",
            })
        else:
            sources.append({
                "label": "Consommation NHS ERIC",
                "emoji": "📊", "color": "#e74c3c", "status": "missing",
                "detail": "Fichier introuvable",
            })
    elif hospital_info.get("data_source") == "nyc_ll84":
        nyc_code = hospital_info.get("nyc_code", "")
        nyc_path = ROOT / "data" / "raw" / "nyc_ll84" / f"{nyc_code}_hourly.csv"
        if nyc_path.exists():
            sources.append({
                "label": "Consommation NYC Local Law 84",
                "emoji": "📊", "color": "#2ecc71", "status": "primary",
                "detail": f"data.cityofnewyork.us · {nyc_path.name}",
            })
        else:
            sources.append({
                "label": "Consommation NYC Local Law 84",
                "emoji": "📊", "color": "#e74c3c", "status": "missing",
                "detail": "Fichier introuvable",
            })
    elif hospital_key == "lacor_uganda":
        if (_RAW_DIR / "lacor_clean.csv").exists():
            sources.append({
                "label": "Consommation Lacor (terrain)",
                "emoji": "📊", "color": "#2ecc71", "status": "primary",
                "detail": "Relevés horaires Hôpital Lacor 2022",
            })
    else:
        sources.append({
            "label": "Consommation (profil cloné Lacor + scaling)",
            "emoji": "📊", "color": "#f39c12", "status": "synthetic",
            "detail": f"Profil Lacor re-mis à l'échelle ({hospital_info.get('avg_load_kw', '?')} kW)",
        })

    # ── 2. Météo ──────────────────────────────────────────────────
    meteo_path = _RAW_DIR / f"meteo_{hospital_key}.csv"
    forecast_path = _RAW_DIR / f"meteo_forecast_{hospital_key}.csv"
    if meteo_path.exists():
        sources.append({
            "label": "Météo Open-Meteo (historique)",
            "emoji": "🌤️", "color": "#2ecc71", "status": "primary",
            "detail": "Historique horaire 2022 (lat/lon hôpital)",
        })
    else:
        sources.append({
            "label": "Météo extrapolée Lacor (offset latitude)",
            "emoji": "🌤️", "color": "#f39c12", "status": "synthetic",
            "detail": "Compromis : météo Lacor avec correction de température",
        })
    if forecast_path.exists():
        sources.append({
            "label": "Météo Open-Meteo (prévisions)",
            "emoji": "🔮", "color": "#3498db", "status": "available",
            "detail": "Prévisions 7 jours pour mode anticipation",
        })

    # ── 3. Qualité de l'air ───────────────────────────────────────
    if (_RAW_DIR / f"air_quality_{hospital_key}.csv").exists():
        sources.append({
            "label": "Qualité de l'air Open-Meteo",
            "emoji": "🌫️", "color": "#1abc9c", "status": "available",
            "detail": "PM2.5, PM10, ozone, CO, NO₂, dust, UV",
        })

    # ── 4. Electricity Maps (réseau local) ───────────────────────
    em_path = _RAW_DIR / f"electricitymaps_{hospital_key}.csv"
    if em_path.exists():
        sources.append({
            "label": "Electricity Maps (réseau local)",
            "emoji": "⚡", "color": "#f1c40f", "status": "available",
            "detail": "Zone locale, charge réseau, intensité carbone, mix",
        })
    else:
        sources.append({
            "label": "Electricity Maps (réseau local)",
            "emoji": "⚡", "color": "#e74c3c", "status": "missing",
            "detail": "Fichier introuvable (lancer ingest_electricitymaps)",
        })

    # ── 5. Sismique USGS ──────────────────────────────────────────
    if (_RAW_DIR / f"usgs_earthquake_{hospital_key}.csv").exists():
        sources.append({
            "label": "Séismes USGS",
            "emoji": "🌍", "color": "#8b6f47", "status": "available",
            "detail": "Magnitude ≥ 3.0 dans un rayon de 500 km",
        })

    # ── 6. GDACS ──────────────────────────────────────────────────
    if (_RAW_DIR / f"gdacs_{hospital_key}.csv").exists():
        sources.append({
            "label": "Catastrophes GDACS (JRC/OCHA)",
            "emoji": "🚨", "color": "#e67e22", "status": "available",
            "detail": "Inondations, cyclones, séismes, sécheresses, feux",
        })

    # ── 7. GDELT (signal médiatique) ──────────────────────────────
    if (_RAW_DIR / f"gdelt_{hospital_key}.csv").exists():
        sources.append({
            "label": "Signal médiatique GDELT 2.0",
            "emoji": "📰", "color": "#e84393", "status": "available",
            "detail": "Volume / tonalité : énergie, météo, santé, désastres",
        })

    # ── 8. NOAA Storm Events (USA) ────────────────────────────────
    if hospital_key == "phoenix_usa" and (
        _RAW_DIR / "noaa_storm" / "storm_events_details_2022.csv"
    ).exists():
        sources.append({
            "label": "NOAA Storm Events (USA)",
            "emoji": "🌩️", "color": "#34495e", "status": "available",
            "detail": "Tempêtes, tornades, vagues de chaleur (Arizona)",
        })

    return sources


def render_data_sources_badges(sources: list[dict]) -> str:
    """Rend une grille de badges HTML pour les sources d'un hôpital."""
    status_label = {
        "primary":   ("Donnée primaire",    "#2ecc71"),
        "available": ("Disponible",          "#3498db"),
        "synthetic": ("Synthétique/cloné",  "#f39c12"),
        "missing":   ("Manquant",            "#e74c3c"),
    }
    items = []
    for s in sources:
        st_lbl, st_col = status_label.get(s["status"], ("?", "#95a5a6"))
        items.append(
            f"""
            <div style='border:1px solid #e0e0e0;border-left:4px solid {s["color"]};
                        border-radius:8px;padding:8px 12px;background:white;
                        display:flex;flex-direction:column;gap:2px'>
                <div style='display:flex;align-items:center;gap:8px;
                            font-size:13px;font-weight:600;color:#2c3e50'>
                    <span style='font-size:16px'>{s["emoji"]}</span>
                    <span>{s["label"]}</span>
                </div>
                <div style='font-size:11px;color:#666;margin-left:24px'>
                    {s["detail"]}
                </div>
                <div style='margin-left:24px'>
                    <span style='font-size:10px;font-weight:700;
                                 color:{st_col};text-transform:uppercase;
                                 letter-spacing:0.5px'>● {st_lbl}</span>
                </div>
            </div>
            """
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));"
        "gap:8px;margin-top:10px'>"
        + "".join(items)
        + "</div>"
    )


# ── Préfixes des features dérivées de signaux externes ─────────────
# Ces colonnes sont calculées à partir de fichiers raw propres à un site
# (gdelt_<hospital>.csv, gdacs_<hospital>.csv, etc.). Quand on clone le
# dataset Lacor pour un autre hôpital, ces colonnes contiennent encore les
# valeurs de l'Ouganda → il faut les neutraliser pour éviter de prédire
# Phoenix avec les épidémies/séismes/pollution de Lacor.
EXTERNAL_SIGNAL_PREFIXES = (
    "gdelt_", "gdacs_", "eq_", "air_",
    "em_",
    "noaa_", "storm_",
)


def _neutralize_external_signals(df: pd.DataFrame, hospital_key: str) -> pd.DataFrame:
    """Pour un hôpital ≠ lacor_uganda, met à 0 toutes les colonnes dérivées
    de signaux externes site-spécifiques (médias GDELT, catastrophes GDACS,
    sismique USGS, qualité de l'air, tempêtes NOAA). Ces signaux n'ont de
    sens que pour le site dont ils proviennent.

    Sans ce nettoyage, l'inférence peut utiliser des signaux non disponibles
    (ou provenant d'un autre site), ce qui dégrade la robustesse.
    """
    if hospital_key == "lacor_uganda":
        return df
    df = df.copy()
    cols_to_zero = [
        c for c in df.columns
        if any(c.startswith(p) for p in EXTERNAL_SIGNAL_PREFIXES)
    ]
    for c in cols_to_zero:
        df[c] = 0
    return df


@st.cache_data
def load_hospital_data(hospital_key: str, hospital_info: dict) -> pd.DataFrame:
    """Charge les données de l'hôpital sélectionné.

    Sources supportées :
      - lacor_uganda : relevés terrain horaires 2022
      - *_nhs        : données NHS ERIC désagrégées en horaire
      - nyc_*        : données NYC LL84 désagrégées en horaire
      - africa_grid  : profil estimé à partir d'un profil de référence
    """
    if hospital_info.get("data_source") == "eric":
        eric_code = hospital_info["eric_code"]
        eric_df = load_eric_features(eric_code, hospital_info)
        if eric_df is not None:
            return _neutralize_external_signals(eric_df, hospital_key)
        st.error(
            f"**Données ERIC introuvables** pour `{eric_code}`. "
            f"Vérifiez `data/raw/eric/eric_{eric_code}_hourly.csv`."
        )
        st.stop()

    if hospital_info.get("data_source") == "nyc_ll84":
        nyc_code = hospital_info["nyc_code"]
        nyc_df = load_nyc_features(nyc_code, hospital_info)
        if nyc_df is not None:
            return _neutralize_external_signals(nyc_df, hospital_key)
        st.error(
            f"**Données NYC LL84 introuvables** pour `{nyc_code}`. "
            f"Vérifiez `data/raw/nyc_ll84/{nyc_code}_hourly.csv`."
        )
        st.stop()

    if hospital_info.get("data_source") == "africa_grid":
        africa_df = load_africa_grid_features(hospital_key, hospital_info)
        if africa_df is not None:
            return africa_df
        st.error(
            f"**Profil africain introuvable** pour `{hospital_key}`. "
            "Vérifiez que `data/features/features_dataset.csv` existe."
        )
        st.stop()

    if hospital_key == "lacor_uganda":
        return load_lacor_features(_features_file_mtime())

    st.error(
        f"Hôpital `{hospital_key}` non supporté : aucune source de "
        "consommation réelle disponible."
    )
    st.stop()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = [c for c in COLS_TO_DROP if c in df.columns]
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


# ── Fonctions utilitaires ────────────────────────────────────────────

def ensure_numeric_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Aligne et convertit les features en numérique pour l'inférence."""
    out = frame.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
        series = out[col]
        if not pd.api.types.is_numeric_dtype(series):
            as_num = pd.to_numeric(series, errors="coerce")
            if as_num.notna().any():
                out[col] = as_num
            else:
                out[col] = pd.factorize(series.fillna("NA").astype(str))[0].astype(float)
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return out[feature_cols]

def risk_display(proba: float):
    if proba > 0.7:
        return "ÉLEVÉ", "#e74c3c", "🔴"
    elif proba > 0.4:
        return "MOYEN", "#f39c12", "🟠"
    else:
        return "FAIBLE", "#2ecc71", "🟢"


def _extract_feature_importances(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "estimators_"):
        return np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            return base.feature_importances_
    fi_path = MODELS_DIR / "feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        return fi_df["importance"].values
    return None


def get_top_factors(model, feature_cols: list[str], values: pd.Series, top_n: int = 5):
    imp_arr = _extract_feature_importances(model)
    if imp_arr is not None and len(imp_arr) == len(feature_cols):
        importances = pd.Series(imp_arr, index=feature_cols)
    else:
        fi_path = MODELS_DIR / "feature_importance.csv"
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            importances = pd.Series(fi_df["importance"].values, index=fi_df["feature"].values)
            importances = importances.reindex(feature_cols, fill_value=0.0)
        else:
            importances = pd.Series(1.0 / len(feature_cols), index=feature_cols)

    importances = importances.sort_values(ascending=False).head(top_n)

    factors = []
    for feat, imp in importances.items():
        factors.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "importance": imp,
            "value": values[feat] if feat in values.index else 0,
        })
    return factors


@st.cache_data
def load_global_shap_importance() -> pd.DataFrame | None:
    """Charge l'importance SHAP moyenne par feature (entraînement)."""
    p = MODELS_DIR / "shap_feature_importance.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "feature" not in df.columns:
            df.columns = ["feature", "mean_abs_shap"]
        else:
            value_col = [c for c in df.columns if c != "feature"][0]
            df = df.rename(columns={value_col: "mean_abs_shap"})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["category"] = df["feature"].apply(get_feature_category)
        df["label"] = df["feature"].apply(feature_label)
        return df
    except Exception:
        return None


def show_top_factors_panel(top_n: int = 12) -> None:
    """Affiche le top N facteurs globaux du modèle, par catégorie & couleur."""
    shap_df = load_global_shap_importance()
    if shap_df is None or shap_df.empty:
        st.info("Aucune importance SHAP globale disponible (relancez l'entraînement).")
        return

    top = shap_df.head(top_n)
    max_val = float(top["mean_abs_shap"].max())

    fig = go.Figure()
    for _, row in top.iloc[::-1].iterrows():
        cat = FEATURE_CATEGORIES.get(row["category"], FEATURE_CATEGORIES["other"])
        fig.add_trace(go.Bar(
            x=[row["mean_abs_shap"]],
            y=[f"{cat['emoji']}  {row['label']}"],
            orientation="h",
            marker_color=cat["color"],
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Catégorie : {cat['label']}<br>"
                f"SHAP |moyen| : {row['mean_abs_shap']:.4f}<extra></extra>"
            ),
            showlegend=False,
            text=[f"{row['mean_abs_shap']:.3f}"],
            textposition="outside",
        ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} facteurs globaux du modèle (importance SHAP)",
                   font=dict(size=14)),
        xaxis=dict(title="Impact moyen sur la prédiction", range=[0, max_val * 1.15]),
        height=max(360, top_n * 30),
        margin=dict(l=260, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def show_category_breakdown() -> None:
    """Décompose l'importance globale par catégorie de feature."""
    shap_df = load_global_shap_importance()
    if shap_df is None or shap_df.empty:
        return
    by_cat = shap_df.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=True)
    by_cat = by_cat[by_cat > 0]
    labels = [
        f"{FEATURE_CATEGORIES[c]['emoji']}  {FEATURE_CATEGORIES[c]['label']}"
        for c in by_cat.index
    ]
    colors = [FEATURE_CATEGORIES[c]["color"] for c in by_cat.index]
    fig = go.Figure(go.Bar(
        x=by_cat.values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in by_cat.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP cumulé : %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Importance cumulée par catégorie de données", font=dict(size=14)),
        xaxis=dict(title="Somme des |SHAP| par catégorie"),
        height=320,
        margin=dict(l=220, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def show_data_sources_panel() -> None:
    """Affiche les sources de données sous forme de cartes."""
    cols = st.columns(2)
    for i, src in enumerate(DATA_SOURCES):
        with cols[i % 2]:
            star = " ⭐" if src.get("key") else ""
            st.markdown(
                f"<div style='border:1px solid #e0e0e0;border-radius:8px;"
                f"padding:10px 14px;margin-bottom:8px;background:#fafafa'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;gap:8px'>"
                f"<b style='font-size:14px'>{src['icon']}  {src['name']}{star}</b>"
                f"<span style='background:#34495e22;color:#34495e;"
                f"padding:2px 8px;border-radius:10px;font-size:10px;"
                f"font-weight:600'>{src['type']}</span>"
                f"</div>"
                f"<div style='color:#666;font-size:12px;margin-top:4px'>"
                f"{src['desc']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


@st.cache_data
def load_electricitymaps_snapshot(hospital_key: str) -> pd.DataFrame | None:
    """Charge le CSV Electricity Maps d'un hôpital (si disponible)."""
    path = ROOT / "data" / "raw" / f"electricitymaps_{hospital_key}.csv"
    if not path.exists():
        return None
    try:
        em = pd.read_csv(path)
    except Exception:
        return None
    if em.empty or "datetime" not in em.columns:
        return None
    em["datetime"] = pd.to_datetime(em["datetime"], errors="coerce")
    em = em.dropna(subset=["datetime"]).sort_values("datetime")
    if em.empty:
        return None
    return em


def show_risk_result(proba: float, hours_away: float, duration: float):
    """Bloc de résultat de risque (carte mise en avant, réutilisée partout)."""
    risk_level, risk_color, risk_icon = risk_display(proba)
    pct = int(proba * 100)
    when_str = f"{hours_away:.0f} h" if hours_away >= 1 else "< 1 h"

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{risk_color}12,{risk_color}22);
                    border:1px solid {risk_color}55;
                    border-left:6px solid {risk_color};
                    border-radius:12px;padding:20px 26px;margin:8px 0 20px 0'>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        flex-wrap:wrap;gap:16px'>
                <div>
                    <div style='font-size:12px;color:rgba(120,120,120,0.95);
                                text-transform:uppercase;letter-spacing:1.5px'>
                        Synthèse du risque
                    </div>
                    <div style='font-size:36px;font-weight:800;color:{risk_color};
                                line-height:1.1;margin-top:4px'>
                        {risk_icon} {risk_level}
                    </div>
                </div>
                <div style='display:flex;gap:32px;flex-wrap:wrap'>
                    <div>
                        <div style='font-size:11px;color:rgba(120,120,120,0.95);text-transform:uppercase;
                                    letter-spacing:1.2px'>Probabilité</div>
                        <div style='font-size:32px;font-weight:700;color:{risk_color}'>
                            {pct}%
                        </div>
                    </div>
                    <div>
                        <div style='font-size:11px;color:rgba(120,120,120,0.95);text-transform:uppercase;
                                    letter-spacing:1.2px'>Délai estimé</div>
                        <div style='font-size:32px;font-weight:700;color:var(--text-color, #222)'>
                            {when_str}
                        </div>
                    </div>
                    <div>
                        <div style='font-size:11px;color:rgba(120,120,120,0.95);text-transform:uppercase;
                                    letter-spacing:1.2px'>Durée probable</div>
                        <div style='font-size:32px;font-weight:700;color:var(--text-color, #222)'>
                            {duration} h
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_step(title: str, detail: str = "") -> None:
    text = f"**{title}**"
    if detail:
        text += f" — {detail}"
    st.markdown(text)


def category_badge_html(cat_key: str) -> str:
    """HTML d'un badge coloré pour une catégorie de feature."""
    cat = FEATURE_CATEGORIES.get(cat_key, FEATURE_CATEGORIES["other"])
    return (
        f"<span style='background:{cat['color']}22;color:{cat['color']};"
        f"padding:2px 8px;border-radius:10px;font-size:11px;"
        f"font-weight:600;white-space:nowrap'>"
        f"{cat['emoji']} {cat['label']}</span>"
    )


def show_factors(factors: list[dict]):
    """Affichage textuel groupé par catégorie pour les facteurs."""
    st.caption("Lecture rapide : contribution estimée des variables les plus influentes.")
    for f in factors:
        cat_key = get_feature_category(f["feature"])
        cat = FEATURE_CATEGORIES.get(cat_key, FEATURE_CATEGORIES["other"])
        pct = f["importance"] * 100
        st.markdown(
            f"<div style='border-left:3px solid {cat['color']};padding:6px 12px;"
            f"margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<b>{f['label']}</b>{category_badge_html(cat_key)}"
            f"</div>"
            f"<span style='color:#888;font-size:12px'>"
            f"Valeur : <code>{f['value']:.2f}</code> · "
            f"Importance : <b>{pct:.1f}%</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def compute_shap_local(explainer, row_df: pd.DataFrame, feature_cols: list[str]):
    """Calcule les SHAP values pour une seule ligne et retourne (shap_values_1d, expected)."""
    if explainer is None:
        return None, None
    try:
        sv = explainer.shap_values(row_df[feature_cols])
        if isinstance(sv, list):
            sv = sv[1]
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1] if len(expected) > 1 else expected[0]
        return sv[0] if sv.ndim == 2 else sv, float(expected)
    except Exception:
        return None, None


def show_shap_waterfall(shap_vals, expected_value, feature_cols: list[str], title: str = ""):
    """Affiche un waterfall SHAP via Plotly, avec préfixe emoji catégorie."""
    indices = np.argsort(np.abs(shap_vals))[::-1][:12]

    cat_keys = [get_feature_category(feature_cols[i]) for i in indices]
    features = [
        f"{FEATURE_CATEGORIES[c]['emoji']}  {feature_label(feature_cols[i])}"
        for i, c in zip(indices, cat_keys)
    ]
    values = [shap_vals[i] for i in indices]

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        y=features[::-1],
        x=values[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in values[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP : %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title or "Facteurs explicatifs (SHAP)",
                   font=dict(size=14)),
        xaxis_title="Impact (log-odds)",
        yaxis_title="",
        height=max(320, len(indices) * 32),
        margin=dict(l=240, r=70, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.3)", line_width=1)
    fig.add_annotation(
        text=f"Base SHAP : {expected_value:.3f}",
        xref="paper", yref="paper",
        x=1.0, y=-0.08,
        showarrow=False,
        font=dict(size=11, color="#888"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("🔴 Rouge : augmente le risque  ·  🟢 Vert : réduit le risque  ·  Emoji : catégorie de donnée")


def apply_extrapolation_stress(
    proba_model: float,
    params: dict,
    df: pd.DataFrame,
) -> tuple[float, list[str]]:
    """
    Le Random Forest ne sait pas extrapoler au-delà des données d'entraînement.
    Cette fonction détecte les paramètres qui dépassent les bornes connues
    et applique un bonus de risque proportionnel au dépassement.

    Retourne (probabilité_ajustée, liste_des_facteurs_de_stress).
    """
    stress = 0.0
    details = []

    bounds = {
        "total_load_kw": ("Consommation", df["total_load_kw"].max(), df["total_load_kw"].quantile(0.95)),
        "temperature_2m": ("Température", df["temperature_2m"].max(), df["temperature_2m"].quantile(0.95)),
        "wind_speed_10m": ("Vent", df["wind_speed_10m"].max(), df["wind_speed_10m"].quantile(0.95)),
        "precipitation": ("Précipitations", df["precipitation"].max(), df["precipitation"].quantile(0.95)),
    }

    param_map = {
        "total_load_kw": params["total_load_kw"],
        "temperature_2m": params["temperature_2m"],
        "wind_speed_10m": params["wind_speed"],
        "precipitation": params["precipitation"],
    }

    for key, (label, data_max, p95) in bounds.items():
        val = param_map[key]
        if val > data_max:
            overshoot = (val - data_max) / max(data_max - p95, 1)
            bonus = min(0.25, overshoot * 0.10)
            stress += bonus
            details.append(f"{label} ({val:.0f}) dépasse le max observé ({data_max:.0f})")
        elif val > p95:
            overshoot = (val - p95) / max(data_max - p95, 1)
            bonus = min(0.10, overshoot * 0.05)
            stress += bonus
            details.append(f"{label} ({val:.0f}) au-dessus du 95e percentile ({p95:.0f})")

    # Synergie : si plusieurs facteurs sont en stress simultanément, le risque est amplifié
    if len(details) >= 2:
        stress *= 1.0 + 0.3 * (len(details) - 1)

    proba_adjusted = min(0.99, proba_model + stress)
    return proba_adjusted, details


def adjust_for_hospital_profile(
    proba: float,
    hospital_info: dict,
) -> tuple[float, list[str]]:
    """
    Ajuste la probabilité selon le profil de risque de l'hôpital sélectionné.

    Le modèle est entraîné sur un corpus multi-hôpitaux avec des niveaux de
    fiabilité réseau différents. On applique un léger ajustement contextuel :
      - Fiabilité basse (ex: Éthiopie 23%) → risque augmenté
      - Fiabilité haute (ex: USA 98%) → risque diminué
    """
    ref_reliability = 50.0
    hospital_reliability = hospital_info.get("who_reliability", ref_reliability)

    delta = (ref_reliability - hospital_reliability) / 100.0
    # delta > 0 quand l'hôpital est moins fiable que la référence → risque augmenté
    # delta < 0 quand l'hôpital est plus fiable → risque diminué

    factor = 1.0 + delta * 1.5

    adjusted = min(0.99, max(0.01, proba * factor))

    notes = []
    stability = hospital_info.get("grid_stability", "moyen")
    if hospital_reliability < 30:
        notes.append(f"Réseau {stability} — fiabilité OMS très basse ({hospital_reliability:.0f}%)")
    elif hospital_reliability < 55:
        notes.append(f"Réseau {stability} — fiabilité OMS basse ({hospital_reliability:.0f}%)")
    elif hospital_reliability > 90:
        notes.append(f"Réseau {stability} — fiabilité OMS élevée ({hospital_reliability:.0f}%)")

    if not hospital_info.get("has_solar"):
        notes.append("Pas de panneaux solaires — dépendance totale au réseau")
    if not hospital_info.get("has_generator"):
        notes.append("Pas de générateur de secours")

    return adjusted, notes


def build_simulation_row(params: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Construit une ligne de features à partir des paramètres utilisateur.

    Stratégie : on cherche dans les données réelles la ligne la plus similaire
    aux conditions demandées (même heure, même mois, consommation proche).
    On part de cette ligne RÉELLE (qui a des features rolling cohérentes)
    et on ne remplace que les paramètres que l'utilisateur a modifiés.
    """
    hour = params["hour"]
    month = params["month"]
    day_of_week = params["day_of_week"]
    load = params["total_load_kw"]
    solar = params["solar_pv_kw"]
    base = params["base_load_kw"]
    steril = params["sterilization_kw"]

    candidates = df.copy()
    candidates["_hour_dist"] = abs(candidates["hour"] - hour)
    candidates["_month_dist"] = abs(candidates["month"] - month)
    candidates["_load_dist"] = abs(candidates["total_load_kw"] - load)
    candidates["_score"] = (
        candidates["_hour_dist"] * 3
        + candidates["_month_dist"]
        + candidates["_load_dist"] / 30
    )
    best_idx = candidates["_score"].idxmin()
    ref = df.loc[best_idx, feature_cols].copy()

    ref["total_load_kw"] = load
    ref["solar_pv_kw"] = solar
    ref["base_load_kw"] = base
    ref["sterilization_kw"] = steril
    ref["temperature_2m"] = params["temperature_2m"]
    ref["relative_humidity_2m"] = params["humidity"]
    ref["wind_speed_10m"] = params["wind_speed"]
    ref["precipitation"] = params["precipitation"]
    ref["surface_pressure"] = params["pressure"]
    ref["shortwave_radiation"] = params["radiation"]

    ref["hour"] = hour
    ref["month"] = month
    ref["day_of_week"] = day_of_week
    ref["is_weekend"] = 1 if day_of_week >= 5 else 0
    ref["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    ref["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    ref["month_sin"] = np.sin(2 * np.pi * month / 12)
    ref["month_cos"] = np.cos(2 * np.pi * month / 12)
    ref["is_public_holiday"] = 0

    total = max(load, 1.0)
    ref["solar_ratio"] = solar / total
    ref["base_load_ratio"] = base / total
    ref["peak_ratio"] = load / max(ref.get("load_rolling_24h", load), 1.0)

    ref["temp_humidity_interaction"] = params["temperature_2m"] * params["humidity"] / 100
    ref["wind_precipitation_interaction"] = params["wind_speed"] * params["precipitation"]
    ref["solar_available"] = 1 if params["radiation"] > 50 else 0
    ref["heat_stress"] = 1 if params["temperature_2m"] > 30 else 0

    # Météo avancée : recalcul cohérent avec les paramètres utilisateur
    t = params["temperature_2m"]
    rh = params["humidity"]
    a, b = 17.27, 237.7
    gamma = (a * t / (b + t)) + np.log(rh / 100 + 1e-10)
    ref["dew_point_2m"] = b * gamma / (a - gamma)
    ref["rain_intensity"] = params["precipitation"] * params["wind_speed"]
    ref["humidity_change_3h"] = 0.0
    ref["pressure_change_3h"] = 0.0

    row_df = pd.DataFrame([ref])
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    return row_df[feature_cols]


# ── Hero / en-tête ──────────────────────────────────────────────────

_mtime = _model_file_mtime()
model = load_model(_mtime)
shap_explainer = load_shap_explainer(_mtime)
lacor_df = load_lacor_features(_features_file_mtime())
feature_cols = get_feature_columns(lacor_df)

# ── Garde-fou : cohérence des features entre entraînement et inférence ──
# Si quelqu'un régénère le dataset sans réentraîner (ou inversement), les
# colonnes peuvent diverger silencieusement et produire des prédictions
# faussées. On force ici l'ordre à correspondre à celui du modèle.
def _model_feature_names(_model):
    fn = getattr(_model, "feature_names_in_", None)
    if fn is None and hasattr(_model, "calibrated_classifiers_"):
        fn = getattr(_model.calibrated_classifiers_[0].estimator,
                      "feature_names_in_", None)
    return list(fn) if fn is not None else None


_model_feats = _model_feature_names(model)
if _model_feats is not None:
    missing_in_data = [c for c in _model_feats if c not in feature_cols]
    extra_in_data = [c for c in feature_cols if c not in _model_feats]
    if missing_in_data or extra_in_data:
        st.warning(
            "**Désynchronisation features ↔ modèle détectée** — "
            "le dataset et le modèle n'ont pas exactement le même set de features. "
            "Re-lancez `python run_pipeline.py` pour ré-entraîner.\n\n"
            f"- Manquantes dans le dataset : {missing_in_data[:6]}{' …' if len(missing_in_data) > 6 else ''}\n"
            f"- Présentes en trop : {extra_in_data[:6]}{' …' if len(extra_in_data) > 6 else ''}"
        )
    feature_cols = _model_feats

_summary_path = MODELS_DIR / "training_summary.json"
_winner_name = "RandomForest"
_n_features_train = len(feature_cols)
if _summary_path.exists():
    try:
        with open(_summary_path) as _f:
            _summary = json.load(_f)
        _winner_name = _summary.get("winner", _winner_name)
    except Exception:
        pass

st.markdown(
    f"""
    <div style='background:linear-gradient(135deg,#1a2530,#2c3e50);
                color:white;padding:24px 32px;border-radius:14px;
                margin-bottom:20px'>
        <div style='display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:20px'>
            <div>
                <div style='font-size:28px;font-weight:800;line-height:1.1'>
                    ⚡ Prédiction de coupures d'électricité
                </div>
                <div style='font-size:14px;color:#bdc3c7;margin-top:4px'>
                    Hôpitaux · Données réelles 2022 · Modèle expliqué par SHAP
                </div>
            </div>
            <div style='display:flex;gap:28px;flex-wrap:wrap'>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Modèle
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#3498db'>
                        {_winner_name}
                    </div>
                </div>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Features
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#2ecc71'>
                        {_n_features_train}
                    </div>
                </div>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Sources
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#f39c12'>
                        {len(DATA_SOURCES)}
                    </div>
                </div>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Hôpitaux
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#e84393'>
                        {len(ALL_HOSPITAL_KEYS)}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sélection de l'hôpital ─────────────────────────────────────────

col_select, col_info = st.columns([1, 2])

with col_select:
    only_real_hospitals = st.toggle(
        "Données réelles uniquement",
        value=True,
        help="Active ce mode pour masquer les profils estimés (africa_grid).",
    )
    hospital_options = REAL_HOSPITAL_KEYS if only_real_hospitals else ALL_HOSPITAL_KEYS
    if only_real_hospitals:
        st.caption("Mode strict : uniquement données réelles (pas d'estimation).")
    else:
        st.caption("Mode complet : tous les hôpitaux (réels + profils estimés).")
    hospital_key = st.selectbox(
        "Hôpital",
        options=hospital_options,
        format_func=lambda k: f"{HOSPITAL_DISPLAY[k]['flag']}  {HOSPITAL_DISPLAY[k]['name']} — {HOSPITAL_DISPLAY[k]['location']}",
    )

hospital = HOSPITAL_DISPLAY[hospital_key]

with col_info:
    reliability = hospital.get("who_reliability", 50)
    if reliability < 30:
        rel_color = "#e74c3c"
    elif reliability < 55:
        rel_color = "#f39c12"
    elif reliability < 80:
        rel_color = "#3498db"
    else:
        rel_color = "#2ecc71"

    solar_html = (
        "<span style='color:#f39c12;font-weight:600'>☀️ Solaire</span>"
        if hospital.get("has_solar")
        else "<span style='color:#999'>✕ Pas de solaire</span>"
    )
    gen_html = (
        "<span style='color:#e67e22;font-weight:600'>⚙️ Générateur</span>"
        if hospital.get("has_generator")
        else "<span style='color:#999'>✕ Pas de générateur</span>"
    )

    eric_line = ""
    if hospital.get("data_source") == "eric":
        area = hospital.get("floor_area_m2", 0)
        annual = hospital.get("annual_electricity_kwh", 0)
        eric_line = (
            f"<div style='margin-top:6px;font-size:12px;color:#666'>"
            f"📊 <b>Données ERIC NHS</b> · {area:,} m² · "
            f"{annual / 1e6:.0f} GWh/an · {annual / area:.0f} kWh/m²"
            f"</div>"
        )
    elif hospital.get("data_source") == "nyc_ll84":
        area = hospital.get("floor_area_m2", 0)
        annual = hospital.get("annual_electricity_kwh", 0)
        eric_line = (
            f"<div style='margin-top:6px;font-size:12px;color:#666'>"
            f"📊 <b>NYC Local Law 84</b> · {area:,} m² · "
            f"{annual / 1e6:.0f} GWh/an · {annual / area:.0f} kWh/m²"
            f"</div>"
        )

    st.markdown(
        f"""
        <div style='border:1px solid #e0e0e0;border-radius:10px;
                    padding:14px 18px;background:#fafafa'>
            <div style='font-size:18px;font-weight:700'>
                {hospital['flag']} {hospital['name']}
            </div>
            <div style='font-size:13px;color:#666'>
                {hospital['location']} · {hospital['type']}
            </div>
            <div style='display:flex;gap:24px;margin-top:10px;flex-wrap:wrap;
                        font-size:13px'>
                <div><b>{hospital['beds']}</b> lits</div>
                <div><b>{hospital.get('avg_load_kw', '?'):,}</b> –
                     <b>{hospital.get('max_load_kw', '?'):,}</b> kW</div>
                <div>{solar_html}</div>
                <div>{gen_html}</div>
                <div>Réseau : <b>{hospital.get('grid_stability', '?')}</b></div>
                <div>Fiabilité OMS :
                     <span style='color:{rel_color};font-weight:700'>
                     {reliability:.0f}%</span></div>
            </div>
            {eric_line}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sources de données spécifiques à cet hôpital ───────────────────
_hospital_sources = detect_hospital_data_sources(hospital_key, hospital)

_sources_html = "".join(
    f"<li style='margin:4px 0'>{s['emoji']}  {s['label']}</li>"
    for s in _hospital_sources
)
st.markdown(
    f"""
    <div style='border:1px solid #e0e0e0;border-radius:10px;
                padding:12px 18px;background:#fafafa;margin-top:10px'>
        <div style='font-size:13px;font-weight:700;color:#2c3e50;
                    margin-bottom:6px'>
            🗂️  Sources utilisées pour {hospital['name']}
            <span style='font-weight:400;color:#888'>
                ({len(_hospital_sources)} sources)
            </span>
        </div>
        <ul style='margin:0;padding-left:20px;font-size:13px;color:#444;
                   columns:2;column-gap:24px'>
            {_sources_html}
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── État réseau local (Electricity Maps, 24 h glissantes) ─────────
_em_df = load_electricitymaps_snapshot(hospital_key)
if _em_df is not None and not _em_df.empty:
    em_24h = _em_df.tail(24).copy()
    _em_last = em_24h.iloc[-1]
    _em_zone = _em_last.get("em_zone", "N/A")
    _em_load = pd.to_numeric(_em_last.get("em_total_load_mw"), errors="coerce")
    _em_carbon = pd.to_numeric(_em_last.get("em_carbon_intensity_gco2_kwh"), errors="coerce")
    _em_ren = pd.to_numeric(_em_last.get("em_renewable_pct"), errors="coerce")
    _em_fossil = pd.to_numeric(_em_last.get("em_fossil_pct"), errors="coerce")
    _em_ts = pd.to_datetime(_em_last.get("datetime"), errors="coerce")

    if "em_total_load_mw" in em_24h.columns:
        em_24h["em_total_load_mw"] = pd.to_numeric(
            em_24h["em_total_load_mw"], errors="coerce",
        )
    em_avg_24h = em_24h["em_total_load_mw"].mean() if "em_total_load_mw" in em_24h else float("nan")
    stress_ratio = (_em_load / em_avg_24h) if (em_avg_24h and not pd.isna(em_avg_24h) and em_avg_24h > 0) else float("nan")

    avg_load_kw_h = float(hospital.get("avg_load_kw", 0) or 0)
    if avg_load_kw_h > 0 and not pd.isna(em_avg_24h) and em_avg_24h > 0:
        em_24h["hospital_load_kw_est"] = avg_load_kw_h * (
            em_24h["em_total_load_mw"] / em_avg_24h
        )
        hospital_now_kw = float(em_24h["hospital_load_kw_est"].iloc[-1])
    else:
        em_24h["hospital_load_kw_est"] = pd.NA
        hospital_now_kw = float("nan")

    st.markdown(
        f"#### ⚡ Réseau local temps réel — Electricity Maps "
        f"<span style='color:#888;font-size:13px;font-weight:400'>"
        f"(zone {_em_zone}, 24 h glissantes)</span>",
        unsafe_allow_html=True,
    )
    em_c1, em_c2, em_c3, em_c4, em_c5 = st.columns(5)
    em_c1.metric("Charge réseau", "N/A" if pd.isna(_em_load) else f"{_em_load:,.0f} MW")
    em_c2.metric(
        "Stress vs moy. 24 h",
        "N/A" if pd.isna(stress_ratio) else f"× {stress_ratio:.2f}",
        delta=None if pd.isna(stress_ratio) else f"{(stress_ratio - 1) * 100:+.1f} %",
    )
    em_c3.metric(
        "Conso hôpital estimée",
        "N/A" if pd.isna(hospital_now_kw) else f"{hospital_now_kw:,.0f} kW",
    )
    em_c4.metric(
        "Intensité carbone",
        "N/A" if pd.isna(_em_carbon) else f"{_em_carbon:,.0f} gCO₂/kWh",
    )
    if not pd.isna(_em_ren) and not pd.isna(_em_fossil):
        em_c5.metric("Mix", f"{_em_ren:.0f}% ren. / {_em_fossil:.0f}% foss.")
    else:
        em_c5.metric("Mix", "N/A")

    if "em_total_load_mw" in em_24h.columns and em_24h["em_total_load_mw"].notna().any():
        em_chart_l, em_chart_r = st.columns(2)
        with em_chart_l:
            fig_grid = go.Figure()
            fig_grid.add_trace(go.Scatter(
                x=em_24h["datetime"], y=em_24h["em_total_load_mw"],
                mode="lines+markers", name="Charge réseau (MW)",
                line=dict(color="#f1c40f", width=2),
            ))
            fig_grid.update_layout(
                title="Charge réseau zone (24 h)",
                xaxis_title="Heure", yaxis_title="MW",
                height=260, margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_grid, width="stretch")
        with em_chart_r:
            if em_24h["hospital_load_kw_est"].notna().any():
                fig_hosp = go.Figure()
                fig_hosp.add_trace(go.Scatter(
                    x=em_24h["datetime"], y=em_24h["hospital_load_kw_est"],
                    mode="lines+markers",
                    name="Conso hôpital estimée (kW)",
                    line=dict(color="#e84393", width=2),
                ))
                fig_hosp.update_layout(
                    title=f"Conso {hospital['name']} estimée (24 h)",
                    xaxis_title="Heure", yaxis_title="kW",
                    height=260, margin=dict(l=40, r=20, t=40, b=40),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_hosp, width="stretch")
            else:
                st.caption(
                    "Estimation indisponible (avg_load_kw inconnu pour cet hôpital)."
                )

    if not pd.isna(_em_ts):
        st.caption(
            f"Dernière mesure Electricity Maps : {_em_ts.strftime('%Y-%m-%d %H:%M UTC')}"
            " — estimation = avg_load_kw × (charge_réseau_now / charge_réseau_moy_24h)."
        )
else:
    st.info(
        "Electricity Maps non disponible pour cet hôpital. "
        "Exécute `python -m src.data.ingest_electricitymaps` pour alimenter ce panneau.",
        icon="⚡",
    )

# ── Bandeau « Sources & facteurs du modèle » ───────────────────────

with st.expander("📊  Sources de données & facteurs du modèle", expanded=False):
    st.markdown(
        f"Le modèle exploite **{len(DATA_SOURCES)} sources de données** complémentaires "
        "pour estimer le risque de coupure. Voici les facteurs ayant le plus "
        "d'impact, en moyenne, sur les prédictions du modèle entraîné."
    )

    panel_left, panel_right = st.columns([3, 2])
    with panel_left:
        show_top_factors_panel(top_n=12)
    with panel_right:
        show_category_breakdown()

    st.markdown("#### Sources de données utilisées")
    show_data_sources_panel()

st.divider()

# ── Avertissement si les signaux externes sont neutralisés ────────
if hospital.get("data_source") == "eric":
    st.info(
        f"**Hôpital NHS — données réelles ERIC 2022-23** · "
        f"La consommation horaire de {hospital['name']} est désagrégée à "
        "partir des relevés annuels officiels NHS Estates Returns "
        "Information Collection. La météo locale est récupérée via "
        "Open-Meteo (latitude/longitude réelles de l'hôpital). Les "
        "signaux événementiels (GDELT, GDACS) ne sont pas ingérés pour "
        "ce site et sont neutralisés à 0.",
        icon="🇬🇧",
    )
elif hospital.get("data_source") == "nyc_ll84":
    st.info(
        f"**Hôpital NYC — données réelles Local Law 84** · "
        f"La consommation annuelle de {hospital['name']} est issue du "
        "registre obligatoire NYC LL84 (data.cityofnewyork.us, dataset "
        "5zyy-y8am, ~120 hôpitaux NYC publiés). Désagrégation horaire avec "
        "pic estival (climatisation Con Edison) et météo Open-Meteo "
        "locale. Signaux événementiels neutralisés à 0.",
        icon="🇺🇸",
    )
elif hospital.get("data_source") == "africa_grid":
    st.info(
        f"**Hôpital Afrique — profil de consommation estimé (temps réel)** · "
        f"La charge de {hospital['name']} est estimée à partir d'un profil "
        "hospitalier de référence (Lacor) mis à l'échelle selon la taille "
        "du site, puis pilotée par la météo locale et le signal réseau "
        "Electricity Maps (zone locale, charge réseau, mix, carbone). "
        "Ce ne sont pas des compteurs internes publiés par l'hôpital.",
        icon="🌍",
    )

# ── Chargement des données spécifiques à l'hôpital ────────────────────
try:
    df = load_hospital_data(hospital_key, hospital)
except Exception as e:
    st.error(
        f"**Impossible de charger les données pour {hospital['name']}** : {e}\n\n"
        "Vérifiez que le pipeline a été exécuté et que les fichiers de données existent."
    )
    st.stop()

if df is None or df.empty:
    st.error(
        f"**Aucune donnée disponible pour {hospital['name']}.**\n\n"
        "Exécutez le pipeline pour générer les données :\n"
        "```bash\npython run_pipeline.py\n```"
    )
    st.stop()

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

# ── Onglets ──────────────────────────────────────────────────────────

tab_predict, tab_forecast, tab_simulate = st.tabs([
    "🔍  Prédiction en temps réel",
    "🔮  Prévisions J+7",
    "🎛️  Simulation manuelle",
])


# ═══════════════════════════════════════════════════════════════════
# ONGLET 1 : PRÉDICTION HISTORIQUE
# ═══════════════════════════════════════════════════════════════════

with tab_predict:
    if hospital.get("data_source") == "eric":
        data_label = "données ERIC NHS (historique)"
    elif hospital.get("data_source") == "nyc_ll84":
        data_label = "données NYC LL84 (historique)"
    elif hospital.get("data_source") == "africa_grid":
        data_label = "profil estimé (quasi temps réel)"
    else:
        data_label = "données historiques"
    st.markdown(
        f"<p style='color:#888'>Estime le risque de coupure à court terme pour <b>{hospital['name']}</b> "
        f"à partir des <b>72 dernières heures</b> ({data_label}). "
        f"Seuil d'alerte principal : <b>50%</b>.</p>",
        unsafe_allow_html=True,
    )
    ui_step("Étape 1", "Lancer l'analyse des données récentes")

    if st.button("Analyser le risque (72 h)", type="primary", width="stretch", key="btn_predict"):
        try:
            with st.spinner("Analyse des 72 dernières heures en cours…"):
                recent = df.tail(72).copy()
                if len(recent) < 2:
                    st.warning("Pas assez de données pour l'analyse (minimum 2 heures requises).")
                    st.stop()
                X = ensure_numeric_feature_frame(recent, feature_cols)
                proba_series = model.predict_proba(X)[:, 1]
                recent["outage_probability"] = proba_series

                high_risk = recent[recent["outage_probability"] > 0.5]
                if high_risk.empty:
                    max_idx = recent["outage_probability"].idxmax()
                    max_proba = recent.loc[max_idx, "outage_probability"]
                    hours_away = abs((recent.loc[max_idx, "datetime"] - recent["datetime"].iloc[-1]).total_seconds() / 3600)
                else:
                    max_proba = high_risk.iloc[0]["outage_probability"]
                    hours_away = max(0, (high_risk.iloc[0]["datetime"] - recent["datetime"].iloc[-1]).total_seconds() / 3600)

                max_proba, h_notes = adjust_for_hospital_profile(max_proba, hospital)
                recent["outage_probability"] = recent["outage_probability"].apply(
                    lambda p: adjust_for_hospital_profile(p, hospital)[0]
                )
                duration = round(1.0 + max_proba * 4.0, 1) if max_proba > 0.5 else 0.5
                last_row = ensure_numeric_feature_frame(df.tail(1), feature_cols).iloc[-1]
                factors = get_top_factors(model, feature_cols, last_row)
                last_row_df = pd.DataFrame([last_row])
                shap_sv, shap_ev = compute_shap_local(shap_explainer, last_row_df, feature_cols)
        except Exception as e:
            st.error(f"**Erreur lors de l'analyse** : {e}")
            st.stop()

        ui_step("Étape 2", "Résumé du risque estimé")
        show_risk_result(max_proba, hours_away, duration)
        if h_notes:
            st.info("**Profil de l'hôpital** :\n" + "\n".join(f"- {n}" for n in h_notes))
        st.divider()

        ui_step("Étape 3", "Pourquoi ce niveau de risque ?")
        col_factors, col_chart = st.columns([2, 3])

        with col_factors:
            st.subheader("Facteurs explicatifs")
            if shap_sv is not None:
                show_shap_waterfall(shap_sv, shap_ev, feature_cols, title="Facteurs explicatifs (SHAP)")
            else:
                show_factors(factors)

        with col_chart:
            st.subheader("Évolution du risque (72 h)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent["datetime"], y=recent["outage_probability"],
                mode="lines", fill="tozeroy",
                line=dict(color="#e74c3c", width=2),
                fillcolor="rgba(231, 76, 60, 0.15)",
                name="Probabilité",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#f39c12",
                          annotation_text="Seuil d'alerte (50%)", annotation_position="top left")
            fig.update_layout(
                yaxis=dict(title="Probabilité", range=[0, 1], tickformat=".0%"),
                xaxis=dict(title=""), height=350,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()

        ui_step("Étape 4", "Contexte de consommation et statistiques")
        st.subheader(f"Consommation observée — {hospital['name']} (72 h)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=recent["datetime"], y=recent["total_load_kw"],
            mode="lines", name="Charge totale", line=dict(color="#3498db", width=2),
        ))
        if "solar_pv_kw" in recent.columns:
            fig2.add_trace(go.Scatter(
                x=recent["datetime"], y=recent["solar_pv_kw"],
                mode="lines", name="Solaire PV", line=dict(color="#f1c40f", width=2),
            ))
        if "generators_kw" in recent.columns:
            fig2.add_trace(go.Scatter(
                x=recent["datetime"], y=recent["generators_kw"],
                mode="lines", name="Générateur", line=dict(color="#e67e22", width=2),
            ))
        outages = recent[recent["is_outage"] == 1]
        if not outages.empty:
            fig2.add_trace(go.Scatter(
                x=outages["datetime"], y=outages["total_load_kw"],
                mode="markers", marker=dict(color="#e74c3c", size=10, symbol="x"),
                name="Coupures",
            ))
        fig2.update_layout(
            yaxis=dict(title="Puissance (kW)"), xaxis=dict(title=""),
            height=300, margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, width="stretch")

        st.subheader(f"Statistiques clés — {hospital['name']}")
        s1, s2, s3, s4 = st.columns(4)
        n_outages = int(df["is_outage"].sum()) if "is_outage" in df.columns else 0
        pct_outage = 100 * df["is_outage"].mean() if "is_outage" in df.columns and len(df) > 0 else 0
        outage_label = "Coupures (2022)" if hospital.get("data_source") != "africa_grid" else "Coupures estimées (fenêtre affichée)"
        s1.metric(outage_label, f"{n_outages}")
        s2.metric("Taux de coupure", f"{pct_outage:.2f}%")
        s3.metric("Charge moyenne", f"{df['total_load_kw'].mean():.0f} kW")
        s4.metric("Charge max", f"{df['total_load_kw'].max():.0f} kW")


# ═══════════════════════════════════════════════════════════════════
# ONGLET 2 : PRÉVISIONS J+7 (à partir d'Open-Meteo Forecast)
# ═══════════════════════════════════════════════════════════════════

with tab_forecast:
    st.markdown(
        "<p style='color:#888'>Projette le risque de coupure sur les <b>7 prochains jours</b> "
        "à partir des prévisions météo et du profil énergétique de l'hôpital. "
        "Seuil d'alerte : <b>50%</b>, seuil critique : <b>70%</b>.</p>",
        unsafe_allow_html=True,
    )
    ui_step("Étape 1", "Vérifier les prévisions météo disponibles")

    forecast_df = load_meteo_forecast(hospital_key, _forecast_file_mtime(hospital_key))

    if forecast_df is None or forecast_df.empty:
        st.warning(
            "**Pas de prévisions météo disponibles pour cet hôpital.**\n\n"
            "Exécutez la récupération des prévisions :\n"
            "```bash\npython -m src.data.ingest_openmeteo_forecast\n```"
        )
    else:
        fetched_at = forecast_df.get("fetched_at", pd.Series([None])).iloc[0]
        info_line = f"Prévisions reçues : **{fetched_at}**  " if fetched_at else ""
        horizon = f"{(forecast_df['datetime'].max() - forecast_df['datetime'].min()).total_seconds() / 3600:.0f} h"
        st.caption(f"{info_line}· Horizon : **{horizon}** · Source : Open-Meteo Forecast API")

        if st.button("Projeter le risque (J+7)", type="primary", width="stretch", key="btn_forecast"):
            try:
                with st.spinner("Projection horaire du risque sur 7 jours…"):
                    preds = build_forecast_predictions(
                        hist_df=df,
                        forecast_df=forecast_df,
                        hospital_info=hospital,
                        feature_cols=feature_cols,
                        model=model,
                    )
            except Exception as e:
                st.error(f"**Erreur lors de la prévision** : {e}")
                st.stop()

            if preds.empty:
                st.warning("Aucune prédiction n'a pu être générée.")
                st.stop()

            max_idx = preds["outage_probability"].idxmax()
            max_proba = float(preds.loc[max_idx, "outage_probability"])
            max_time = preds.loc[max_idx, "datetime"]
            hours_away = max(0.0, (max_time - pd.Timestamp.now(tz=max_time.tz)).total_seconds() / 3600) \
                if max_time.tz is not None else \
                max(0.0, (max_time - pd.Timestamp.now()).total_seconds() / 3600)

            duration = round(1.0 + max_proba * 4.0, 1) if max_proba > 0.5 else 0.5

            ui_step("Étape 2", "Résumé du pic de risque prévisionnel")
            show_risk_result(max_proba, hours_away, duration)

            # ── Bandeau : horaire du pic ─────────────────────────
            max_time_display = pd.to_datetime(max_time).strftime("%a %d %b %Y · %Hh")
            st.info(
                f"**Pic de risque prévu** : {max_time_display}  "
                f"· Dans **{hours_away:.0f} h** · Probabilité **{max_proba:.0%}**"
            )

            st.divider()

            # ── Timeline principale ──────────────────────────────
            ui_step("Étape 3", "Lecture de la trajectoire horaire et du contexte météo")
            st.subheader("Trajectoire du risque — 7 jours")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds["datetime"], y=preds["outage_probability"],
                mode="lines", fill="tozeroy",
                line=dict(color="#e74c3c", width=2),
                fillcolor="rgba(231, 76, 60, 0.15)",
                name="Probabilité",
                hovertemplate="%{x|%a %d %b %Hh}<br>Risque : %{y:.0%}<extra></extra>",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#f39c12",
                          annotation_text="Seuil d'alerte (50%)", annotation_position="top left")
            fig.add_hline(y=0.7, line_dash="dot", line_color="#e74c3c",
                          annotation_text="Seuil critique (70%)", annotation_position="top left")
            fig.update_layout(
                yaxis=dict(title="Probabilité", range=[0, 1], tickformat=".0%"),
                xaxis=dict(title=""), height=350,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            # ── Météo prévue ─────────────────────────────────────
            st.subheader("Contexte météo prévu")
            fig_meteo = go.Figure()
            fig_meteo.add_trace(go.Scatter(
                x=preds["datetime"], y=preds["temperature_2m"],
                mode="lines", name="Température (°C)",
                line=dict(color="#e67e22", width=2), yaxis="y1",
            ))
            fig_meteo.add_trace(go.Bar(
                x=preds["datetime"], y=preds["precipitation"],
                name="Précipitations (mm)", marker_color="#3498db",
                yaxis="y2", opacity=0.6,
            ))
            fig_meteo.update_layout(
                height=280,
                margin=dict(l=40, r=40, t=20, b=40),
                yaxis=dict(title="Température (°C)", side="left"),
                yaxis2=dict(title="Pluie (mm)", side="right", overlaying="y", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_meteo, width="stretch")

            # ── Résumé quotidien ─────────────────────────────────
            st.subheader("Synthèse par jour")
            daily = preds.copy()
            daily["date"] = daily["datetime"].dt.date
            summary = daily.groupby("date").agg(
                proba_max=("outage_probability", "max"),
                proba_mean=("outage_probability", "mean"),
                heures_risque=("outage_probability", lambda x: int((x > 0.5).sum())),
                temp_max=("temperature_2m", "max"),
                pluie_mm=("precipitation", "sum"),
            ).reset_index()

            def _risk_label(p: float) -> str:
                if p > 0.7:
                    return "🔴 Élevé"
                if p > 0.4:
                    return "🟠 Moyen"
                return "🟢 Faible"

            summary["Niveau"] = summary["proba_max"].map(_risk_label)
            summary = summary.sort_values(["proba_max", "proba_mean"], ascending=[False, False]).reset_index(drop=True)
            summary_display = pd.DataFrame({
                "Jour": pd.to_datetime(summary["date"]).dt.strftime("%a %d %b"),
                "Niveau": summary["Niveau"],
                "Risque max": summary["proba_max"].map(lambda p: f"{p:.0%}"),
                "Risque moyen": summary["proba_mean"].map(lambda p: f"{p:.0%}"),
                "Heures à risque (>50%)": summary["heures_risque"],
                "Temp. max (°C)": summary["temp_max"].round(1),
                "Pluie (mm)": summary["pluie_mm"].round(1),
            })
            st.dataframe(summary_display, hide_index=True, width="stretch")

            # ── Top 5 heures critiques ───────────────────────────
            st.subheader("Top 5 heures les plus à risque")
            top5 = preds.nlargest(5, "outage_probability")[[
                "datetime", "outage_probability", "temperature_2m",
                "precipitation", "wind_speed_10m",
            ]].copy()
            top5_display = pd.DataFrame({
                "Date & heure": top5["datetime"].dt.strftime("%a %d %b %Hh"),
                "Probabilité": top5["outage_probability"].map(lambda p: f"{p:.0%}"),
                "Temp. (°C)": top5["temperature_2m"].round(1),
                "Pluie (mm)": top5["precipitation"].round(1),
                "Vent (km/h)": top5["wind_speed_10m"].round(1),
            })
            st.dataframe(top5_display, hide_index=True, width="stretch")


# ═══════════════════════════════════════════════════════════════════
# ONGLET 3 : SIMULATION MANUELLE
# ═══════════════════════════════════════════════════════════════════

with tab_simulate:
    st.markdown(
        "<p style='color:#888'>Ajustez les paramètres ci-dessous pour simuler "
        "un scénario et voir la probabilité de coupure correspondante.</p>",
        unsafe_allow_html=True,
    )

    # ── Paramètres de simulation ─────────────────────────────────

    st.subheader("Paramètres de la simulation")

    col_time, col_energy, col_meteo = st.columns(3)

    with col_time:
        st.markdown("**⏰ Temporel**")
        sim_hour = st.slider("Heure", 0, 23, 14, key="sim_hour")
        sim_month = st.slider("Mois", 1, 12, 6, key="sim_month")
        day_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        sim_dow = st.selectbox("Jour de la semaine", options=range(7),
                               format_func=lambda x: day_names[x], index=2, key="sim_dow")

    h_avg = hospital.get("avg_load_kw", 133)
    h_max = hospital.get("max_load_kw", 235)
    h_solar = hospital.get("has_solar", True)

    with col_energy:
        st.markdown("**🔌 Énergie**")
        sim_load = st.slider(
            "Consommation totale (kW)",
            min_value=10.0, max_value=float(h_max * 1.5),
            value=float(h_avg),
            step=5.0, key="sim_load",
        )
        if h_solar:
            sim_solar = st.slider(
                "Production solaire PV (kW)",
                min_value=0.0, max_value=float(h_max * 0.7),
                value=float(h_avg * 0.3),
                step=5.0, key="sim_solar",
            )
        else:
            st.slider("Production solaire PV (kW)", min_value=0.0, max_value=1.0,
                       value=0.0, disabled=True, key="sim_solar_disabled")
            sim_solar = 0.0
        sim_base = st.slider(
            "Charge de base (kW)",
            min_value=10.0, max_value=float(h_max),
            value=float(h_avg * 0.85),
            step=5.0, key="sim_base",
        )
        sim_steril = st.slider(
            "Stérilisation (kW)",
            min_value=0.0, max_value=float(h_max * 0.3),
            value=float(h_avg * 0.06),
            step=1.0, key="sim_steril",
        )

    with col_meteo:
        st.markdown("**🌡️ Météo**")
        sim_temp = st.slider("Température (°C)", -10.0, 50.0, 25.0, step=0.5, key="sim_temp")
        sim_hum = st.slider("Humidité (%)", 0, 100, 70, key="sim_hum")
        sim_wind = st.slider("Vent (km/h)", 0.0, 100.0, 10.0, step=1.0, key="sim_wind")
        sim_precip = st.slider("Précipitations (mm)", 0.0, 50.0, 0.0, step=0.5, key="sim_precip")
        sim_pressure = st.slider("Pression (hPa)", 900.0, 1050.0, 1013.0, step=1.0, key="sim_pres")
        sim_rad = st.slider("Rayonnement solaire (W/m²)", 0.0, 1000.0, 200.0, step=10.0, key="sim_rad")

    st.divider()

    # ── Lancer la simulation ─────────────────────────────────────

    if st.button("🎯  Simuler", type="primary", width="stretch", key="btn_simulate"):

        params = {
            "hour": sim_hour,
            "month": sim_month,
            "day_of_week": sim_dow,
            "total_load_kw": sim_load,
            "solar_pv_kw": sim_solar,
            "base_load_kw": sim_base,
            "sterilization_kw": sim_steril,
            "temperature_2m": sim_temp,
            "humidity": sim_hum,
            "wind_speed": sim_wind,
            "precipitation": sim_precip,
            "pressure": sim_pressure,
            "radiation": sim_rad,
        }

        try:
            with st.spinner("Simulation en cours…"):
                sim_row = build_simulation_row(params, df, feature_cols)
                sim_row = ensure_numeric_feature_frame(sim_row, feature_cols)
                proba_raw = model.predict_proba(sim_row)[0][1]
                proba_stress, stress_details = apply_extrapolation_stress(proba_raw, params, df)
                proba, hospital_notes = adjust_for_hospital_profile(proba_stress, hospital)
                duration = round(1.0 + proba * 4.0, 1) if proba > 0.5 else 0.5
                hours_away = max(1, round((1 - proba) * 24))
                factors = get_top_factors(model, feature_cols, sim_row.iloc[0])
                sim_shap_sv, sim_shap_ev = compute_shap_local(shap_explainer, sim_row, feature_cols)
        except Exception as e:
            st.error(f"**Erreur lors de la simulation** : {e}")
            st.stop()

        show_risk_result(proba, hours_away, duration)

        if hospital_notes:
            st.info(
                f"**Profil de l'hôpital** :\n"
                + "\n".join(f"- {n}" for n in hospital_notes)
            )

        if stress_details:
            st.warning(
                "**Conditions extrêmes détectées** (hors des données d'entraînement) :\n"
                + "\n".join(f"- {d}" for d in stress_details)
                + f"\n\nProbabilité du modèle seul : {proba_raw:.0%} → ajustée à **{proba:.0%}**"
            )

        st.divider()

        col_gauge, col_explain = st.columns([1, 1])

        with col_gauge:
            st.subheader("Jauge de risque")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if proba > 0.5 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(46, 204, 113, 0.2)"},
                        {"range": [40, 70], "color": "rgba(243, 156, 18, 0.2)"},
                        {"range": [70, 100], "color": "rgba(231, 76, 60, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f39c12", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

        with col_explain:
            st.subheader("Facteurs explicatifs (SHAP)")
            if sim_shap_sv is not None:
                show_shap_waterfall(sim_shap_sv, sim_shap_ev, feature_cols,
                                    title="Pourquoi ce risque ?")
            else:
                show_factors(factors)

        st.divider()

        # ── Résumé du scénario simulé ────────────────────────────
        st.subheader("Résumé du scénario")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("**⏰ Temporel**")
            st.markdown(
                f"- Heure : **{sim_hour}h**\n"
                f"- Mois : **{sim_month}**\n"
                f"- Jour : **{day_names[sim_dow]}**\n"
                f"- Week-end : **{'Oui' if sim_dow >= 5 else 'Non'}**"
            )
        with r2:
            st.markdown("**🔌 Énergie**")
            st.markdown(
                f"- Consommation : **{sim_load} kW**\n"
                f"- Solaire PV : **{sim_solar} kW**\n"
                f"- Charge de base : **{sim_base} kW**\n"
                f"- Stérilisation : **{sim_steril} kW**"
            )
        with r3:
            st.markdown("**🌡️ Météo**")
            st.markdown(
                f"- Température : **{sim_temp}°C**\n"
                f"- Humidité : **{sim_hum}%**\n"
                f"- Vent : **{sim_wind} km/h**\n"
                f"- Précipitations : **{sim_precip} mm**\n"
                f"- Pression : **{sim_pressure} hPa**\n"
                f"- Rayonnement : **{sim_rad} W/m²**"
            )

        # ── Comparaison avec la médiane ──────────────────────────
        st.divider()
        st.subheader("Comparaison avec les conditions moyennes")

        median_row = build_simulation_row({
            "hour": 12, "month": 6, "day_of_week": 2,
            "total_load_kw": float(h_avg),
            "solar_pv_kw": float(h_avg * 0.3) if h_solar else 0.0,
            "base_load_kw": float(h_avg * 0.85),
            "sterilization_kw": float(h_avg * 0.06),
            "temperature_2m": 25.0, "humidity": 70, "wind_speed": 10.0,
            "precipitation": 0.0, "pressure": 1013.0, "radiation": 200.0,
        }, df, feature_cols)
        median_row = ensure_numeric_feature_frame(median_row, feature_cols)
        median_proba_raw = model.predict_proba(median_row)[0][1]
        median_proba, _ = adjust_for_hospital_profile(median_proba_raw, hospital)

        delta = proba - median_proba
        delta_str = f"{delta:+.0%}"

        c1, c2, c3 = st.columns(3)
        c1.metric("Votre scénario", f"{proba:.0%}")
        c2.metric("Conditions moyennes", f"{median_proba:.0%}")
        c3.metric("Différence", delta_str, delta=f"{delta:+.0%}",
                   delta_color="inverse")
