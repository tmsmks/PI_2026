"""
Contenu statique de présentation pour l'app Streamlit (app.py).

Pur : libellés lisibles des features, catégories d'affichage, catalogue des
sources de données, et helpers de catégorisation. Aucune dépendance Streamlit
→ testable et importable isolément. Extrait d'app.py (#10) pour alléger le
monolithe.
"""

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

# Types de sources réellement consommés par le modèle servi (cf. #3 : les
# signaux externes ont été exclus pour éliminer le décalage entraînement/
# service et le proxy temporel GDELT). Les autres sources restent ingérées et
# affichées comme CONTEXTE, mais ne nourrissent pas le modèle.
MODEL_USED_SOURCE_TYPES = {
    "Hospitalier", "Météo historique", "Météo prévision",
}


def _source_used_by_model(src: dict) -> bool:
    return src.get("type") in MODEL_USED_SOURCE_TYPES
