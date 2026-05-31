"""
Configuration centralisée du projet.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
MODELS_DIR = ROOT_DIR / "models"

# ── API Météo ───────────────────────────────────────────────────────
METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
METEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "surface_pressure",
    "shortwave_radiation",
    "cloud_cover",
    "visibility",
    "et0_fao_evapotranspiration",
    "cape",
    "weathercode",
]

# ── Coordonnées des hôpitaux ────────────────────────────────────────
# Dérivé automatiquement du catalogue UI dans `src/utils/hospitals.py`.
# Utilisé par les ingestions géolocalisées (Open-Meteo, Forecast, USGS,
# GDACS, GDELT, NOAA, Electricity Maps).
from src.utils.hospitals import HOSPITAL_DISPLAY, build_hospital_locations  # noqa: E402

HOSPITAL_LOCATIONS = build_hospital_locations()

# ── GDELT DOC 2.0 (signal médiatique événementiel) ──────────────────
# Base d'articles mondiale, sans clé, utile là où on n'a pas d'API réseau.
# Documentation : https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
# Requêtes thématiques par hôpital. Quatre thèmes capturent des dimensions
# distinctes susceptibles d'affecter la stabilité électrique et/ou la
# consommation d'un hôpital :
#   - power    : coupures, délestage, problèmes réseau direct
#   - weather  : événements météo extrêmes
#   - health   : crises sanitaires, surcharge urgences
#   - disaster : épidémies, catastrophes, déplacés (signal indirect fort
#                pour Lacor : afflux patients = pics consommation)
GDELT_QUERIES = {
    "lacor_uganda": {
        "power": '(uganda OR gulu) (blackout OR "power outage" OR "power cut" OR UMEME OR "load shedding")',
        "weather": '(uganda OR gulu) (storm OR flood OR "heavy rain" OR lightning OR thunderstorm)',
        "health": '(uganda OR gulu) (hospital OR clinic) (emergency OR crisis OR disruption)',
        "disaster": '(uganda OR gulu) (epidemic OR outbreak OR ebola OR malaria OR cholera OR refugee OR displaced OR landslide)',
    },
    "phoenix_usa": {
        "power": '(arizona OR phoenix) (blackout OR "power outage" OR "grid failure" OR APS)',
        "weather": '(arizona OR phoenix) (monsoon OR flood OR haboob OR "dust storm" OR wildfire OR heatwave)',
        "health": '(arizona OR phoenix) (hospital OR ER) (emergency OR overloaded OR disruption)',
        "disaster": '(arizona OR phoenix) (wildfire OR evacuation OR "extreme heat" OR drought OR outbreak OR pandemic)',
    },
}

# ── Open-Meteo Air Quality (pollution & poussière) ──────────────────
# Sans clé, granularité horaire, couverture mondiale.
# Variables critiques pour l'hôpital : PM2.5, PM10, ozone, dust
# (surcharge climatisation, filtration, afflux respiratoires).
AIR_QUALITY_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIR_QUALITY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "uv_index",
    "european_aqi",
]

# ── USGS Earthquake API (séismes) ───────────────────────────────────
# API publique gratuite, pas de clé. Renvoie tous les séismes dans un
# rayon donné autour des coordonnées fournies, avec magnitude et heure.
USGS_EARTHQUAKE_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_SEARCH_RADIUS_KM = 500   # rayon autour de l'hôpital
USGS_MIN_MAGNITUDE = 3.0      # seuil minimal (sous ce seuil = bruit)

# ── GDACS (Global Disaster Alert and Coordination System) ──────────
# API publique sans clé, géré conjointement par l'UE (JRC) et l'OCHA.
# Recense les catastrophes naturelles majeures avec niveau d'alerte
# (Vert/Orange/Rouge) et fenêtres temporelles précises.
# Couverture mondiale : inondations, séismes, cyclones, sécheresses,
# feux de forêt, éruptions volcaniques.
# Documentation : https://www.gdacs.org/Knowledge/archivedata.aspx
GDACS_BASE = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
# Mapping niveau d'alerte → score numérique pour pondération
GDACS_ALERT_SCORE = {"Green": 1.0, "Orange": 2.0, "Red": 3.0}
# Mapping type d'événement GDACS → catégorie interne
GDACS_EVENT_TYPES = {
    "FL": "flood",
    "TC": "cyclone",
    "EQ": "earthquake",
    "VO": "volcano",
    "DR": "drought",
    "WF": "wildfire",
    "TS": "tsunami",
}
# Filtres par hôpital : code pays ISO3 (les hôpitaux non listés sont ignorés)
GDACS_FILTERS = {
    "lacor_uganda":         "UGA",
    "phoenix_usa":          "USA",
    "kenyatta_kenya":       "KEN",
    "tikur_ethiopia":       "ETH",
    "groote_schuur_sa":     "ZAF",
    "dhaka_bangladesh":     "BGD",
    "fann_senegal":         "SEN",
    "parirenyatwa_zimbabwe":"ZWE",
    "muhimbili_tanzania":   "TZA",
    "luth_nigeria":         "NGA",
    "korle_bu_ghana":       "GHA",
    "ibn_sina_morocco":     "MAR",
    "kasr_alainy_egypt":    "EGY",
    "chuk_rwanda":          "RWA",
}

# ── Open-Meteo Forecast (prédictions futures pour l'app Streamlit) ──
METEO_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
METEO_FORECAST_DAYS = 7

# ── Electricity Maps API (charge & mix réseau temps réel) ──────────
# API commerciale (token gratuit pour usage perso/recherche, payant pour
# usage pro). Couverture mondiale, granularité horaire (5 min/15 min sur
# certaines zones).
# Documentation : https://static.electricitymaps.com/api/docs/index.html
#
# Endpoints clés exploités ici (préfixe API_BASE) :
#   /v4/zone                      → résolution lat/lon → zone réseau
#   /v4/total-load/latest         → charge totale instantanée du réseau (MW)
#   /v4/total-load/history        → charge horaire des 24 dernières heures
#   /v4/carbon-intensity/latest   → intensité carbone instantanée (gCO2/kWh)
#   /v4/carbon-intensity/history  → intensité carbone horaire 24 h
#   /v4/electricity-mix/history   → mix de production (renouvelable/fossile)
#
# Pourquoi c'est utile pour la prédiction de coupure :
#   - La charge totale (total_load) reflète le STRESS du réseau local
#     auquel l'hôpital est connecté → forte corrélation avec le risque
#     de délestage et de blackout en cascade.
#   - Le mix de production (% renouvelable, % fossile) traduit la
#     stabilité de l'alimentation : plus de fossile → réseau plus
#     pilotable mais plus cher ; plus de solaire/éolien → variabilité
#     accrue, dépendance météo.
#   - L'intensité carbone est un indicateur composite (mix + flux) qui
#     bouge avant une instabilité (déconnexion d'une centrale, etc.).
ELECTRICITYMAPS_BASE = "https://api.electricitymap.org"
ELECTRICITYMAPS_TOKEN_ENV = "ELECTRICITY_MAPS_TOKEN"

# Mapping hôpital → zone réseau électrique locale (key Electricity Maps).
# Quand un hôpital est dans un pays mono-zone on prend le code ISO2 ; pour
# les pays découpés en sous-réseaux (USA, Canada, Australie…) on prend le
# code du Balancing Authority / ISO le plus proche géographiquement.
#   Phoenix       → US-SW-AZPS  (Arizona Public Service, balancing area)
#   NYC           → US-NY-NYIS  (NY-ISO)
#   UK NHS        → GB           (réseau National Grid unique pour GB)
#   Lacor         → UG, etc.
HOSPITAL_ELECTRICITY_ZONES = {
    "lacor_uganda":            "UG",
    "phoenix_usa":             "US-SW-AZPS",
    "kenyatta_kenya":          "KE",
    "tikur_ethiopia":          "ET",
    "groote_schuur_sa":        "ZA",
    "dhaka_bangladesh":        "BD",
    "fann_senegal":            "SN",
    "parirenyatwa_zimbabwe":   "ZW",
    "muhimbili_tanzania":      "TZ",
    "luth_nigeria":            "NG",
    "korle_bu_ghana":          "GH",
    "ibn_sina_morocco":        "MA",
    "kasr_alainy_egypt":       "EG",
    "chuk_rwanda":             "RW",
    "st_thomas_nhs":           "GB",
    "addenbrookes_nhs":        "GB",
    "manchester_nhs":          "GB",
    "kings_college_nhs":       "GB",
    "john_radcliffe_nhs":      "GB",
}

# ── NOAA Storm Events (USA uniquement) ──────────────────────────────
# Base de données publique d'événements météorologiques extrêmes.
# Documentation : https://www.ncdc.noaa.gov/stormevents/
NOAA_STORM_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
NOAA_STORM_CACHE_DIR = RAW_DIR / "noaa_storm"
# Mapping hôpital → filtre (état, comté) pour restreindre le jeu brut.
# Seuls les sites US sont couverts ; les sites hors USA sont ignorés.
NOAA_STORM_FILTERS = {
    "phoenix_usa": {"state": "ARIZONA", "county": "MARICOPA"},
}
# Types d'événements jugés pertinents pour la stabilité électrique.
NOAA_STORM_EVENT_GROUPS = {
    "thunderstorm": ["Thunderstorm Wind", "Lightning", "Tornado", "Hail"],
    "flood": ["Flood", "Flash Flood", "Heavy Rain"],
    "wind": ["High Wind", "Strong Wind"],
    "heat": ["Heat", "Excessive Heat"],
    "winter": ["Winter Storm", "Ice Storm", "Blizzard", "Heavy Snow"],
    "dust": ["Dust Storm", "Dust Devil"],
}

# ── Fichiers de données brutes ──────────────────────────────────────
LACOR_FILE = RAW_DIR / "lacor_hospital.xlsx"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# ── Colonnes exclues du modèle (centralisé) ─────────────────────────
# Réutilisé par `train_baseline.py` (pour `prepare_data`) et par `app.py`
# (pour `get_feature_columns`). Toute modification ici se propage des
# deux côtés.
COLS_TO_DROP = [
    "datetime",
    "is_outage",
    # Colonnes avec fuite directe (connues uniquement pendant la coupure)
    "grid_availability_ratio",
    "generators_kw",
    "generator_active",
    "generator_ratio",
    "grid_availability_rolling_6h",
    "recent_outages_6h",
    "recent_outages_24h",
    # Constantes sur la série mono-hôpital
    "storm_risk",
    # Colonnes brutes météo redondantes avec les features dérivées
    "cloud_cover",
    "visibility",
    "et0_fao_evapotranspiration",
]

# ── Signaux externes exclus du modèle (cf. #3 train-serve skew) ──────
# Ces familles de features (médias GDELT, catastrophes GDACS, sismique USGS,
# qualité de l'air, charge réseau Electricity Maps, tempêtes NOAA) sont :
#   1. mises à 0 à l'inférence pour tout hôpital ≠ Lacor → décalage
#      entraînement/service (train-serve skew) ;
#   2. dominées par les volumes de presse GDELT, qui agissaient comme un
#      proxy temporel spurieux (top importance sur 1 an / 1 hôpital).
# On les exclut donc du jeu de features du modèle. Les colonnes restent
# présentes dans `features_dataset` (inspection / réactivation éventuelle),
# mais ne sont jamais fournies au modèle. Réutilisé par `train_baseline`
# (prepare_data) et `app.py` (get_feature_columns).
EXTERNAL_SIGNAL_PREFIXES = (
    "gdelt_", "gdacs_", "eq_", "air_", "em_", "noaa_", "storm_",
)


def is_external_signal(col: str) -> bool:
    """True si la colonne dérive d'un signal externe exclu du modèle."""
    return col.startswith(EXTERNAL_SIGNAL_PREFIXES)


def drop_external_signal_columns(columns) -> list:
    """Retourne `columns` sans les colonnes de signaux externes."""
    return [c for c in columns if not is_external_signal(c)]

# ── Jours fériés Uganda 2022 (Lacor) ─────────────────────────────────
# Centralisé pour éviter la duplication dans build_features.py et app.py.
UGANDA_PUBLIC_HOLIDAYS_2022 = [
    "2022-01-01",  # New Year
    "2022-01-26",  # NRM Liberation Day
    "2022-02-16",  # Archbishop Janani Luwum Day
    "2022-03-08",  # International Women's Day
    "2022-04-15",  # Good Friday
    "2022-04-18",  # Easter Monday
    "2022-05-01",  # Labour Day
    "2022-05-02",  # Eid al-Fitr (approx.)
    "2022-06-03",  # Martyrs' Day
    "2022-06-09",  # National Heroes' Day
    "2022-07-09",  # Eid al-Adha (approx.)
    "2022-10-09",  # Independence Day
    "2022-12-25",  # Christmas
    "2022-12-26",  # Boxing Day
]

# ── Performance tuning ───────────────────────────────────────────────
# Ces valeurs servent de défauts globaux et peuvent être surchargées
# via CLI dans run_pipeline.py.
CV_FOLDS = 5
FAST_MODE = False
GRID_SCALE = "full"  # "compact" | "full"
SHAP_SAMPLE_SIZE = 5000
