"""
Configuration centralisée du projet.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
MODELS_DIR = ROOT_DIR / "models"

# ── APIs OMS ────────────────────────────────────────────────────────
WHO_BASE = "https://ghoapi.azureedge.net/api"
WHO_ENDPOINTS = {
    "reliability": f"{WHO_BASE}/HCF_REL_ELECTRICITY",
    "countries": f"{WHO_BASE}/Dimension/COUNTRY/DimensionValues",
}

# ── API Météo ───────────────────────────────────────────────────────
METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
METEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
    "shortwave_radiation",
]

# ── Coordonnées des hôpitaux ────────────────────────────────────────
HOSPITAL_LOCATIONS = {
    "lacor_uganda": {"lat": 2.77, "lon": 32.30, "country": "UGA"},
    "phoenix_usa": {"lat": 33.45, "lon": -112.07, "country": "USA"},
}

# ── Fichiers de données brutes ──────────────────────────────────────
LACOR_FILE = RAW_DIR / "lacor_hospital.xlsx"
PHOENIX_FILE = RAW_DIR / "phoenix_hospital.xlsx"
SA_ESKOM_FILE = RAW_DIR / "sa_electricity" / "ESK2033.csv"
SA_LOADSHED_FILE = RAW_DIR / "sa_electricity" / "EskomSePush_history.csv"
KAGGLE_HOSPITAL_FILE = RAW_DIR / "kaggle_hospital" / "hospital_communication_energy_system.csv"
ERIC_DIR = RAW_DIR / "eric"

RANDOM_SEED = 42
TEST_SIZE = 0.2
PYTHON = "/Users/thomaks/miniconda3/bin/python"
