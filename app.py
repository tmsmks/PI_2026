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
    "loadshed_avg_stage",
    "loadshed_max_stage",
    "loadshed_pct_active",
    "who_reliability_pct",
    "reliability_risk",
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
    "phoenix_usa": {
        "name": "Phoenix Hospital",
        "location": "Phoenix, Arizona, USA",
        "flag": "🇺🇸",
        "beds": 350,
        "type": "Hôpital de référence",
        "who_reliability": 98.0,
        "lat": 33.45, "lon": -112.07,
        "avg_load_kw": 1156, "max_load_kw": 1576,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
    },
    "kenyatta_kenya": {
        "name": "Kenyatta National Hospital",
        "location": "Nairobi, Kenya",
        "flag": "🇰🇪",
        "beds": 1800,
        "type": "Hôpital universitaire national",
        "who_reliability": 54.0,
        "lat": -1.30, "lon": 36.81,
        "avg_load_kw": 280, "max_load_kw": 450,
        "has_solar": True, "has_generator": True,
        "grid_stability": "faible",
    },
    "tikur_ethiopia": {
        "name": "Tikur Anbessa Hospital",
        "location": "Addis-Abeba, Éthiopie",
        "flag": "🇪🇹",
        "beds": 700,
        "type": "Hôpital universitaire",
        "who_reliability": 23.0,
        "lat": 9.01, "lon": 38.75,
        "avg_load_kw": 180, "max_load_kw": 320,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très faible",
    },
    "groote_schuur_sa": {
        "name": "Groote Schuur Hospital",
        "location": "Le Cap, Afrique du Sud",
        "flag": "🇿🇦",
        "beds": 975,
        "type": "Hôpital universitaire",
        "who_reliability": 65.0,
        "lat": -33.94, "lon": 18.46,
        "avg_load_kw": 420, "max_load_kw": 650,
        "has_solar": True, "has_generator": True,
        "grid_stability": "instable (load shedding)",
    },
    "dhaka_bangladesh": {
        "name": "Dhaka Medical College",
        "location": "Dacca, Bangladesh",
        "flag": "🇧🇩",
        "beds": 2600,
        "type": "Hôpital universitaire",
        "who_reliability": 20.0,
        "lat": 23.73, "lon": 90.40,
        "avg_load_kw": 350, "max_load_kw": 550,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très faible",
    },
    "fann_senegal": {
        "name": "CHU de Fann",
        "location": "Dakar, Sénégal",
        "flag": "🇸🇳",
        "beds": 500,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 45.0,
        "lat": 14.69, "lon": -17.46,
        "avg_load_kw": 200, "max_load_kw": 380,
        "has_solar": True, "has_generator": True,
        "grid_stability": "faible",
    },
    "parirenyatwa_zimbabwe": {
        "name": "Parirenyatwa Hospital",
        "location": "Harare, Zimbabwe",
        "flag": "🇿🇼",
        "beds": 1000,
        "type": "Hôpital central",
        "who_reliability": 48.0,
        "lat": -17.79, "lon": 31.05,
        "avg_load_kw": 220, "max_load_kw": 400,
        "has_solar": False, "has_generator": True,
        "grid_stability": "instable",
    },
    "muhimbili_tanzania": {
        "name": "Muhimbili National Hospital",
        "location": "Dar es Salaam, Tanzanie",
        "flag": "🇹🇿",
        "beds": 1500,
        "type": "Hôpital national de référence",
        "who_reliability": 63.0,
        "lat": -6.80, "lon": 39.27,
        "avg_load_kw": 310, "max_load_kw": 500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
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
}

FEATURE_LABELS = {
    "solar_ratio": "Part du solaire dans la charge",
    "hour": "Heure de la journée",
    "solar_pv_kw": "Production solaire (kW)",
    "load_std_24h": "Variabilité de la charge (24h)",
    "peak_ratio": "Ratio pic / moyenne",
    "load_diff_24h": "Variation de charge (24h)",
    "total_load_kw": "Consommation totale (kW)",
    "sterilization_kw": "Stérilisation (kW)",
    "load_rolling_6h": "Charge moyenne (6h)",
    "base_load_ratio": "Ratio charge de base",
    "hour_sin": "Cycle horaire (sin)",
    "hour_cos": "Cycle horaire (cos)",
    "base_load_kw": "Charge de base (kW)",
    "load_pct_change_1h": "Variation relative (1h)",
    "load_rolling_24h": "Charge moyenne (24h)",
    "load_diff_1h": "Variation de charge (1h)",
    "month": "Mois",
    "month_sin": "Cycle mensuel (sin)",
    "month_cos": "Cycle mensuel (cos)",
    "day_of_week": "Jour de la semaine",
    "is_weekend": "Week-end",
    "temperature_2m": "Température (°C)",
    "relative_humidity_2m": "Humidité relative (%)",
    "wind_speed_10m": "Vitesse du vent (km/h)",
    "precipitation": "Précipitations (mm)",
    "surface_pressure": "Pression (hPa)",
    "shortwave_radiation": "Rayonnement solaire (W/m²)",
    "temp_humidity_interaction": "Interaction temp × humidité",
    "wind_precipitation_interaction": "Interaction vent × pluie",
    "solar_available": "Solaire disponible",
    "heat_stress": "Stress thermique",
}


# ── Chargement ───────────────────────────────────────────────────────

ERIC_DIR = ROOT / "data" / "raw" / "eric"


@st.cache_resource
def load_model():
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
def load_shap_explainer():
    explainer_path = MODELS_DIR / "shap_explainer.joblib"
    if not explainer_path.exists():
        return None
    try:
        return joblib.load(explainer_path)
    except Exception:
        return None


@st.cache_data
def load_lacor_features():
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

    for mcol in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                  "precipitation", "surface_pressure", "shortwave_radiation"]:
        if mcol not in df.columns:
            df[mcol] = 0.0

    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["wind_precipitation_interaction"] = df["wind_speed_10m"] * df["precipitation"]
    df["solar_available"] = (df["shortwave_radiation"] > 50).astype(int)
    df["heat_stress"] = (df["temperature_2m"] > 30).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


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

    lacor_meteo = ROOT / "data" / "raw" / "meteo_lacor_uganda.csv"
    if lacor_meteo.exists():
        meteo = pd.read_csv(lacor_meteo)
        meteo["datetime"] = pd.to_datetime(meteo["datetime"])
        lat = hospital_info.get("lat", 51.5)
        temp_offset = (51.5 - lat) * 0.15
        meteo["temperature_2m"] = meteo["temperature_2m"] - temp_offset
        meteo_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                      "precipitation", "surface_pressure", "shortwave_radiation"]
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in meteo_cols:
            if col in meteo.columns:
                df[col] = meteo[col].values[:len(df)]

    df = _apply_feature_engineering(df)
    return df


@st.cache_data
def load_hospital_data(hospital_key: str, hospital_info: dict) -> pd.DataFrame:
    """Charge les données de l'hôpital sélectionné avec features complètes."""
    if hospital_info.get("data_source") == "eric":
        eric_code = hospital_info["eric_code"]
        eric_df = load_eric_features(eric_code, hospital_info)
        if eric_df is not None:
            return eric_df

    if hospital_key == "lacor_uganda":
        return load_lacor_features()

    base_df = load_lacor_features().copy()
    h_avg = hospital_info.get("avg_load_kw", 133)
    lacor_avg = base_df["total_load_kw"].mean()
    scale = h_avg / max(lacor_avg, 1)

    for col in ["total_load_kw", "solar_pv_kw", "base_load_kw", "sterilization_kw",
                "generators_kw", "load_rolling_6h", "load_rolling_24h",
                "load_std_24h", "load_diff_1h", "load_diff_24h"]:
        if col in base_df.columns:
            base_df[col] = base_df[col] * scale

    total = base_df["total_load_kw"].replace(0, np.nan)
    if "solar_pv_kw" in base_df.columns:
        if not hospital_info.get("has_solar", True):
            base_df["solar_pv_kw"] = 0.0
        base_df["solar_ratio"] = (base_df["solar_pv_kw"] / total).fillna(0).clip(0, 1)
    base_df["base_load_ratio"] = (base_df["base_load_kw"] / total).fillna(0).clip(0, 1)
    base_df["peak_ratio"] = (base_df["total_load_kw"] / base_df["load_rolling_24h"].replace(0, np.nan)).fillna(1)

    return base_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = [c for c in COLS_TO_DROP if c in df.columns]
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


# ── Fonctions utilitaires ────────────────────────────────────────────

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


def show_risk_result(proba: float, hours_away: float, duration: float):
    """Affiche le bloc de résultat de risque (réutilisé dans les 2 onglets)."""
    risk_level, risk_color, risk_icon = risk_display(proba)

    st.markdown(
        f"<h2 style='text-align:center;'>"
        f"{risk_icon} Niveau de risque : "
        f"<span style='color:{risk_color}'>{risk_level}</span>"
        f"</h2>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilité de coupure", f"{proba:.0%}")
    m2.metric(
        "Temps estimé avant coupure",
        f"{hours_away:.0f} h" if hours_away >= 1 else "< 1 h",
    )
    m3.metric("Durée estimée", f"{duration} h")


def show_factors(factors: list[dict]):
    for f in factors:
        pct = f["importance"] * 100
        st.markdown(
            f"**{f['label']}**<br>"
            f"<span style='color:#888'>Valeur : <code>{f['value']:.2f}</code> · "
            f"Importance : {pct:.1f}%</span>",
            unsafe_allow_html=True,
        )
        st.progress(min(f["importance"] / 0.15, 1.0))


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
    """Affiche un waterfall SHAP via Plotly."""
    indices = np.argsort(np.abs(shap_vals))[::-1][:10]

    features = [FEATURE_LABELS.get(feature_cols[i], feature_cols[i]) for i in indices]
    values = [shap_vals[i] for i in indices]

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        y=features[::-1],
        x=values[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title or "Impact SHAP sur cette prédiction",
        xaxis_title="SHAP value (impact sur le log-odds)",
        yaxis_title="",
        height=max(300, len(indices) * 35),
        margin=dict(l=200, r=60, t=40, b=40),
    )
    fig.add_annotation(
        text=f"Base : {expected_value:.3f}",
        xref="paper", yref="paper",
        x=1.0, y=-0.08,
        showarrow=False,
        font=dict(size=11, color="#888"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rouge = pousse vers la coupure · Vert = réduit le risque")


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

    Le modèle est entraîné sur Lacor (fiabilité OMS ~50%).
    Pour un hôpital dans un pays à fiabilité différente, on module le risque :
      - Fiabilité basse (ex: Éthiopie 23%) → risque augmenté
      - Fiabilité haute (ex: USA 98%) → risque diminué
    """
    ref_reliability = 50.0
    hospital_reliability = hospital_info.get("who_reliability", ref_reliability)

    delta = (ref_reliability - hospital_reliability) / 100.0
    # delta > 0 quand l'hôpital est MOINS fiable que Lacor → risque augmenté
    # delta < 0 quand l'hôpital est PLUS fiable → risque diminué

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

    total = max(load, 1.0)
    ref["solar_ratio"] = solar / total
    ref["base_load_ratio"] = base / total
    ref["peak_ratio"] = load / max(ref["load_rolling_24h"], 1.0)

    ref["temp_humidity_interaction"] = params["temperature_2m"] * params["humidity"] / 100
    ref["wind_precipitation_interaction"] = params["wind_speed"] * params["precipitation"]
    ref["solar_available"] = 1 if params["radiation"] > 50 else 0
    ref["heat_stress"] = 1 if params["temperature_2m"] > 30 else 0

    row_df = pd.DataFrame([ref])
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    return row_df[feature_cols]


# ── En-tête ──────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align: center;'>⚡ Prédiction de coupures d'électricité</h1>",
    unsafe_allow_html=True,
)

model = load_model()
shap_explainer = load_shap_explainer()
lacor_df = load_lacor_features()
feature_cols = get_feature_columns(lacor_df)

# ── Sélection de l'hôpital (commun aux 2 onglets) ───────────────────

col_select, col_info = st.columns([1, 2])

with col_select:
    hospital_key = st.selectbox(
        "Hôpital",
        options=list(HOSPITAL_DISPLAY.keys()),
        format_func=lambda k: f"{HOSPITAL_DISPLAY[k]['flag']}  {HOSPITAL_DISPLAY[k]['name']} — {HOSPITAL_DISPLAY[k]['location']}",
    )

hospital = HOSPITAL_DISPLAY[hospital_key]

with col_info:
    solar_icon = "☀️ Solaire" if hospital.get("has_solar") else "❌ Pas de solaire"
    gen_icon = "⚙️ Générateur" if hospital.get("has_generator") else "❌ Pas de générateur"
    reliability = hospital.get("who_reliability", 50)
    if reliability < 30:
        rel_color = "#e74c3c"
    elif reliability < 55:
        rel_color = "#f39c12"
    elif reliability < 80:
        rel_color = "#3498db"
    else:
        rel_color = "#2ecc71"

    eric_line = ""
    if hospital.get("data_source") == "eric":
        area = hospital.get("floor_area_m2", 0)
        annual = hospital.get("annual_electricity_kwh", 0)
        eric_line = (
            f"<br><small>📊 <b>Données ERIC NHS</b> : "
            f"{area:,} m² · {annual / 1e6:.0f} GWh/an · "
            f"{annual / area:.0f} kWh/m²</small>"
        )

    st.markdown(
        f"**{hospital['flag']} {hospital['name']}** · {hospital['location']}<br>"
        f"Type : {hospital['type']} · **{hospital['beds']} lits** · "
        f"Charge : {hospital.get('avg_load_kw', '?'):,} – {hospital.get('max_load_kw', '?'):,} kW<br>"
        f"{solar_icon} · {gen_icon} · "
        f"Réseau : {hospital.get('grid_stability', '?')} · "
        f"Fiabilité : <span style='color:{rel_color};font-weight:bold'>{reliability:.0f}%</span>"
        f"{eric_line}",
        unsafe_allow_html=True,
    )

st.divider()

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

tab_predict, tab_simulate = st.tabs(["🔍  Prédiction en temps réel", "🎛️  Simulation manuelle"])


# ═══════════════════════════════════════════════════════════════════
# ONGLET 1 : PRÉDICTION HISTORIQUE
# ═══════════════════════════════════════════════════════════════════

with tab_predict:
    data_label = "données ERIC NHS" if hospital.get("data_source") == "eric" else "données historiques"
    st.markdown(
        f"<p style='color:#888'>Analyse les 72 dernières heures ({data_label}) "
        f"de <b>{hospital['name']}</b> pour estimer le risque de prochaine coupure.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Lancer l'analyse", type="primary", use_container_width=True, key="btn_predict"):
        try:
            with st.spinner("Analyse en cours…"):
                recent = df.tail(72).copy()
                if len(recent) < 2:
                    st.warning("Pas assez de données pour l'analyse (minimum 2 heures requises).")
                    st.stop()
                X = recent[feature_cols]
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
                last_row = df[feature_cols].iloc[-1]
                factors = get_top_factors(model, feature_cols, last_row)
                last_row_df = pd.DataFrame([last_row])
                shap_sv, shap_ev = compute_shap_local(shap_explainer, last_row_df, feature_cols)
        except Exception as e:
            st.error(f"**Erreur lors de l'analyse** : {e}")
            st.stop()

        show_risk_result(max_proba, hours_away, duration)
        if h_notes:
            st.info("**Profil de l'hôpital** :\n" + "\n".join(f"- {n}" for n in h_notes))
        st.divider()

        col_chart, col_factors = st.columns([3, 2])

        with col_chart:
            st.subheader("Probabilité de coupure (72 dernières heures)")
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
            st.plotly_chart(fig, use_container_width=True)

        with col_factors:
            st.subheader("Facteurs explicatifs")
            if shap_sv is not None:
                show_shap_waterfall(shap_sv, shap_ev, feature_cols,
                                    title="Pourquoi cette prédiction ?")
            else:
                show_factors(factors)

        st.divider()

        st.subheader(f"Consommation — {hospital['name']} (72 dernières heures)")
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
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(f"Statistiques — {hospital['name']}")
        s1, s2, s3, s4 = st.columns(4)
        n_outages = int(df["is_outage"].sum()) if "is_outage" in df.columns else 0
        pct_outage = 100 * df["is_outage"].mean() if "is_outage" in df.columns and len(df) > 0 else 0
        s1.metric("Coupures (2022)", f"{n_outages}")
        s2.metric("Taux de coupure", f"{pct_outage:.2f}%")
        s3.metric("Charge moyenne", f"{df['total_load_kw'].mean():.0f} kW")
        s4.metric("Charge max", f"{df['total_load_kw'].max():.0f} kW")


# ═══════════════════════════════════════════════════════════════════
# ONGLET 2 : SIMULATION MANUELLE
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

    if st.button("🎯  Simuler", type="primary", use_container_width=True, key="btn_simulate"):

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
            st.plotly_chart(fig_gauge, use_container_width=True)

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
        median_proba_raw = model.predict_proba(median_row)[0][1]
        median_proba, _ = adjust_for_hospital_profile(median_proba_raw, hospital)

        delta = proba - median_proba
        delta_str = f"{delta:+.0%}"

        c1, c2, c3 = st.columns(3)
        c1.metric("Votre scénario", f"{proba:.0%}")
        c2.metric("Conditions moyennes", f"{median_proba:.0%}")
        c3.metric("Différence", delta_str, delta=f"{delta:+.0%}",
                   delta_color="inverse")
