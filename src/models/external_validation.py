"""
Validation EXTERNE : le signal météo appris sur Lacor généralise-t-il à un
site RÉEL, indépendant et sur un autre continent ?

Site externe : comté de **Maricopa (Phoenix, Arizona)** — coupures RÉELLES
issues d'EAGLE-I (ORNL/DOE, figshare 24237376, CC BY 4.0), 2022, 15 min,
agrégées à l'heure. Météo : Open-Meteo Phoenix 2022 (déjà ingérée).

Pourquoi un modèle « exogène » : EAGLE-I ne fournit que des *comptes de
clients coupés* (pas la consommation hospitalière). On ne peut donc pas
appliquer le modèle complet (49 features dont charge/historique). On entraîne
un modèle restreint aux features **météo + temporelles** sur Lacor, puis on le
teste sur Maricopa. C'est exactement le pouvoir prédictif EXOGÈNE / d'alerte
précoce (cf. axe #2) — sans auto-régression sur les coupures passées.

Lecture : la métrique clé est le **ROC-AUC sur Maricopa** (capacité de
classement, indépendante du seuil). > ~0.5 ⇒ le signal météo transfère ;
≈ 0.5 ⇒ la relation météo→coupure est spécifique au site.

Lancer : python -m src.models.external_validation
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED, ROOT_DIR
from src.utils.io import load_table, setup_logging
from src.features.build_features import apply_feature_engineering_single
from src.models.train_baseline import REAL_DATA_HOSPITALS, TARGET, compute_metrics

logger = logging.getLogger(__name__)

# Features purement EXOGÈNES (météo + temporel) — aucune consommation, aucune
# auto-régression de coupures, aucun état réseau. Servables pour n'importe quel
# site disposant de météo.
EXOGENOUS_FEATURES = [
    # temporel
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # météo brute
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "precipitation", "surface_pressure", "shortwave_radiation", "cape", "weathercode",
    # météo dérivée
    "temp_humidity_interaction", "wind_precipitation_interaction",
    "solar_available", "heat_stress", "cloud_cover_pct", "dew_point_2m",
    "visibility_m", "evapotranspiration", "rain_intensity",
    "thermal_amplitude_24h", "humidity_change_3h", "pressure_change_3h",
]

EAGLEI_MARICOPA = ROOT_DIR / "data" / "external" / "eaglei_maricopa_2022.csv"
PHOENIX_METEO = ROOT_DIR / "data" / "raw" / "meteo_phoenix_usa.csv"
TARGET_POSITIVE_RATE = 0.10  # binarisation Maricopa visant ~taux Lacor


def _exogenous_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in EXOGENOUS_FEATURES if c in df.columns]


def _load_maricopa_hourly() -> pd.DataFrame:
    """Charge EAGLE-I Maricopa, agrège à l'heure (max clients coupés)."""
    raw = pd.read_csv(EAGLEI_MARICOPA)
    raw["run_start_time"] = pd.to_datetime(raw["run_start_time"])
    raw = raw.set_index("run_start_time")
    hourly = (
        raw["customers_out"].resample("1h").max().fillna(0).rename("customers_out")
        .reset_index().rename(columns={"run_start_time": "datetime"})
    )
    # Couvrir toute l'année 2022 (les heures sans ligne EAGLE-I = 0 coupé)
    full = pd.DataFrame({"datetime": pd.date_range("2022-01-01", "2022-12-31 23:00", freq="1h")})
    hourly = full.merge(hourly, on="datetime", how="left").fillna({"customers_out": 0})
    return hourly


def _binarize(hourly: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """Seuil sur clients coupés visant ~TARGET_POSITIVE_RATE d'heures positives."""
    thr = float(hourly["customers_out"].quantile(1 - TARGET_POSITIVE_RATE))
    thr = max(thr, 1.0)  # au moins 1 client coupé
    hourly = hourly.copy()
    hourly[TARGET] = (hourly["customers_out"] >= thr).astype(int)
    return hourly, thr, int(hourly[TARGET].sum())


def _build_maricopa_exog(hourly_labeled: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    meteo = pd.read_csv(PHOENIX_METEO)
    meteo["datetime"] = pd.to_datetime(meteo["datetime"])
    df = hourly_labeled.merge(
        meteo.drop(columns=[c for c in ["hospital"] if c in meteo.columns]),
        on="datetime", how="inner",
    )
    # EAGLE-I ne fournit pas la consommation ; le pipeline de features en a
    # besoin. On ajoute des colonnes factices (0) — elles ne sont PAS dans
    # EXOGENOUS_FEATURES, donc sans effet sur le modèle exogène.
    for c in ["total_load_kw", "solar_pv_kw", "base_load_kw", "generators_kw", "sterilization_kw"]:
        df[c] = 0.0
    # Features identiques au pipeline (calcule aussi des cols conso, ignorées)
    fe = apply_feature_engineering_single(df)
    for c in feats:
        if c not in fe.columns:
            fe[c] = 0.0
    X = fe[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[TARGET].astype(int)
    return X, y


def run() -> dict:
    # ── Modèle exogène entraîné sur Lacor ────────────────────────────
    lacor = load_table(FEATURES_DIR / "features_dataset.csv")
    lacor["datetime"] = pd.to_datetime(lacor["datetime"])
    lacor = lacor[lacor["hospital"].isin(REAL_DATA_HOSPITALS)].sort_values("datetime")
    feats = _exogenous_columns(lacor)
    logger.info("Features exogènes (météo+temporel) : %d", len(feats))

    X_lac = lacor[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_lac = lacor[TARGET].astype(int)

    # Hold-out temporel interne Lacor (référence) : train 1–9, test 10–12
    m = lacor["datetime"].dt.month.to_numpy()
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=8,
        class_weight={0: 1, 1: 10}, random_state=RANDOM_SEED, n_jobs=-1,
    )
    ref = clone(clf).fit(X_lac[m <= 9], y_lac[m <= 9])
    p_ref = ref.predict_proba(X_lac[m >= 10])[:, 1]
    lacor_internal = compute_metrics(y_lac[m >= 10], (p_ref >= 0.5).astype(int), p_ref)
    logger.info("── Réf. interne Lacor (exogène, test mois 10–12) : F1=%.3f ROC=%.3f ──",
                lacor_internal["f1"], lacor_internal["roc_auc"])

    # Modèle final exogène : entraîné sur TOUT Lacor
    model = clone(clf).fit(X_lac, y_lac)

    # ── Site externe : Maricopa / Phoenix (EAGLE-I réel) ─────────────
    hourly = _load_maricopa_hourly()
    hourly, thr, n_pos = _binarize(hourly)
    X_mar, y_mar = _build_maricopa_exog(hourly, feats)
    rate = float(y_mar.mean())
    logger.info("Maricopa 2022 : %d heures | seuil coupure ≥ %.0f clients | %d heures positives (%.1f%%)",
                len(y_mar), thr, n_pos, 100 * rate)

    proba = model.predict_proba(X_mar)[:, 1]
    pred = (proba >= 0.5).astype(int)
    ext = compute_metrics(y_mar, pred, proba)

    # Baselines sur Maricopa
    base_rate = float(y_mar.mean())
    # climatologie mois×heure (taux historique) comme score de ranking
    clim = (
        pd.DataFrame({"m": hourly["datetime"].dt.month, "h": hourly["datetime"].dt.hour, "y": y_mar.values})
        .groupby(["m", "h"])["y"].transform("mean")
    )
    clim_auc = roc_auc_score(y_mar, clim) if y_mar.nunique() > 1 else float("nan")

    logger.info("══════════ VALIDATION EXTERNE — Maricopa/Phoenix (EAGLE-I réel) ══════════")
    logger.info("  ROC-AUC (modèle Lacro→Maricopa) : %.3f", ext["roc_auc"])
    logger.info("  vs climatologie mois×heure       : %.3f", clim_auc)
    logger.info("  vs hasard                        : 0.500")
    logger.info("  F1=%.3f precision=%.3f recall=%.3f (seuil 0.5)", ext["f1"], ext["precision"], ext["recall"])

    summary = {
        "external_site": "Maricopa County / Phoenix (Arizona, USA)",
        "external_source": "EAGLE-I (ORNL/DOE, figshare 24237376, CC BY 4.0), 2022, agrégé horaire",
        "feature_set": "exogène (météo + temporel)",
        "n_exogenous_features": len(feats),
        "maricopa_outage_threshold_customers": round(thr, 1),
        "maricopa_outage_rate": round(rate, 4),
        "maricopa_n_hours": int(len(y_mar)),
        "external_metrics": ext,
        "external_climatology_roc_auc": round(float(clim_auc), 4),
        "lacor_internal_exogenous_ref": lacor_internal,
        "interpretation": (
            "ROC-AUC nettement > 0.5 ⇒ le signal météo appris sur Lacor classe "
            "mieux que le hasard les heures à coupure d'un site RÉEL indépendant. "
            "≈ 0.5 ⇒ la relation météo→coupure ne transfère pas (spécifique au site). "
            "EAGLE-I = comptes de clients coupés au comté (cible reformulée), pas une "
            "coupure d'hôpital ; horodatages alignés à l'heure (léger flou de fuseau possible)."
        ),
    }
    with open(MODELS_DIR / "external_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Résumé → models/external_validation_summary.json")
    return summary


if __name__ == "__main__":
    setup_logging()
    run()
