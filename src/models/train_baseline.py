"""
Pipeline d'entraînement multi-modèles avec tuning, validation croisée,
calibration et explications SHAP.

Modèles comparés :
  - Random Forest (scikit-learn)
  - XGBoost (xgboost)
  - LightGBM (lightgbm)

Pipeline :
  1. Split temporel 80/20 (train / hold-out test)
  2. Pour chaque modèle : GridSearchCV + TimeSeriesSplit (5 folds)
  3. Tableau comparatif → sélection du meilleur (F1 sur CV)
  4. Entraînement final du meilleur sur tout le train set
  5. Calibration isotonique
  6. Évaluation finale sur le hold-out
  7. SHAP : TreeExplainer sur le test set + sauvegarde
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED, TEST_SIZE
from src.utils.io import load_csv, setup_logging

logger = logging.getLogger(__name__)

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
TARGET = "is_outage"
N_CV_FOLDS = 5

# ── Grilles d'hyperparamètres par modèle ─────────────────────────────

OUTAGE_RATIO = 0.097  # ~9.7 % de coupures

MODEL_CONFIGS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "param_grid": {
            "n_estimators": [200, 300],
            "max_depth": [12, 18, 25],
            "min_samples_leaf": [4, 8],
            "class_weight": [{0: 1, 1: 18}, {0: 1, 1: 22}],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method="hist",
        ),
        "param_grid": {
            "n_estimators": [200, 300],
            "max_depth": [5, 8, 12],
            "learning_rate": [0.05, 0.1],
            "scale_pos_weight": [
                round(1 / OUTAGE_RATIO),
                round(1.5 / OUTAGE_RATIO),
            ],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    },
    "LightGBM": {
        "estimator": LGBMClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "param_grid": {
            "n_estimators": [200, 300],
            "max_depth": [8, 15, -1],
            "learning_rate": [0.05, 0.1],
            "scale_pos_weight": [
                round(1 / OUTAGE_RATIO),
                round(1.5 / OUTAGE_RATIO),
            ],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    },
}


# ── Fonctions utilitaires ────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> tuple:
    drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    y = df[TARGET].astype(int)
    return X, y


def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "brier": round(brier_score_loss(y_true, y_proba), 4),
    }


def _log_metrics(metrics: dict, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    for k, v in metrics.items():
        logger.info("%s%-10s : %.4f", tag, k.capitalize(), v)


# ── Grid Search multi-modèles ────────────────────────────────────────

def run_model_comparison(X_train, y_train) -> dict:
    """
    Pour chaque modèle, exécute un GridSearchCV avec TimeSeriesSplit.
    Retourne un dict {nom: {best_params, best_f1, best_estimator}}.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    results = {}

    for name, cfg in MODEL_CONFIGS.items():
        logger.info("═══ Grid Search : %s ═══", name)
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=tscv,
            scoring="f1",
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        logger.info("  Meilleur F1 (CV) : %.4f", grid.best_score_)
        for k, v in grid.best_params_.items():
            logger.info("    %-25s : %s", k, v)

        results[name] = {
            "best_params": grid.best_params_,
            "best_f1_cv": round(grid.best_score_, 4),
            "best_estimator": grid.best_estimator_,
        }

    return results


def print_comparison_table(comparison: dict, X_test, y_test) -> pd.DataFrame:
    """Évalue chaque meilleur modèle sur le test set et affiche un tableau comparatif."""
    rows = []
    for name, info in comparison.items():
        est = info["best_estimator"]
        y_pred = est.predict(X_test)
        y_proba = est.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["model"] = name
        m["f1_cv"] = info["best_f1_cv"]
        rows.append(m)

    table = pd.DataFrame(rows).set_index("model")
    table = table[["f1_cv", "accuracy", "precision", "recall", "f1", "roc_auc", "brier"]]
    table = table.sort_values("f1", ascending=False)

    logger.info("═══ Comparaison des modèles (test set) ═══")
    logger.info("\n%s", table.to_string())
    return table


# ── Calibration ──────────────────────────────────────────────────────

def calibrate_model(model, X_train, y_train):
    logger.info("═══ Calibration isotonique ═══")
    tscv_calib = TimeSeriesSplit(n_splits=3)
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="isotonic",
        cv=tscv_calib,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_calibration(y_true, y_proba_raw, y_proba_cal) -> None:
    brier_raw = brier_score_loss(y_true, y_proba_raw)
    brier_cal = brier_score_loss(y_true, y_proba_cal)
    logger.info("Brier (brut) : %.4f → Brier (calibré) : %.4f  (%.1f%% mieux)",
                brier_raw, brier_cal, (brier_raw - brier_cal) / brier_raw * 100)

    for label, proba in [("brut", y_proba_raw), ("calibré", y_proba_cal)]:
        try:
            frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=5, strategy="quantile")
            logger.info("Calibration (%s) :", label)
            for mp, fp in zip(mean_pred, frac_pos):
                logger.info("  Prédit: %.2f → Observé: %.2f", mp, fp)
        except ValueError:
            pass


# ── Feature importance ───────────────────────────────────────────────

def extract_feature_importances(model, feature_names: list) -> pd.DataFrame:
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            imp = base.feature_importances_

    if imp is None:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False)

    logger.info("── Top 15 features (MDI) ──")
    for _, row in df.head(15).iterrows():
        logger.info("  %-40s %.4f", row["feature"], row["importance"])
    return df


# ── SHAP ─────────────────────────────────────────────────────────────

def compute_and_save_shap(model, X_test: pd.DataFrame, feature_names: list) -> None:
    """
    Calcule les SHAP values via TreeExplainer et sauvegarde :
      - models/shap_values.npz   (matrice SHAP + expected value)
      - models/shap_feature_importance.csv  (|SHAP| moyen global)
    """
    logger.info("═══ Calcul des SHAP values (TreeExplainer) ═══")

    raw_model = model
    if hasattr(model, "calibrated_classifiers_"):
        raw_model = model.calibrated_classifiers_[0].estimator

    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = expected[1] if len(expected) > 1 else expected[0]

    np.savez_compressed(
        MODELS_DIR / "shap_values.npz",
        shap_values=sv,
        expected_value=np.array([expected]),
        feature_names=np.array(feature_names),
    )
    logger.info("SHAP values sauvegardées → models/shap_values.npz  (%d lignes, %d features)",
                sv.shape[0], sv.shape[1])

    mean_abs = np.abs(sv).mean(axis=0)
    shap_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(MODELS_DIR / "shap_feature_importance.csv", index=False)

    logger.info("── Top 15 features (SHAP |mean|) ──")
    for _, row in shap_imp.head(15).iterrows():
        logger.info("  %-40s %.4f", row["feature"], row["mean_abs_shap"])

    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    logger.info("Explainer SHAP sauvegardé → models/shap_explainer.joblib")


# ── Sauvegarde ───────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modèle sauvegardé → %s", path)


# ── Pipeline principal ───────────────────────────────────────────────

def run() -> None:
    df = load_csv(FEATURES_DIR / "features_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    X, y = prepare_data(df)
    logger.info("Features : %d colonnes, %d lignes", X.shape[1], X.shape[0])
    logger.info("Cible (is_outage) : %d coupures / %d total (%.1f%%)",
                y.sum(), len(y), 100 * y.mean())

    X_train, X_test, y_train, y_test = temporal_split(X, y)
    logger.info("Train : %d | Test : %d", len(X_train), len(X_test))

    # ── 1. Comparaison multi-modèles ──────────────────────────────
    comparison = run_model_comparison(X_train, y_train)
    comp_table = print_comparison_table(comparison, X_test, y_test)
    comp_table.to_csv(MODELS_DIR / "model_comparison.csv")

    winner_name = comp_table.index[0]
    winner_model = comparison[winner_name]["best_estimator"]
    winner_params = comparison[winner_name]["best_params"]
    logger.info("═══ Meilleur modèle : %s ═══", winner_name)

    # ── 2. Évaluation du gagnant (brut) ──────────────────────────
    y_pred_raw = winner_model.predict(X_test)
    y_proba_raw = winner_model.predict_proba(X_test)[:, 1]
    raw_metrics = compute_metrics(y_test, y_pred_raw, y_proba_raw)
    logger.info("── %s (brut) ──", winner_name)
    _log_metrics(raw_metrics, prefix=winner_name)
    logger.info("\n%s", classification_report(y_test, y_pred_raw, zero_division=0))

    # ── 3. Calibration ────────────────────────────────────────────
    calibrated = calibrate_model(winner_model, X_train, y_train)
    y_pred_cal = calibrated.predict(X_test)
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    cal_metrics = compute_metrics(y_test, y_pred_cal, y_proba_cal)
    logger.info("── %s (calibré) ──", winner_name)
    _log_metrics(cal_metrics, prefix=f"{winner_name} cal.")
    logger.info("\n%s", classification_report(y_test, y_pred_cal, zero_division=0))
    evaluate_calibration(y_test, y_proba_raw, y_proba_cal)

    # ── 4. Feature importance (MDI) ──────────────────────────────
    importances = extract_feature_importances(winner_model, list(X.columns))
    importances.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    # ── 5. SHAP values ───────────────────────────────────────────
    compute_and_save_shap(calibrated, X_test, list(X.columns))

    # ── 6. Sauvegarder ───────────────────────────────────────────
    save_model(winner_model, MODELS_DIR / "baseline_rf.joblib")
    save_model(calibrated, MODELS_DIR / "calibrated_rf.joblib")

    summary = {
        "winner": winner_name,
        "winner_params": winner_params,
        "n_cv_folds": N_CV_FOLDS,
        "models_compared": list(MODEL_CONFIGS.keys()),
        "test_metrics_raw": raw_metrics,
        "test_metrics_calibrated": cal_metrics,
        "comparison": {
            name: {
                "best_params": info["best_params"],
                "best_f1_cv": info["best_f1_cv"],
            }
            for name, info in comparison.items()
        },
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Résumé → models/training_summary.json")


if __name__ == "__main__":
    setup_logging()
    run()
