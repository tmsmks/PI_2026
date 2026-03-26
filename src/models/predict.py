"""
Module de prédiction : charge le modèle entraîné et retourne les outputs attendus.

Outputs :
  - Probabilité de coupure (0 → 1)
  - Temps estimé avant la prochaine coupure (heures)
  - Durée estimée de la coupure (heures)
  - Explication (top features contribuant à la prédiction)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)


def load_model(path: Path = MODELS_DIR / "baseline_rf.joblib"):
    return joblib.load(path)


def predict_outage(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque ligne de X, retourne :
      - outage_probability : probabilité de coupure
      - estimated_hours_to_outage : estimation du temps avant coupure
      - estimated_outage_duration_h : durée estimée si coupure
    """
    proba = model.predict_proba(X)[:, 1]

    # Temps estimé avant coupure : heuristique basée sur la probabilité
    # Plus la probabilité est haute, plus la coupure est imminente
    hours_to_outage = np.where(proba > 0.5, np.maximum(1, (1 - proba) * 24), 24 + (1 - proba) * 48)

    # Durée estimée : si on avait un modèle de régression dédié on l'utiliserait ici
    # Pour le baseline, on utilise une heuristique basée sur la probabilité
    duration = np.where(proba > 0.5, 1 + proba * 4, 0)

    results = pd.DataFrame({
        "outage_probability": np.round(proba, 4),
        "estimated_hours_to_outage": np.round(hours_to_outage, 1),
        "estimated_outage_duration_h": np.round(duration, 1),
    })
    return results


def explain_prediction(model, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Pour chaque prédiction, retourne les top_n features les plus importantes.
    Utilise la feature importance globale du Random Forest
    (pour des explications locales, on passerait à SHAP).
    """
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    top_features = importances.head(top_n)
    explanations = []
    for idx in range(len(X)):
        row_explanation = {}
        for feat in top_features.index:
            row_explanation[feat] = {
                "value": float(X.iloc[idx][feat]),
                "global_importance": float(importances[feat]),
            }
        explanations.append(row_explanation)

    return explanations
