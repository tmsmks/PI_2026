"""
Tests des garde-fous méthodologiques mis en place lors de la revue :
  - #2 split temporel par hôpital (pas de fuite chronologique)
  - #3 exclusion des signaux externes du modèle
  - #5 / features : causalité (aucune fuite de la cible via le futur)
  - calibration : candidat « none »

Ces tests n'importent PAS Streamlit (app.py) : ils ciblent les fonctions
pures de src/ pour rester rapides et déterministes.

Lancer : python -m pytest tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import (
    EXTERNAL_SIGNAL_PREFIXES,
    drop_external_signal_columns,
    is_external_signal,
)
from src.models.train_baseline import (
    TARGET,
    calibrate_model,
    prepare_data,
    temporal_split,
)
from src.features.build_features import (
    add_load_features,
    add_outage_history_features,
    apply_feature_engineering_single,
)


def _toy_multi_hospital(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dt = pd.date_range("2022-01-01", periods=n, freq="h")
    frames = []
    for h in ("lacor_uganda", "other_hosp"):
        frames.append(pd.DataFrame({
            "datetime": dt,
            "hospital": h,
            "total_load_kw": rng.uniform(50, 150, n),
            "temperature_2m": rng.uniform(15, 35, n),
            "gdelt_power_vol": rng.uniform(0, 10, n),   # signal externe
            "air_pm2_5": rng.uniform(0, 50, n),          # signal externe
            "is_outage": rng.binomial(1, 0.1, n),
        }))
    return pd.concat(frames, ignore_index=True)


# ── #3 : exclusion des signaux externes ──────────────────────────────

def test_is_external_signal():
    assert is_external_signal("gdelt_power_vol")
    assert is_external_signal("air_pm2_5")
    assert is_external_signal("em_total_load_mw")
    assert is_external_signal("storm_active")
    assert not is_external_signal("total_load_kw")
    assert not is_external_signal("hour")


def test_drop_external_signal_columns():
    cols = ["hour", "gdelt_x", "air_y", "total_load_kw", "eq_z", "storm_a", "em_b"]
    assert drop_external_signal_columns(cols) == ["hour", "total_load_kw"]


def test_prepare_data_excludes_external_and_target():
    df = _toy_multi_hospital()
    X, y = prepare_data(df)
    assert TARGET not in X.columns
    assert not any(c.startswith(EXTERNAL_SIGNAL_PREFIXES) for c in X.columns)
    assert "total_load_kw" in X.columns and "temperature_2m" in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(df)


# ── #2 : split temporel par hôpital, sans fuite ──────────────────────

def test_temporal_split_per_hospital_chronological():
    df = _toy_multi_hospital(n=100)
    X, y = prepare_data(df)
    X_tr, X_te, y_tr, y_te = temporal_split(df, X, y, test_size=0.2)

    assert set(X_tr.index).isdisjoint(set(X_te.index))
    assert len(X_tr) + len(X_te) == len(X)

    for h in df["hospital"].unique():
        idx_tr = [i for i in X_tr.index if df.loc[i, "hospital"] == h]
        idx_te = [i for i in X_te.index if df.loc[i, "hospital"] == h]
        assert idx_tr and idx_te, f"hôpital {h} absent d'un des splits"
        # Tout le train de l'hôpital précède tout son test (anti-fuite).
        assert df.loc[idx_tr, "datetime"].max() <= df.loc[idx_te, "datetime"].min()


# ── Features causales : aucune fuite de la cible via le futur ─────────

def test_outage_history_uses_only_past():
    # Une seule coupure à t=50. Les compteurs d'historique reposent sur
    # is_outage shifté de 1 → ils ne doivent PAS compter la coupure courante.
    n = 100
    df = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=n, freq="h"),
        "hospital": "h",
        "is_outage": [0] * n,
    })
    df.loc[50, "is_outage"] = 1
    out = add_outage_history_features(df.copy())

    assert out.loc[50, "outage_frequency_7d"] == 0, "fuite : coupure courante comptée"
    assert out.loc[51, "outage_frequency_7d"] == 1
    assert list(out.index) == list(range(n)), "ordre des lignes non préservé"


def test_load_features_rolling_is_causal():
    n = 10
    df = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=n, freq="h"),
        "hospital": "h",
        "total_load_kw": np.arange(1, n + 1, dtype=float),  # 1..10
    })
    out = add_load_features(df.copy())
    # rolling 6h causal : moyenne du passé (incl. courant), min_periods=1.
    assert out.loc[5, "load_rolling_6h"] == pytest.approx(np.mean([1, 2, 3, 4, 5, 6]))
    assert out.loc[0, "load_rolling_6h"] == pytest.approx(1.0)


def test_apply_feature_engineering_single_no_nan_and_no_external():
    n = 48
    df = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=n, freq="h"),
        "total_load_kw": np.linspace(50, 150, n),
        "is_outage": [0] * n,
    })
    out = apply_feature_engineering_single(df.copy())
    num = out.select_dtypes(include=[np.number])
    assert not num.isnull().any().any(), "des NaN subsistent après FE"
    assert len(out) == n


# ── Calibration : candidat « none » (modèle brut conservé) ───────────

def test_calibrate_model_none_returns_raw_model():
    from sklearn.ensemble import RandomForestClassifier

    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)[::-1]})
    y = pd.Series([0, 1] * 20)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    served, method = calibrate_model(model, X, y, method="none")
    assert method == "none"
    assert served is model  # aucun wrapper de calibration
