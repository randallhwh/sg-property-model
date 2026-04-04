"""
model.py — LightGBM training with walk-forward cross-validation and SHAP.

Usage:
    from src.model import train, evaluate, save, load, explain

Walk-forward CV:
    We never train on future data. CV folds split by year:
    Fold 1: train ≤2018, test 2019
    Fold 2: train ≤2019, test 2020
    …etc.

Target: log(PSF)  — log-transforms stabilise variance across districts.
Prediction is converted back to PSF at inference time.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Default LightGBM hyperparameters ─────────────────────────────────────────

DEFAULT_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "learning_rate":    0.05,
    "n_estimators":     800,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "verbose":          -1,
    "n_jobs":           -1,
}

# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X: pd.DataFrame,
    y_psf: pd.Series,
    params: dict | None = None,
    eval_set: tuple | None = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM model on log(PSF).
    Returns the fitted model.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(**p)

    fit_kwargs: dict = {}
    if eval_set:
        X_val, y_val = eval_set
        fit_kwargs = dict(
            eval_set=[(X_val, np.log(y_val))],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

    model.fit(X, np.log(y_psf), **fit_kwargs)
    return model


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_psf: pd.Series,
    min_train_years: int = 3,
) -> pd.DataFrame:
    """
    Walk-forward CV: for each test year Y, train on all data before Y.
    Returns a DataFrame with columns: year, mae, mape, rmse, n_test.
    """
    years = sorted(df["year"].unique())
    results = []

    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue
        train_idx = df["year"] < test_year
        test_idx  = df["year"] == test_year
        if test_idx.sum() < 10:
            continue

        X_tr, y_tr = X[train_idx], y_psf[train_idx]
        X_te, y_te = X[test_idx],  y_psf[test_idx]

        model = train(X_tr, y_tr)
        y_pred = np.exp(model.predict(X_te))

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = mean_squared_error(y_te, y_pred) ** 0.5
        mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100

        results.append({
            "year":   test_year,
            "mae":    round(mae, 1),
            "rmse":   round(rmse, 1),
            "mape":   round(mape, 2),
            "n_test": int(test_idx.sum()),
        })
        print(f"  {test_year}: MAE=${mae:.0f}  MAPE={mape:.1f}%  n={test_idx.sum()}")

    return pd.DataFrame(results)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    y_psf_test: pd.Series,
) -> dict:
    """Return evaluation metrics on a held-out set."""
    y_pred = np.exp(model.predict(X_test))
    mae   = mean_absolute_error(y_psf_test, y_pred)
    rmse  = mean_squared_error(y_psf_test, y_pred) ** 0.5
    mape  = float(np.mean(np.abs((y_psf_test - y_pred) / y_psf_test)) * 100)
    within_10pct = float(np.mean(np.abs(y_psf_test - y_pred) / y_psf_test < 0.10) * 100)
    within_20pct = float(np.mean(np.abs(y_psf_test - y_pred) / y_psf_test < 0.20) * 100)
    return {
        "mae":           round(mae, 1),
        "rmse":          round(rmse, 1),
        "mape":          round(mape, 2),
        "within_10pct":  round(within_10pct, 1),
        "within_20pct":  round(within_20pct, 1),
        "n":             len(y_psf_test),
    }


# ── Quantile models for confidence intervals ──────────────────────────────────

def train_quantile_models(
    X: pd.DataFrame,
    y_psf: pd.Series,
    quantiles: tuple = (0.10, 0.90),
) -> dict[float, lgb.LGBMRegressor]:
    """
    Train low/high quantile regression models for confidence intervals.
    """
    models = {}
    for q in quantiles:
        p = {**DEFAULT_PARAMS, "objective": "quantile", "alpha": q}
        m = lgb.LGBMRegressor(**p)
        m.fit(X, np.log(y_psf))
        models[q] = m
    return models


# ── Save / load ───────────────────────────────────────────────────────────────

def save(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    metadata: dict,
    quantile_models: dict | None = None,
    model_dir: str = "models",
):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "lgbm_psf_model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, default=str)
    if quantile_models:
        joblib.dump(
            quantile_models,
            os.path.join(model_dir, "quantile_models.joblib")
        )
    print(f"Model saved to {model_dir}/")


def load(model_dir: str = "models") -> dict:
    """
    Returns dict with keys:
      model, feature_cols, metadata, quantile_models (or None)
    """
    model = joblib.load(os.path.join(model_dir, "lgbm_psf_model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        metadata = json.load(f)
    qm_path = os.path.join(model_dir, "quantile_models.joblib")
    quantile_models = joblib.load(qm_path) if os.path.exists(qm_path) else None
    return {
        "model":           model,
        "feature_cols":    feature_cols,
        "metadata":        metadata,
        "quantile_models": quantile_models,
    }


# ── SHAP explanation ──────────────────────────────────────────────────────────

def explain(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    max_display: int = 15,
) -> pd.DataFrame:
    """
    Compute SHAP values for rows in X.
    Returns DataFrame with one column per feature showing SHAP contribution
    to log(PSF). Positive = pushes price up; negative = pushes down.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        result = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        return result
    except ImportError:
        # fallback: use LightGBM feature importance
        imp = pd.Series(model.feature_importances_, index=X.columns)
        return imp.sort_values(ascending=False).head(max_display).to_frame("importance")


def shap_summary(shap_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """
    Aggregate SHAP values: mean absolute value per feature.
    Returns top_n features sorted by importance.
    """
    if "importance" in shap_df.columns:
        return shap_df.head(top_n)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    return mean_abs.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})
