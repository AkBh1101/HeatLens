"""
Fits multiple regression algorithms with:
  - GroupShuffleSplit for city-aware train/test partitioning
  - GridSearchCV with GroupKFold inner loop for hyperparameter search
  - MAE, RMSE, R² evaluation metrics
  - Mean-predictor baseline for skill scoring
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    PROCESSED_DATA_PATH, BEST_MODEL_PATH, METRICS_PATH, FEATURES_PATH, SCALER_PATH,
)
from logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("regression_pipeline")


# ─── Optional boosting backends ───────────────────────────────────────────────

def _try_load_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    except ImportError:
        return None


def _try_load_lightgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    except ImportError:
        return None


# ─── Estimator catalogue ──────────────────────────────────────────────────────

def build_estimator_pool() -> dict:
    pool = {
        "Linear Regression":   LinearRegression(),
        "Ridge Regression":    Ridge(),
        "Lasso Regression":    Lasso(max_iter=5000),
        "Decision Tree":       DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest":       RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }
    xgb_model = _try_load_xgboost()
    if xgb_model:
        pool["XGBoost"] = xgb_model
        logger.info("  XGBoost available")
    lgbm_model = _try_load_lightgbm()
    if lgbm_model:
        pool["LightGBM"] = lgbm_model
        logger.info("  LightGBM available")
    return pool


# ─── Hyperparameter search spaces ─────────────────────────────────────────────

SEARCH_GRIDS = {
    "Ridge Regression":    {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression":    {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "Decision Tree":       {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
    "Random Forest":       {"n_estimators": [100, 200], "max_depth": [5, 8, None],
                            "min_samples_leaf": [1, 2]},
    "Gradient Boosting":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                            "max_depth": [3, 5]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "XGBoost":             {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                            "max_depth": [4, 6], "subsample": [0.8, 1.0]},
    "LightGBM":            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                            "max_depth": [4, 6]},
}


# ─── Feature importance extractor ─────────────────────────────────────────────

def extract_importances(fitted_model, feat_names: list) -> dict:
    try:
        if hasattr(fitted_model, "feature_importances_"):
            scores = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            scores = np.abs(fitted_model.coef_)
        else:
            return {}
        scores = scores / (scores.sum() + 1e-9)
        return dict(zip(feat_names, scores.round(5).tolist()))
    except Exception:
        return {}


# ─── Main training routine ────────────────────────────────────────────────────

def run_training(force: bool = False) -> dict:
    """
    Fit all estimators, evaluate on held-out cities, persist the top model.
    Returns the complete results dictionary.
    """
    if METRICS_PATH.exists() and BEST_MODEL_PATH.exists() and not force:
        logger.info("Saved artefacts detected. Returning cached metrics ...")
        with open(METRICS_PATH) as fh:
            return json.load(fh)

    logger.info("=" * 60)
    logger.info("STEP 3 – MODEL TRAINING")
    logger.info("=" * 60)

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

    data = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"  Loaded {len(data)} rows")

    # ── Resolve feature columns ────────────────────────────────────────────
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH) as fh:
            feat_names = json.load(fh)
    else:
        feat_names = [c for c in FEATURE_COLUMNS if c in data.columns]

    feat_names = [f for f in feat_names if f in data.columns]

    # city_name drives GroupKFold splits but is not fed to the model
    city_labels = data["city_name"].values if "city_name" in data.columns else None
    n_unique_cities = len(np.unique(city_labels)) if city_labels is not None else "?"
    logger.info(
        f"  Features: {len(feat_names)}  |  Samples: {len(data)}  |  Cities: {n_unique_cities}"
    )

    X_raw = data[feat_names].values
    y_all = data[TARGET_COLUMN].values

    # ── Feature scaling ────────────────────────────────────────────────────
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as fh:
            normaliser = pickle.load(fh)
        X_norm = normaliser.transform(X_raw)
    else:
        normaliser = StandardScaler()
        X_norm = normaliser.fit_transform(X_raw)
        with open(SCALER_PATH, "wb") as fh:
            pickle.dump(normaliser, fh)

    # ── City-aware train/test split ────────────────────────────────────────
    if city_labels is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(splitter.split(X_norm, y_all, groups=city_labels))
        X_tr, X_te   = X_norm[tr_idx], X_norm[te_idx]
        y_tr, y_te   = y_all[tr_idx],  y_all[te_idx]
        tr_groups     = city_labels[tr_idx]
        held_cities   = np.unique(city_labels[te_idx])
        used_cities   = np.unique(city_labels[tr_idx])
        logger.info(f"  Train cities ({len(used_cities)}): {list(used_cities)}")
        logger.info(f"  Test  cities ({len(held_cities)}):  {list(held_cities)}")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_norm, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        tr_groups = None

    logger.info(f"  Train: {len(X_tr)}  Test: {len(X_te)}")

    # ── Dummy baseline: always predict training mean ───────────────────────
    dummy_preds   = np.full(len(y_te), y_tr.mean())
    base_rmse     = float(np.sqrt(mean_squared_error(y_te, dummy_preds)))
    base_mae      = float(mean_absolute_error(y_te, dummy_preds))
    base_r2       = float(r2_score(y_te, dummy_preds))
    logger.info(
        f"  Baseline (mean predictor) → "
        f"RMSE={base_rmse:.4f}  MAE={base_mae:.4f}  R²={base_r2:.4f}"
    )

    # ── Inner CV respects city boundaries during tuning ───────────────────
    if tr_groups is not None:
        n_inner = min(CV_FOLDS, len(np.unique(tr_groups)))
        inner_cv_strategy = GroupKFold(n_splits=n_inner)
    else:
        inner_cv_strategy = CV_FOLDS

    results_store  = {}
    champion_name  = None
    champion_rmse  = float("inf")
    champion_model = None

    for algo_name, base_est in build_estimator_pool().items():
        logger.info(f"  Fitting: {algo_name} ...")
        try:
            if algo_name in SEARCH_GRIDS:
                tuner = GridSearchCV(
                    base_est,
                    SEARCH_GRIDS[algo_name],
                    cv=inner_cv_strategy,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    refit=True,
                )
                fit_kw = {"groups": tr_groups} if tr_groups is not None else {}
                tuner.fit(X_tr, y_tr, **fit_kw)
                fitted_est   = tuner.best_estimator_
                chosen_params = tuner.best_params_
                logger.info(f"    Best params: {chosen_params}")
            else:
                fitted_est   = base_est
                fitted_est.fit(X_tr, y_tr)
                chosen_params = {}

            preds  = fitted_est.predict(X_te)
            rmse   = float(np.sqrt(mean_squared_error(y_te, preds)))
            mae    = float(mean_absolute_error(y_te, preds))
            r2     = float(r2_score(y_te, preds))

            # Outer GroupKFold on full dataset
            if city_labels is not None:
                n_outer = min(CV_FOLDS, len(np.unique(city_labels)))
                outer_cv_strategy = GroupKFold(n_splits=n_outer)
                outer_kw = {"groups": city_labels}
            else:
                outer_cv_strategy = CV_FOLDS
                outer_kw = {}

            cv_raw  = cross_val_score(
                fitted_est, X_norm, y_all,
                cv=outer_cv_strategy,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                **outer_kw,
            )
            cv_rmse = float(-cv_raw.mean())
            cv_std  = float(cv_raw.std())

            skill_score = round(1.0 - (rmse / (base_rmse + 1e-9)), 4)

            results_store[algo_name] = {
                "rmse":               round(rmse, 4),
                "mae":                round(mae, 4),
                "r2":                 round(r2, 4),
                "cv_rmse":            round(cv_rmse, 4),
                "cv_std":             round(cv_std, 4),
                "skill_vs_baseline":  skill_score,
                "best_params":        chosen_params,
                "feature_importance": extract_importances(fitted_est, feat_names),
            }

            logger.info(
                f"    RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
                f"CV={cv_rmse:.4f}±{cv_std:.4f}  skill={skill_score:.3f}"
            )

            if rmse < champion_rmse:
                champion_rmse  = rmse
                champion_name  = algo_name
                champion_model = fitted_est

        except Exception as err:
            logger.error(f"    Failed: {algo_name}: {err}")

    if champion_model is None:
        raise RuntimeError("All estimators failed to fit!")

    # ── Persist best model ─────────────────────────────────────────────────
    with open(BEST_MODEL_PATH, "wb") as fh:
        pickle.dump(champion_model, fh)
    logger.info(
        f"\n  Best model: {champion_name}  RMSE={champion_rmse:.4f}  "
        f"vs baseline RMSE={base_rmse:.4f}"
    )

    # ── Persist full results ───────────────────────────────────────────────
    output_dict = {
        "best_model":    champion_name,
        "best_rmse":     round(champion_rmse, 4),
        "feature_names": feat_names,
        "baseline_rmse": round(base_rmse, 4),
        "baseline_mae":  round(base_mae, 4),
        "train_cities":  list(used_cities) if city_labels is not None else [],
        "test_cities":   list(held_cities) if city_labels is not None else [],
        "models":        results_store,
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(output_dict, fh, indent=2)
    logger.info(f"  Saved metrics → {METRICS_PATH}")

    return output_dict


if __name__ == "__main__":
    final_metrics = run_training(force=True)
    top = final_metrics["best_model"]
    print(f"\nBest: {top}  RMSE={final_metrics['best_rmse']}  "
          f"(baseline={final_metrics['baseline_rmse']})")
