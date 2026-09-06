"""Trains the credit default model.

Two models are trained and compared:
  - Logistic Regression (class_weight='balanced') as an interpretable baseline.
  - LightGBM (scale_pos_weight tuned to the observed imbalance) as the primary
    model - chosen because it (a) handles the mixed numeric/categorical,
    high-missingness tabular data natively, (b) trains in seconds on this
    dataset size ("lightweight platform"), and (c) is directly explainable
    via SHAP's fast TreeExplainer.

The model with the higher validation ROC-AUC is saved as the production model.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_train_df
from src.data.preprocessor import Preprocessor
from src.ml.evaluate import evaluate
from src.utils.config import (
    FEATURE_LIST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    TARGET_COL,
)
from src.utils.docker_utils import ensure_runtime_dirs
from src.utils.helpers import save_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def train() -> dict:
    ensure_runtime_dirs()
    df = load_train_df()

    y = df[TARGET_COL].astype(int).values
    preprocessor = Preprocessor()
    X = preprocessor.fit_transform(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    log.info("Train: %d rows | Val: %d rows | positive rate: %.3f", len(X_train), len(X_val), y.mean())

    # --- Baseline: Logistic Regression --------------------------------------------------
    # The preprocessor leaves NaN in place (for LightGBM); a linear model needs
    # imputation + scaling, so the baseline carries its own pipeline. Without
    # scaling, lbfgs does not converge on this feature set.
    baseline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(y_val, baseline.predict_proba(X_val)[:, 1])
    log.info("Logistic Regression baseline: ROC-AUC=%.4f PR-AUC=%.4f",
              baseline_metrics["roc_auc"], baseline_metrics["pr_auc"])

    # --- Primary model: LightGBM ---------------------------------------------------------
    # Imbalance note: scale_pos_weight / is_unbalance were both tested and rejected -
    # the reweighted objective spikes validation AUC on the first 1-3 trees and then
    # plateaus, so early stopping cuts training off immediately (AUC ~0.73). Plain
    # LightGBM trains to convergence (AUC ~0.76); imbalance is instead handled
    # downstream via threshold-independent metrics (ROC-AUC / PR-AUC) and the
    # isotonic calibration step below, which anchors probabilities to the true
    # ~8% base rate. observed_pos_weight is recorded in metrics.json for reference.
    observed_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    lgbm = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=34,
        max_depth=-1,
        min_child_samples=70,
        subsample=0.8,
        subsample_freq=1,          # subsample is ignored unless freq > 0
        colsample_bytree=0.8,
        reg_alpha=0.04,
        reg_lambda=0.07,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )
    log.info("LightGBM stopped at %d trees", lgbm.best_iteration_ or lgbm.n_estimators)
    lgbm_metrics = evaluate(y_val, lgbm.predict_proba(X_val)[:, 1])
    log.info("LightGBM: ROC-AUC=%.4f PR-AUC=%.4f", lgbm_metrics["roc_auc"], lgbm_metrics["pr_auc"])

    champion_name, champion_model, champion_metrics = (
        ("lightgbm", lgbm, lgbm_metrics)
        if lgbm_metrics["roc_auc"] >= baseline_metrics["roc_auc"]
        else ("logistic_regression", baseline, baseline_metrics)
    )
    log.info("Champion model (pre-calibration): %s", champion_name)

    # LightGBM's raw scores are well-ranked but not probabilities on the true scale
    # (a severe ~8/92 imbalance pushes the raw positive-class scores low, so the
    # "High" band >0.5 would rarely fire). Isotonic calibration on the held-out
    # validation fold remaps scores to the applicants' true empirical default rate
    # without retraining, making the Low/Medium/High risk bands meaningful. This is
    # also the pipeline's primary class-imbalance control (see the LightGBM note).
    calibrated_model = CalibratedClassifierCV(champion_model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_val, y_val)
    calibrated_metrics = evaluate(y_val, calibrated_model.predict_proba(X_val)[:, 1])
    log.info(
        "Calibrated %s: ROC-AUC=%.4f PR-AUC=%.4f (probabilities remapped to true scale)",
        champion_name, calibrated_metrics["roc_auc"], calibrated_metrics["pr_auc"],
    )
    champion_model, champion_metrics = calibrated_model, calibrated_metrics

    import joblib

    joblib.dump(champion_model, MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)
    save_json(preprocessor.feature_columns, FEATURE_LIST_PATH)
    save_json(
        {
            "champion_model": champion_name,
            "observed_pos_weight": float(observed_pos_weight),
            "imbalance_strategy": (
                "Logistic Regression baseline: class_weight='balanced'. "
                "LightGBM champion: scale_pos_weight / is_unbalance were tested and "
                "rejected (they collapse early stopping to 1-3 trees on this data); "
                "imbalance is instead handled with threshold-independent metrics "
                "(ROC-AUC, PR-AUC) and isotonic probability calibration to the true "
                "~8% base rate, with the Low/Medium/High bands read off the calibrated "
                "probability rather than a fixed 0.5 cutoff. LightGBM also splits on "
                "missing values natively and gets <col>_MISSING indicator features "
                "instead of mean-imputed ones. SMOTE not used - see README limitations."
            ),
            "logistic_regression": baseline_metrics,
            "lightgbm": lgbm_metrics,
            "champion": champion_metrics,
            "calibration": "isotonic (CalibratedClassifierCV, cv='prefit' on the validation fold) - "
                            "anchors the ranking scores to the true empirical default rate so the "
                            "Low/Medium/High risk bands are meaningful; also the primary imbalance control.",
            "n_features": len(preprocessor.feature_columns),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
        },
        METRICS_PATH,
    )
    log.info("Saved model artifacts to %s", MODEL_PATH.parent)
    return champion_metrics


if __name__ == "__main__":
    train()
