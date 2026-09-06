"""Explainable AI layer (Part 4 of the spec): SHAP on top of the trained LightGBM model.

SHAP was chosen over LIME because:
  - `TreeExplainer` is exact and fast for tree ensembles (no local surrogate
    sampling needed), which matters for a "lightweight platform" serving
    interactive per-applicant explanations in the UI.
  - It naturally gives both a global feature-importance view (for the model
    tab) and a per-prediction, signed, additive breakdown (for the
    Explainability tab) from the same explainer object.

Falls back gracefully to permutation-based `shap.Explainer` if the champion
model turned out to be the Logistic Regression baseline instead of LightGBM.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

from src.ml.predict import RiskModel


class RiskExplainer:
    def __init__(self, risk_model: RiskModel | None = None) -> None:
        self.risk_model = risk_model or RiskModel()
        model = self.risk_model.model
        # train.py wraps the champion in CalibratedClassifierCV (isotonic, cv='prefit')
        # to fix probability scaling; SHAP needs the raw tree booster underneath it,
        # not the calibration wrapper, to compute tree-based attributions.
        if isinstance(model, CalibratedClassifierCV):
            model = model.calibrated_classifiers_[0].estimator
        if isinstance(model, lgb.LGBMClassifier):
            self.explainer = shap.TreeExplainer(model)
            self._is_tree = True
        else:
            # Logistic Regression fallback: linear explainer is exact and cheap.
            self.explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(
                np.zeros((1, len(self.risk_model.preprocessor.feature_columns)))
            ))
            self._is_tree = False

    def global_importance(self, background_df: pd.DataFrame, max_display: int = 15) -> pd.DataFrame:
        X = self.risk_model.transform_features(background_df)
        shap_values = self._shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = (
            pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(max_display)
            .reset_index(drop=True)
        )
        return importance

    def explain_one(self, applicant_row: pd.DataFrame, top_n: int = 8) -> dict:
        """Returns the top contributing features (with signed impact) for a single applicant."""
        X = self.risk_model.transform_features(applicant_row)
        shap_values = self._shap_values(X)[0]
        base_value = self._base_value()

        contributions = pd.DataFrame({
            "feature": X.columns,
            "value": X.iloc[0].values,
            "shap_value": shap_values,
        })
        contributions["direction"] = np.where(contributions["shap_value"] >= 0, "increases risk", "decreases risk")
        contributions = contributions.reindex(
            contributions["shap_value"].abs().sort_values(ascending=False).index
        ).head(top_n)

        return {
            "base_value": float(base_value),
            "predicted_probability": float(self.risk_model.predict_proba(applicant_row).iloc[0]),
            "top_features": contributions.to_dict(orient="records"),
        }

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.explainer.shap_values(X)
        if isinstance(raw, list):  # some SHAP/LightGBM combos return [class0, class1]
            raw = raw[1]
        return np.asarray(raw)

    def _base_value(self) -> float:
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(np.asarray(ev).reshape(-1)[-1])
        return float(ev)


if __name__ == "__main__":
    from src.data.loader import load_train_df

    df = load_train_df().drop(columns=["TARGET"])
    explainer = RiskExplainer()
    print(explainer.global_importance(df.sample(2000, random_state=42)))
    print(explainer.explain_one(df.head(1)))
