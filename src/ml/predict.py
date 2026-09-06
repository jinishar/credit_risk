"""Inference: score one or more applicants and map probability -> risk band."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.data.preprocessor import Preprocessor
from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH, RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD
from src.utils.helpers import risk_band


def _unwrap_calibration(model):
    """Split `train.py`'s `CalibratedClassifierCV(isotonic, cv='prefit')` into its
    base estimator and the fitted isotonic calibrator, so `predict_proba` can
    apply the calibrator explicitly to the base model's positive-class
    probability.

    Why not just call `CalibratedClassifierCV.predict_proba`: the isotonic step
    is fitted on `predict_proba` output ([0, 1]), but since scikit-learn 1.7 the
    wrapper feeds it the base estimator's `decision_function` margins instead
    (LGBMClassifier now exposes one), and isotonic's out-of-bounds clip then
    collapses every negative-margin applicant — ~99% of them — to exactly 0.0.
    """
    if isinstance(model, CalibratedClassifierCV):
        calibrated = model.calibrated_classifiers_[0]
        return calibrated.estimator, calibrated.calibrators[0]
    return model, None


class RiskModel:
    """Loads the saved champion model + preprocessor once and serves predictions."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {MODEL_PATH.parent}. Run `python -m src.ml.train` first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor: Preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
        self._base, self._calibrator = _unwrap_calibration(self.model)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(df)
        proba = self._base.predict_proba(X)[:, 1]
        if self._calibrator is not None:
            proba = self._calibrator.predict(proba)
        return pd.Series(proba, index=df.index, name="default_probability")

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        bands = proba.apply(lambda p: risk_band(p, RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD))
        result = df.copy()
        result["default_probability"] = proba
        result["risk_band"] = bands
        return result

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exposes the fully-engineered feature matrix (used by explain.py)."""
        return self.preprocessor.transform(df)


if __name__ == "__main__":
    from src.data.loader import load_train_df

    model = RiskModel()
    sample = load_train_df().drop(columns=["TARGET"]).head(5)
    print(model.score(sample)[["SK_ID_CURR", "default_probability", "risk_band"]])
