"""Inference: score one or more applicants and map probability -> risk band."""
from __future__ import annotations

import joblib
import pandas as pd

from src.data.preprocessor import Preprocessor
from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH, RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD
from src.utils.helpers import risk_band


class RiskModel:
    """Loads the saved champion model + preprocessor once and serves predictions."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {MODEL_PATH.parent}. Run `python -m src.ml.train` first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor: Preprocessor = Preprocessor.load(PREPROCESSOR_PATH)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)[:, 1]
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
