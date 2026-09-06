"""Cleaning, imputation, encoding and feature engineering for the credit risk model.

Design goals:
  - A single fitted `Preprocessor` object handles both training and inference,
    so predict.py can never drift out of sync with train.py.
  - Sensible, explainable transforms (one-hot for low-cardinality categoricals,
    frequency encoding for high-cardinality ones) rather than opaque ones, since
    SHAP/rule outputs need to stay interpretable to a non-technical user.
  - Numeric missing values are left as NaN, not mean-imputed: LightGBM (the
    champion) splits on missingness natively, and on this dataset the strongest
    predictors (EXT_SOURCE_1/2/3) are 40-55% missing, so imputing them destroys
    signal. A `<col>_MISSING` indicator is emitted for every numeric column that
    was materially missing at fit time so "value is absent" stays a usable
    feature. The Logistic Regression baseline gets its own median-impute +
    scale pipeline in train.py (linear models can't take NaN).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src.utils.config import ID_COL, TARGET_COL
from src.utils.logger import get_logger

log = get_logger(__name__)

ANOMALY_DAYS_EMPLOYED = 365243
HIGH_CARDINALITY_THRESHOLD = 15
MISSING_INDICATOR_MIN_RATE = 0.02  # emit a <col>_MISSING flag for numerics at least this missing at fit time

_INVALID_FEATURE_CHARS = re.compile(r"[^0-9A-Za-z_]+")


def _sanitize_feature_name(name: str) -> str:
    """LightGBM rejects JSON special characters (commas, spaces, brackets, ...) in
    feature names, which one-hot category labels like 'Spouse, partner' contain."""
    return _INVALID_FEATURE_CHARS.sub("_", name).strip("_")


@dataclass
class Preprocessor:
    numeric_medians: dict[str, float] = field(default_factory=dict)
    categorical_modes: dict[str, str] = field(default_factory=dict)
    low_card_categories: dict[str, list[str]] = field(default_factory=dict)   # -> one-hot
    high_card_frequencies: dict[str, dict[str, float]] = field(default_factory=dict)  # -> frequency encoding
    feature_columns: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    low_card_cols: list[str] = field(default_factory=list)
    high_card_cols: list[str] = field(default_factory=list)
    missing_indicator_cols: list[str] = field(default_factory=list)  # numerics that get a <col>_MISSING flag

    # ---- feature engineering shared by fit & transform -----------------------------------
    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # The real dataset's famous placeholder for "not currently employed".
        df["DAYS_EMPLOYED_ANOMALY"] = (df.get("DAYS_EMPLOYED") == ANOMALY_DAYS_EMPLOYED).astype(int)
        if "DAYS_EMPLOYED" in df.columns:
            df.loc[df["DAYS_EMPLOYED"] == ANOMALY_DAYS_EMPLOYED, "DAYS_EMPLOYED"] = np.nan

        if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
            df["EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
        if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
            df["GOODS_PRICE_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)

        ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
        if ext_cols:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

        return df

    def _split_columns(self, df: pd.DataFrame) -> None:
        drop_cols = {ID_COL, TARGET_COL}
        candidate_cols = [c for c in df.columns if c not in drop_cols]

        self.numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in candidate_cols if c not in self.numeric_cols]

        self.low_card_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= HIGH_CARDINALITY_THRESHOLD]
        self.high_card_cols = [c for c in cat_cols if c not in self.low_card_cols]

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = self._engineer(df)
        self._split_columns(df)

        for col in self.numeric_cols:
            self.numeric_medians[col] = float(df[col].median()) if df[col].notna().any() else 0.0

        self.missing_indicator_cols = [
            col for col in self.numeric_cols
            if float(df[col].isna().mean()) >= MISSING_INDICATOR_MIN_RATE
        ]

        for col in self.low_card_cols:
            mode = df[col].mode(dropna=True)
            self.categorical_modes[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            self.low_card_categories[col] = sorted(df[col].dropna().astype(str).unique().tolist())

        for col in self.high_card_cols:
            mode = df[col].mode(dropna=True)
            self.categorical_modes[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            freq = df[col].astype(str).value_counts(normalize=True)
            self.high_card_frequencies[col] = freq.to_dict()

        self.feature_columns = self._build(df, fitting=True).columns.tolist()
        return self

    def _build(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []

        if self.numeric_cols:
            # NaN is kept deliberately - LightGBM splits on it; the LR pipeline imputes.
            numeric = {
                col: pd.to_numeric(df[col], errors="coerce")
                for col in self.numeric_cols
            }
            blocks.append(pd.DataFrame(numeric, index=df.index))

        if self.missing_indicator_cols:
            indicators = {
                f"{col}_MISSING": (
                    pd.to_numeric(df[col], errors="coerce").isna().astype(int)
                    if col in df.columns else pd.Series(1, index=df.index)
                )
                for col in self.missing_indicator_cols
            }
            blocks.append(pd.DataFrame(indicators, index=df.index))

        for col in self.low_card_cols:
            filled = df[col].astype(str).fillna(self.categorical_modes.get(col, "Unknown"))
            filled = filled.where(filled.isin(self.low_card_categories.get(col, [])), "Other")
            dummies = pd.get_dummies(filled, prefix=col)
            if not fitting:
                expected = [f"{col}_{v}" for v in self.low_card_categories.get(col, [])]
                dummies = dummies.reindex(columns=expected, fill_value=0)
            blocks.append(dummies)

        if self.high_card_cols:
            freq = {
                f"{col}_FREQ": df[col].astype(str).map(self.high_card_frequencies.get(col, {})).fillna(0.0)
                for col in self.high_card_cols
            }
            blocks.append(pd.DataFrame(freq, index=df.index))

        out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
        out.columns = [_sanitize_feature_name(c) for c in out.columns]
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._engineer(df)
        out = self._build(df, fitting=False)
        return out.reindex(columns=self.feature_columns, fill_value=0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path) -> None:
        joblib.dump(self, path)
        log.info("Saved preprocessor to %s (%d output features)", path, len(self.feature_columns))

    @staticmethod
    def load(path) -> "Preprocessor":
        return joblib.load(path)
