"""Generates a schema-accurate SYNTHETIC Home Credit dataset.

No real application_train.csv was available in this environment (only the
220-row HomeCredit_columns_description.csv column dictionary was provided).
This module parses that real dictionary to get the exact 121 application-table
column names, then fabricates statistically realistic values for each one
(known category sets, missingness rates, and target correlations taken from
the well-documented public Home Credit Default Risk competition) so every
downstream layer -- EDA, the ML model, SHAP, the rule engine and the NL-to-SQL
chatbot -- has something structurally real to work with.

IMPORTANT: this is NOT real applicant data. Drop a real `application_train.csv`
(and optionally `application_test.csv`) from
https://www.kaggle.com/competitions/home-credit-default-risk/data into
DATA_DIR and the loader (src/data/loader.py) will use those instead --
no code changes required anywhere else in the pipeline.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import (
    COLUMN_DICTIONARY_CSV,
    N_SYNTHETIC_ROWS,
    RANDOM_SEED,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Real Home Credit category sets for the fields the chatbot/EDA/rules lean on most.
CATEGORY_VALUES: dict[str, list[str]] = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "CODE_GENDER": ["F", "M", "XNA"],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_TYPE_SUITE": [
        "Unaccompanied", "Family", "Spouse, partner", "Children",
        "Other_A", "Other_B", "Group of people",
    ],
    "NAME_INCOME_TYPE": [
        "Working", "Commercial associate", "Pensioner", "State servant",
        "Unemployed", "Student", "Businessman", "Maternity leave",
    ],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special", "Higher education",
        "Incomplete higher", "Lower secondary", "Academic degree",
    ],
    "NAME_FAMILY_STATUS": [
        "Married", "Single / not married", "Civil marriage",
        "Separated", "Widow",
    ],
    "NAME_HOUSING_TYPE": [
        "House / apartment", "With parents", "Municipal apartment",
        "Rented apartment", "Office apartment", "Co-op apartment",
    ],
    "OCCUPATION_TYPE": [
        "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
        "High skill tech staff", "Accountants", "Medicine staff",
        "Security staff", "Cooking staff", "Cleaning staff",
        "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
        "Secretaries", "Realty agents", "HR staff", "IT staff",
    ],
    "WEEKDAY_APPR_PROCESS_START": [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY",
        "SATURDAY", "SUNDAY",
    ],
    "ORGANIZATION_TYPE": [
        "Business Entity Type 3", "Self-employed", "Other", "Medicine",
        "Business Entity Type 2", "Government", "School",
        "Business Entity Type 1", "Trade: type 7", "Kindergarten",
        "Construction", "Transport: type 4", "Military", "Security Ministries",
        "Trade: type 3", "Housing", "Industry: type 11", "Bank", "Agriculture",
        "Police", "XNA",
    ],
    "FONDKAPREMONT_MODE": [
        "reg oper account", "org spec account", "reg oper spec account", "not specified",
    ],
    "HOUSETYPE_MODE": ["block of flats", "terraced house", "specific housing"],
    "WALLSMATERIAL_MODE": [
        "Panel", "Stone, brick", "Block", "Wooden", "Mixed", "Monolithic", "Others",
    ],
    "EMERGENCYSTATE_MODE": ["No", "Yes"],
}

# Missingness rates observed in the real dataset for its highest-missing fields.
MISSING_RATES: dict[str, float] = {
    "EXT_SOURCE_1": 0.56,
    "EXT_SOURCE_3": 0.20,
    "OWN_CAR_AGE": 0.66,
    "OCCUPATION_TYPE": 0.31,
    "FONDKAPREMONT_MODE": 0.68,
    "HOUSETYPE_MODE": 0.50,
    "WALLSMATERIAL_MODE": 0.51,
    "EMERGENCYSTATE_MODE": 0.47,
    "NAME_TYPE_SUITE": 0.004,
    "AMT_ANNUITY": 0.0004,
    "AMT_GOODS_PRICE": 0.001,
    "CNT_FAM_MEMBERS": 0.0001,
    "DAYS_LAST_PHONE_CHANGE": 0.0001,
}
BUILDING_INFO_MISSING_RATE = 0.55  # shared by the *_AVG/_MODE/_MEDI apartment block

# Categorical fields whose distribution shifts with TARGET, so EDA finds genuine,
# pronounced (not just noise-level) default-rate differences across categories --
# ordered from safest to riskiest within each field, per well-documented real-world
# credit risk patterns for this dataset.
CONDITIONAL_CATEGORY_PROBS: dict[str, tuple[list[str], list[float], list[float]]] = {
    "NAME_EDUCATION_TYPE": (
        ["Academic degree", "Higher education", "Secondary / secondary special",
         "Incomplete higher", "Lower secondary"],
        [0.03, 0.30, 0.55, 0.08, 0.04],   # target = 0 (repaid)
        [0.005, 0.12, 0.55, 0.18, 0.145],  # target = 1 (defaulted)
    ),
    "NAME_INCOME_TYPE": (
        ["Working", "Commercial associate", "Pensioner", "State servant",
         "Unemployed", "Student", "Businessman", "Maternity leave"],
        [0.52, 0.18, 0.18, 0.07, 0.01, 0.01, 0.02, 0.01],
        [0.55, 0.15, 0.08, 0.03, 0.10, 0.02, 0.01, 0.06],
    ),
    "NAME_FAMILY_STATUS": (
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        [0.58, 0.13, 0.10, 0.09, 0.10],
        [0.40, 0.22, 0.18, 0.12, 0.08],
    ),
    "NAME_HOUSING_TYPE": (
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        [0.90, 0.04, 0.03, 0.015, 0.01, 0.005],
        [0.70, 0.12, 0.05, 0.09, 0.02, 0.02],
    ),
}


def _conditional_choice(name: str, n: int, rng: np.random.Generator, target: np.ndarray) -> pd.Series:
    categories, p0, p1 = CONDITIONAL_CATEGORY_PROBS[name]
    out = np.empty(n, dtype=object)
    mask1 = target == 1
    out[~mask1] = rng.choice(categories, size=(~mask1).sum(), p=p0)
    out[mask1] = rng.choice(categories, size=mask1.sum(), p=p1)
    return pd.Series(out)

_BUILDING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")
_BUILDING_STEMS = (
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
    "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
    "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA", "TOTALAREA",
)


def _load_application_columns() -> list[str]:
    """Read the real column dictionary and return the ordered application-table columns."""
    columns: list[str] = []
    with open(COLUMN_DICTIONARY_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table = (row.get("Table") or "").strip()
            col = (row.get("Row") or "").strip()
            if table.startswith("application_") and col:
                columns.append(col)
    if not columns:
        raise RuntimeError(f"No application_* columns found in {COLUMN_DICTIONARY_CSV}")
    return columns


def _is_building_col(name: str) -> bool:
    return name.endswith(_BUILDING_SUFFIXES) and any(name.startswith(stem) for stem in _BUILDING_STEMS)


def _apply_missingness(series: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    if rate <= 0:
        return series
    mask = rng.random(len(series)) < rate
    series = series.astype("object")
    series[mask] = np.nan
    return series


def _gen_column(name: str, n: int, rng: np.random.Generator, target: np.ndarray) -> pd.Series:
    """Generate one synthetic column, correlating a handful of signal columns with `target`."""

    # --- explicit, meaningful/known columns -------------------------------------------------
    if name == "CODE_GENDER":
        return pd.Series(rng.choice(["F", "M"], size=n, p=[0.66, 0.34]))
    if name in CONDITIONAL_CATEGORY_PROBS:
        return _conditional_choice(name, n, rng, target)
    if name in CATEGORY_VALUES:
        return pd.Series(rng.choice(CATEGORY_VALUES[name], size=n))

    if name == "DAYS_BIRTH":
        # Age 20-69, younger applicants skew slightly riskier.
        age_years = rng.uniform(20, 69, n) - target * rng.uniform(0, 6, n)
        return pd.Series((-age_years * 365.25).astype(int))

    if name == "DAYS_EMPLOYED":
        days = -rng.exponential(2000, n).astype(int) - 30
        anomaly_mask = rng.random(n) < 0.18  # the famous 365243 "not employed" placeholder
        out = days.astype(object)
        out[anomaly_mask] = 365243
        return pd.Series(out)

    if name in ("DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"):
        return pd.Series(-rng.exponential(2500, n).astype(int))

    if name == "OWN_CAR_AGE":
        return pd.Series(rng.exponential(6, n).round(1))

    if name.startswith("EXT_SOURCE_"):
        # Strong protective signal: higher score -> lower default risk.
        base = rng.beta(2.5, 2.5, n)
        adj = np.clip(base - target * rng.uniform(0.05, 0.25, n), 0, 1)
        return pd.Series(adj)

    if name == "AMT_INCOME_TOTAL":
        return pd.Series(np.round(rng.lognormal(11.8, 0.5, n), 0))

    if name == "AMT_CREDIT":
        income = rng.lognormal(11.8, 0.5, n)
        return pd.Series(np.round(income * rng.uniform(1.5, 6.0, n), 0))

    if name == "AMT_ANNUITY":
        return pd.Series(np.round(rng.lognormal(9.8, 0.4, n), 1))

    if name == "AMT_GOODS_PRICE":
        return pd.Series(np.round(rng.lognormal(11.7, 0.5, n), 0))

    if name == "CNT_CHILDREN":
        return pd.Series(rng.poisson(0.4, n))

    if name == "CNT_FAM_MEMBERS":
        return pd.Series(rng.poisson(0.4, n) + 1.0)

    if name == "REGION_POPULATION_RELATIVE":
        return pd.Series(rng.beta(2, 8, n).round(6))

    if name in ("REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"):
        return pd.Series(rng.choice([1, 2, 3], size=n, p=[0.15, 0.55, 0.30]))

    if name == "HOUR_APPR_PROCESS_START":
        return pd.Series(rng.normal(12, 3, n).clip(0, 23).astype(int))

    if name.startswith("FLAG_DOCUMENT_"):
        # FLAG_DOCUMENT_3 is near-universal in the real data; the rest are rare.
        p = 0.71 if name == "FLAG_DOCUMENT_3" else 0.02
        return pd.Series(rng.binomial(1, p, n))

    if name.startswith("FLAG_") or name.startswith("REG_") or name.startswith("LIVE_") or name.startswith("NFLAG"):
        return pd.Series(rng.binomial(1, 0.15, n))

    if name.startswith("OBS_") or name.startswith("DEF_"):
        return pd.Series(rng.poisson(1.5 if name.startswith("OBS_") else 0.1, n))

    if name.startswith("AMT_REQ_CREDIT_BUREAU_"):
        return pd.Series(rng.poisson(0.3, n))

    if _is_building_col(name):
        return pd.Series(rng.beta(2, 5, n).round(4))

    if name in ("SK_ID_CURR",):
        return pd.Series(np.arange(100_000, 100_000 + n))

    # --- generic fallback based on the DAYS_ / CNT_ / AMT_ naming convention ----------------
    if name.startswith("DAYS_"):
        return pd.Series(-rng.exponential(1500, n).astype(int))
    if name.startswith("CNT_"):
        return pd.Series(rng.poisson(1, n))
    if name.startswith("AMT_"):
        return pd.Series(np.round(rng.lognormal(9, 0.6, n), 1))

    # Unknown text/categorical field with no known value set -> small generic category set.
    return pd.Series(rng.choice(["Type_A", "Type_B", "Type_C"], size=n))


def generate(n_rows: int = N_SYNTHETIC_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the full synthetic application-level dataframe, TARGET included."""
    rng = np.random.default_rng(seed)
    columns = [c for c in _load_application_columns() if c not in ("SK_ID_CURR", TARGET_COL)]

    log.warning(
        "No real Kaggle CSV found - generating %d rows of SYNTHETIC data "
        "matching the Home Credit schema. Replace with the real "
        "application_train.csv for production use.",
        n_rows,
    )

    target = rng.binomial(1, 0.081, n_rows)  # matches the real ~8.07% default rate

    data: dict[str, pd.Series] = {"SK_ID_CURR": pd.Series(np.arange(100_000, 100_000 + n_rows))}
    for col in columns:
        series = _gen_column(col, n_rows, rng, target)
        rate = MISSING_RATES.get(col, BUILDING_INFO_MISSING_RATE if _is_building_col(col) else 0.0)
        data[col] = _apply_missingness(series, rate, rng)

    data[TARGET_COL] = pd.Series(target)
    df = pd.DataFrame(data)

    # Feature/target interaction nudges so the model has genuine, non-random signal.
    credit_income_ratio = pd.to_numeric(df["AMT_CREDIT"], errors="coerce") / pd.to_numeric(
        df["AMT_INCOME_TOTAL"], errors="coerce"
    ).replace(0, np.nan)
    high_leverage = (credit_income_ratio > credit_income_ratio.quantile(0.85)).fillna(False)
    flip_to_default = high_leverage & (rng.random(n_rows) < 0.35) & (df[TARGET_COL] == 0)
    df.loc[flip_to_default, TARGET_COL] = 1

    return df


def generate_and_save(n_rows: int = N_SYNTHETIC_ROWS, seed: int = RANDOM_SEED) -> tuple[Path, Path]:
    """Generate train (with TARGET) and test (without TARGET) CSVs and write them to DATA_DIR."""
    df = generate(n_rows=n_rows, seed=seed)

    test_size = max(1, int(n_rows * 0.1))
    test_df = df.sample(n=test_size, random_state=seed).drop(columns=[TARGET_COL]).reset_index(drop=True)
    train_df = df.drop(index=test_df.index, errors="ignore").reset_index(drop=True) if False else df

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    log.info("Wrote synthetic %s (%d rows) and %s (%d rows)", TRAIN_CSV, len(train_df), TEST_CSV, len(test_df))
    return TRAIN_CSV, TEST_CSV


if __name__ == "__main__":
    generate_and_save()
