"""Exploratory data analysis for the Home Credit application dataset.

Runnable as a plain script (`python notebooks/eda.py`) or copy-pasted cell by
cell into notebooks/eda.ipynb. Produces:
  - a printed dataset summary + feature categorization
  - a data-quality / missing-value report
  - >=5 business insight charts saved to notebooks/figures/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_train_df
from src.utils.config import TARGET_COL

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def dataset_summary(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("1. DATASET SUMMARY")
    print("=" * 80)
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(4)}")
    imbalance_ratio = (df[TARGET_COL] == 0).sum() / max((df[TARGET_COL] == 1).sum(), 1)
    print(f"Class imbalance ratio (non-default : default): {imbalance_ratio:.1f} : 1")


def feature_categorization(df: pd.DataFrame) -> dict[str, list[str]]:
    print("\n" + "=" * 80)
    print("2. FEATURE CATEGORIZATION")
    print("=" * 80)
    groups: dict[str, list[str]] = {
        "Identifiers": ["SK_ID_CURR"],
        "Target": ["TARGET"],
        "Demographics": [c for c in df.columns if c in (
            "CODE_GENDER", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS",
            "NAME_EDUCATION_TYPE", "DAYS_BIRTH", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
        )],
        "Financials": [c for c in df.columns if c.startswith("AMT_")],
        "Employment / Org": [c for c in df.columns if c in (
            "DAYS_EMPLOYED", "ORGANIZATION_TYPE", "NAME_INCOME_TYPE",
        )],
        "External credit scores": [c for c in df.columns if c.startswith("EXT_SOURCE")],
        "Building / housing info (normalized)": [
            c for c in df.columns if c.endswith(("_AVG", "_MODE", "_MEDI"))
        ],
        "Document flags": [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")],
        "Region / contact flags": [c for c in df.columns if c.startswith(
            ("REGION_", "REG_", "LIVE_", "FLAG_")
        ) and not c.startswith("FLAG_DOCUMENT_") and not c.startswith("FLAG_OWN")],
        "Credit bureau enquiries": [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")],
        "Social circle": [c for c in df.columns if c.startswith(("OBS_", "DEF_"))],
    }
    categorized = {c for cols in groups.values() for c in cols}
    groups["Other"] = [c for c in df.columns if c not in categorized]
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} columns")
    return groups


def data_quality(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 80)
    print("3. DATA QUALITY / MISSING VALUES")
    print("=" * 80)
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing[missing > 0].head(15)
    print("Top columns by % missing:")
    print(top_missing.round(1))
    print(f"\nColumns with >50% missing: {(missing > 50).sum()}")
    print(f"Columns with no missing values: {(missing == 0).sum()}")

    if "DAYS_EMPLOYED" in df.columns:
        anomaly_rate = (df["DAYS_EMPLOYED"] == 365243).mean() * 100
        print(f"\nData quality issue: DAYS_EMPLOYED == 365243 anomaly placeholder in {anomaly_rate:.1f}% of rows "
              "(represents 'not currently employed' rather than a real day count).")

    fig, ax = plt.subplots(figsize=(9, 5))
    top_missing.head(12).iloc[::-1].plot(kind="barh", ax=ax, color="#c0392b")
    ax.set_xlabel("% missing")
    ax.set_title("Top columns by missing-value rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_missing_values.png", dpi=110)
    plt.close(fig)
    return missing


def business_insights(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("4. BUSINESS INSIGHTS (>=5, with charts)")
    print("=" * 80)

    # Insight 1: default rate by education level
    fig, ax = plt.subplots(figsize=(8, 5))
    rate = df.groupby("NAME_EDUCATION_TYPE")[TARGET_COL].mean().sort_values() * 100
    rate.plot(kind="barh", ax=ax, color="#2980b9")
    ax.set_xlabel("Default rate (%)")
    ax.set_title("Insight 1: Default rate by education level")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_default_by_education.png", dpi=110)
    plt.close(fig)
    print(f"1. Default rate by education:\n{rate.round(2)}")

    # Insight 2: default rate by age bucket
    age_years = -df["DAYS_BIRTH"] / 365.25
    age_bucket = pd.cut(age_years, bins=[18, 25, 35, 45, 55, 65, 100],
                         labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    fig, ax = plt.subplots(figsize=(8, 5))
    rate2 = df.groupby(age_bucket, observed=True)[TARGET_COL].mean() * 100
    rate2.plot(kind="bar", ax=ax, color="#e67e22")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Insight 2: Default rate by age bucket")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_default_by_age.png", dpi=110)
    plt.close(fig)
    print(f"\n2. Default rate by age bucket:\n{rate2.round(2)}")

    # Insight 3: EXT_SOURCE_2 vs default (strongest external signal)
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="EXT_SOURCE_2", by=TARGET_COL, ax=ax)
    ax.set_title("Insight 3: EXT_SOURCE_2 distribution by default status")
    ax.set_xlabel("TARGET (1 = defaulted)")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_extsource2_by_target.png", dpi=110)
    plt.close(fig)
    corr = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", TARGET_COL]].corr()[TARGET_COL]
    print(f"\n3. Correlation of external scores with TARGET:\n{corr.round(3)}")

    # Insight 4: credit-to-income ratio vs default
    cir = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)).clip(upper=20)
    cir_bucket = pd.qcut(cir, 5, duplicates="drop")
    fig, ax = plt.subplots(figsize=(8, 5))
    rate4 = df.groupby(cir_bucket, observed=True)[TARGET_COL].mean() * 100
    rate4.plot(kind="bar", ax=ax, color="#8e44ad")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Insight 4: Default rate by credit-to-income ratio quintile")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_default_by_credit_income_ratio.png", dpi=110)
    plt.close(fig)
    print(f"\n4. Default rate by credit/income quintile:\n{rate4.round(2)}")

    # Insight 5: default rate by contract type & family status
    fig, ax = plt.subplots(figsize=(8, 5))
    rate5 = df.groupby("NAME_FAMILY_STATUS")[TARGET_COL].mean().sort_values() * 100
    rate5.plot(kind="barh", ax=ax, color="#16a085")
    ax.set_xlabel("Default rate (%)")
    ax.set_title("Insight 5: Default rate by family status")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_default_by_family_status.png", dpi=110)
    plt.close(fig)
    print(f"\n5. Default rate by family status:\n{rate5.round(2)}")

    # Insight 6 (bonus): employment anomaly vs default
    if "DAYS_EMPLOYED" in df.columns:
        anomaly = (df["DAYS_EMPLOYED"] == 365243)
        rate6 = df.groupby(anomaly)[TARGET_COL].mean() * 100
        print(f"\n6. (bonus) Default rate for DAYS_EMPLOYED anomaly (not employed) vs normal:\n{rate6.round(2)}")


def run() -> None:
    df = load_train_df()
    dataset_summary(df)
    feature_categorization(df)
    data_quality(df)
    business_insights(df)
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run()
