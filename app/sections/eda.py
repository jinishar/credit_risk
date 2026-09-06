"""EDA section: dataset summary, data quality, and the key business-insight charts.

Each insight gets a form matched to its job and its own single hue — imbalance
ratio bar, magnitude bars, change-over-an-ordered-axis areas, an identity
grouped bar — so the page reads as a varied dashboard, not a stack of bars.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data, theme, ui

_AGE_BINS = [18, 25, 35, 45, 55, 65, 100]
_AGE_LABELS = ["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
C = theme.CATEGORICAL


def render() -> None:
    df = data.dataset()
    ui.section_header(
        "Exploratory Data Analysis",
        "Home Credit applications — demographics, financials, and data quality.",
        kicker="Dataset overview",
    )

    default_rate = df["TARGET"].mean() * 100
    imbalance = (df["TARGET"] == 0).sum() / max((df["TARGET"] == 1).sum(), 1)
    ui.kpi_row([
        ui.kpi_card("Applicants", f"{len(df):,}", accent=C[0]),
        ui.kpi_card("Features", f"{df.shape[1] - 2}", accent=C[2]),
        ui.kpi_card("Default rate", f"{default_rate:.1f}%", accent=C[1]),
        ui.kpi_card("Class imbalance", f"{imbalance:.1f} : 1", sub="non-default : default", accent=C[7]),
    ])

    counts = df["TARGET"].value_counts()
    with ui.panel("Repaid vs defaulted", "The modelling challenge in one bar — an 8% positive class."):
        ui.chart(charts.ratio_bar(int(counts.get(0, 0)), int(counts.get(1, 0))))

    _data_quality(df)

    st.subheader("Business insights")
    a, b = st.columns(2)
    with a:
        _rate_chart(df, "NAME_EDUCATION_TYPE", "Default rate by education level", C[1])
        _rate_chart(df, "NAME_FAMILY_STATUS", "Default rate by family status", C[2])
    with b:
        _age(df)
        _ext_source(df)
    _gender_car(df)


def _data_quality(df: pd.DataFrame) -> None:
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing[missing > 0].head(12).sort_values()
    with ui.panel("Data quality", "Top columns by share of missing values."):
        ui.chart(charts.hbar(top_missing.index, top_missing.values, color=C[0],
                             suffix="%", height=340))
        if "DAYS_EMPLOYED" in df.columns:
            rate = (df["DAYS_EMPLOYED"] == 365243).mean() * 100
            ui.callout(
                f"<code>DAYS_EMPLOYED == 365243</code> is a <i>not currently employed</i> "
                f"placeholder in {rate:.1f}% of rows — handled as an explicit anomaly flag, "
                f"not a day count."
            )


def _rate_chart(df: pd.DataFrame, col: str, title: str, color: str) -> None:
    rate = (df.groupby(col)["TARGET"].mean() * 100).sort_values()
    with ui.panel(title):
        ui.chart(charts.hbar(rate.index, rate.values, color=color, suffix="%", height=260))


def _age(df: pd.DataFrame) -> None:
    bucket = pd.cut(-df["DAYS_BIRTH"] / 365.25, bins=_AGE_BINS, labels=_AGE_LABELS)
    rate = df.groupby(bucket, observed=True)["TARGET"].mean() * 100
    with ui.panel("Default rate by age"):
        ui.chart(charts.area(rate.index.astype(str), rate.values, color=C[6],
                             suffix="%", height=260))


def _ext_source(df: pd.DataFrame) -> None:
    decile = pd.qcut(df["EXT_SOURCE_2"], 10, duplicates="drop")
    rate = df.groupby(decile, observed=True)["TARGET"].mean() * 100
    with ui.panel("Default rate by EXT_SOURCE_2 decile", "Low → high external credit score."):
        ui.chart(charts.area([f"D{i + 1}" for i in range(len(rate))], rate.values, color=C[5],
                             suffix="%", height=260))


def _gender_car(df: pd.DataFrame) -> None:
    sub = df[df["CODE_GENDER"].isin(["F", "M"])]
    pivot = sub.groupby(["CODE_GENDER", "FLAG_OWN_CAR"])["TARGET"].mean().mul(100).unstack()
    with ui.panel("Default rate by gender and car ownership"):
        ui.chart(charts.grouped_bar(
            ["Female", "Male"],
            {"Owns a car": pivot["Y"].tolist(), "No car": pivot["N"].tolist()},
            suffix="%", height=300,
        ))
