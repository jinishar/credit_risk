"""EDA section: dataset summary, data quality, and the key business-insight charts.

Each insight gets a form matched to its job and its own single hue — imbalance
ratio bar, magnitude bars, change-over-an-ordered-axis areas, an identity
grouped bar — so the page reads as a varied dashboard, not a stack of bars.

Every aggregation is computed once in `_aggregates()` and cached: the dataset
doesn't change during a session, so reruns just redraw the cached frames.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data, theme, ui

_AGE_BINS = [18, 25, 35, 45, 55, 65, 100]
_AGE_LABELS = ["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
C = theme.CATEGORICAL


@st.cache_data(show_spinner=False)
def _aggregates() -> dict:
    df = data.dataset()
    counts = df["TARGET"].value_counts()

    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing[missing > 0].head(12).sort_values()

    anomaly_rate = (
        float((df["DAYS_EMPLOYED"] == 365243).mean() * 100)
        if "DAYS_EMPLOYED" in df.columns else None
    )

    age_bucket = pd.cut(-df["DAYS_BIRTH"] / 365.25, bins=_AGE_BINS, labels=_AGE_LABELS)
    age_rate = df.groupby(age_bucket, observed=True)["TARGET"].mean() * 100
    age_rate.index = age_rate.index.astype(str)

    ext_decile = pd.qcut(df["EXT_SOURCE_2"], 10, duplicates="drop")
    ext_rate = (df.groupby(ext_decile, observed=True)["TARGET"].mean() * 100).tolist()

    gender_sub = df[df["CODE_GENDER"].isin(["F", "M"])]
    pivot = gender_sub.groupby(["CODE_GENDER", "FLAG_OWN_CAR"])["TARGET"].mean().mul(100).unstack()

    return {
        "n_rows": len(df),
        "n_features": df.shape[1] - 2,
        "default_rate": float(df["TARGET"].mean() * 100),
        "imbalance": (df["TARGET"] == 0).sum() / max((df["TARGET"] == 1).sum(), 1),
        "repaid": int(counts.get(0, 0)),
        "defaulted": int(counts.get(1, 0)),
        "top_missing": top_missing,
        "anomaly_rate": anomaly_rate,
        "edu_rate": (df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100).sort_values(),
        "family_rate": (df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean() * 100).sort_values(),
        "age_rate": age_rate,
        "ext_rate": ext_rate,
        "gender_car": {"Owns a car": pivot["Y"].tolist(), "No car": pivot["N"].tolist()},
    }


def render() -> None:
    ui.section_header(
        "Exploratory Data Analysis",
        "Home Credit applications — demographics, financials, and data quality.",
        kicker="Dataset overview",
    )
    agg = _aggregates()

    ui.kpi_row([
        ui.kpi_card("Applicants", f"{agg['n_rows']:,}", accent=C[0]),
        ui.kpi_card("Features", f"{agg['n_features']}", accent=C[2]),
        ui.kpi_card("Default rate", f"{agg['default_rate']:.1f}%", accent=C[1]),
        ui.kpi_card("Class imbalance", f"{agg['imbalance']:.1f} : 1", sub="non-default : default", accent=C[7]),
    ])

    with ui.panel("Repaid vs defaulted", "The modelling challenge in one bar — an 8% positive class."):
        ui.chart(charts.ratio_bar(agg["repaid"], agg["defaulted"]))

    _data_quality(agg)

    st.subheader("Business insights")
    a, b = st.columns(2)
    with a:
        _rate_chart(agg["edu_rate"], "Default rate by education level", C[1])
        _rate_chart(agg["family_rate"], "Default rate by family status", C[2])
    with b:
        _age(agg["age_rate"])
        _ext_source(agg["ext_rate"])
    _gender_car(agg["gender_car"])


def _data_quality(agg: dict) -> None:
    top_missing = agg["top_missing"]
    with ui.panel("Data quality", "Top columns by share of missing values."):
        ui.chart(charts.hbar(top_missing.index, top_missing.values, color=C[0],
                             suffix="%", height=340))
        if agg["anomaly_rate"] is not None:
            ui.callout(
                f"<code>DAYS_EMPLOYED == 365243</code> is a <i>not currently employed</i> "
                f"placeholder in {agg['anomaly_rate']:.1f}% of rows — handled as an explicit anomaly flag, "
                f"not a day count."
            )


def _rate_chart(rate: pd.Series, title: str, color: str) -> None:
    with ui.panel(title):
        ui.chart(charts.hbar(rate.index, rate.values, color=color, suffix="%", height=260))


def _age(rate: pd.Series) -> None:
    with ui.panel("Default rate by age"):
        ui.chart(charts.area(rate.index.astype(str), rate.values, color=C[6],
                             suffix="%", height=260))


def _ext_source(rate: list) -> None:
    with ui.panel("Default rate by EXT_SOURCE_2 decile", "Low → high external credit score."):
        ui.chart(charts.area([f"D{i + 1}" for i in range(len(rate))], rate, color=C[5],
                             suffix="%", height=260))


def _gender_car(pivot: dict) -> None:
    with ui.panel("Default rate by gender and car ownership"):
        ui.chart(charts.grouped_bar(["Female", "Male"], pivot, suffix="%", height=300))
