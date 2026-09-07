"""Risk scoring section: score an applicant, show probability + calibrated band."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data, ui
from src.utils.config import RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD


def render() -> None:
    df = data.dataset()
    ui.section_header(
        "Risk Scoring",
        "Calibrated default probability and a Low / Medium / High risk band per applicant.",
        kicker="Score an applicant",
    )

    features = df.drop(columns=["TARGET"])
    mode = st.radio("Applicant source", ["Sample from dataset", "Manual entry"], horizontal=True)
    if mode == "Sample from dataset":
        row = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
        applicant = features.iloc[[row]]
    else:
        applicant = _manual_form(df, features)

    scored = data.score_applicant(applicant)
    proba = float(scored["default_probability"].iloc[0])
    band = scored["risk_band"].iloc[0]

    st.session_state["current_applicant"] = applicant
    st.session_state["current_proba"] = proba
    st.session_state["current_band"] = band

    left, right = st.columns([3, 2])
    with left:
        with ui.panel("Default probability", "Isotonic-calibrated — read as a real rate."):
            ui.chart(charts.gauge(proba, band))
    with right:
        with ui.panel("Risk band"):
            st.markdown(ui.band_pill(band, proba), unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel__caption" style="margin-top:.6rem">'
                f'Low &lt; {RISK_LOW_THRESHOLD:.0%} &nbsp;·&nbsp; '
                f'Medium &lt; {RISK_HIGH_THRESHOLD:.0%} &nbsp;·&nbsp; '
                f'High ≥ {RISK_HIGH_THRESHOLD:.0%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_attribute_chips(applicant.iloc[0]), unsafe_allow_html=True)

    ui.callout("Open <b>Explainability</b> for the SHAP breakdown of this score.")


def _manual_form(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    a = features.iloc[[0]].copy()
    c1, c2, c3 = st.columns(3)
    a["AMT_INCOME_TOTAL"] = c1.number_input("Annual income", value=150_000.0, step=10_000.0)
    a["AMT_CREDIT"] = c2.number_input("Requested credit", value=500_000.0, step=10_000.0)
    a["AMT_ANNUITY"] = c3.number_input("Annuity", value=25_000.0, step=1_000.0)
    a["NAME_EDUCATION_TYPE"] = c1.selectbox("Education", sorted(df["NAME_EDUCATION_TYPE"].dropna().unique()))
    a["NAME_FAMILY_STATUS"] = c2.selectbox("Family status", sorted(df["NAME_FAMILY_STATUS"].dropna().unique()))
    a["NAME_HOUSING_TYPE"] = c3.selectbox("Housing type", sorted(df["NAME_HOUSING_TYPE"].dropna().unique()))
    a["EXT_SOURCE_2"] = c1.slider("External credit score (EXT_SOURCE_2)", 0.0, 1.0, 0.5)
    return a


def _attribute_chips(row: pd.Series) -> str:
    def num(col, fmt):
        v = row.get(col)
        return fmt(v) if v is not None and v == v else "—"

    chips = [
        ui.stat_chip("Age", num("DAYS_BIRTH", lambda v: f"{-v / 365.25:.0f}")),
        ui.stat_chip("Income", num("AMT_INCOME_TOTAL", lambda v: f"{v:,.0f}")),
        ui.stat_chip("Credit", num("AMT_CREDIT", lambda v: f"{v:,.0f}")),
        ui.stat_chip("Education", str(row.get("NAME_EDUCATION_TYPE", "—"))),
        ui.stat_chip("EXT_SOURCE_2", num("EXT_SOURCE_2", lambda v: f"{v:.2f}")),
    ]
    return " ".join(chips)
