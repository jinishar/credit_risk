"""Explainability section: SHAP contributions for the last-scored applicant."""
from __future__ import annotations

import streamlit as st

from app import charts, data, theme, ui


def render() -> None:
    ui.section_header(
        "Explainability",
        "Which features drove the model's score — SHAP, per applicant and global.",
        kicker="Why this score",
    )

    if "current_applicant" not in st.session_state:
        ui.callout("Score an applicant in <b>Risk scoring</b> first, then come back here.")
        return

    applicant = st.session_state["current_applicant"]
    explanation = data.applicant_explanation(applicant)
    ui.kpi_row([
        ui.kpi_card("Predicted default probability",
                    f"{explanation['predicted_probability'] * 100:.1f}%", accent=theme.SHAP_UP),
    ])

    frame = charts.shap_factors_frame(explanation)
    with ui.panel("Top factors for this applicant",
                  "Signed SHAP contribution — red raises the predicted risk, blue lowers it."):
        ui.chart(charts.diverging_bar(frame, title="← lowers risk      |      raises risk →"))

    with ui.panel("Global feature importance", "Mean |SHAP| over a 1,000-applicant sample."):
        importance = data.global_importance().iloc[::-1]
        ui.chart(charts.hbar(importance["feature"], importance["mean_abs_shap"],
                             color=theme.CATEGORICAL[0], height=360))
