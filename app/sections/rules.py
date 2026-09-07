"""Business rules section: the model-derived IF/THEN policy and observed-rate bins."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data, ui
from src.rules.rule_engine import rules_for_applicant


def render() -> None:
    ui.section_header(
        "Business Rules",
        "Human-readable credit policy derived from the model — for analyst review and audit.",
        kicker="Model → policy",
    )

    rules = data.rules()
    if not rules:
        ui.callout("No rules found. Run <code>python -m src.rules.rule_engine</code> to derive them.")
        return

    tree = [r for r in rules if r["type"] == "surrogate_tree"]
    bins = [r for r in rules if r["type"] == "threshold_bin"]

    with ui.panel("Policy rules", "Surrogate decision tree (depth ≤ 3) trained to mimic the champion."):
        for r in tree:
            st.markdown(
                ui.rule_card(r["conditions"], r["predicted_default_rate"], r["n_samples"]),
                unsafe_allow_html=True,
            )

    if bins:
        with ui.panel("Observed default rate by key feature",
                      "Computed against the real TARGET labels, not model predictions."):
            table = pd.DataFrame(bins)[["feature", "range", "observed_default_rate", "n_samples"]].copy()
            table["observed_default_rate"] = (table["observed_default_rate"] * 100).round(1)
            table = table.rename(columns={
                "feature": "Feature", "range": "Range",
                "observed_default_rate": "Observed default %", "n_samples": "Applicants",
            })
            st.dataframe(table, hide_index=True, use_container_width=True)

    if "current_applicant" in st.session_state:
        with ui.panel("Rules triggered by the current applicant"):
            row = data.transformed_features(st.session_state["current_applicant"]).iloc[0]
            hits = rules_for_applicant(rules, row)
            if hits:
                for r in hits:
                    st.markdown(f"- **IF** {' AND '.join(r['conditions'])}")
            else:
                st.write("No surrogate-tree path matches exactly — the applicant sits on a low-risk branch.")
