"""Multi-section Streamlit UI for the Credit Risk Intelligence Platform.

Thin entrypoint: page config, first-run training check, sidebar navigation,
and dispatch to one `render()` per section (see `app/sections/`). All modelling
lives in `src/`; this file contains no business logic.

Run: `streamlit run app/streamlit_app.py`  (or via docker-compose).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import data, ui
from app.sections import chat, eda, explainability, rules, scoring
from src.utils.docker_utils import model_artifacts_present

st.set_page_config(page_title="Credit Risk Intelligence Platform", page_icon="📊", layout="wide")
ui.inject_css()

PAGES = {
    "EDA": eda.render,
    "Risk scoring": scoring.render,
    "Explainability": explainability.render,
    "Business rules": rules.render,
    "Ask the data": chat.render,
}


def _ensure_pipeline_ready() -> None:
    """On a fresh clone/volume there are no model artifacts yet — train once."""
    if model_artifacts_present():
        return
    st.warning("No trained model yet — running the training pipeline once (~1–2 min).")
    with st.spinner("Training the model and deriving business rules…"):
        from src.ml.train import train
        from src.rules.rule_engine import derive_and_save

        train()
        derive_and_save()
    st.cache_resource.clear()
    st.rerun()


def _sidebar(df) -> str:
    ui.sidebar_brand()
    choice = st.radio("Section", list(PAGES), label_visibility="collapsed")
    st.divider()

    m = data.metrics()
    champ = m.get("champion") or {}
    ui.sidebar_model_card(
        source="real Kaggle" if data.data_is_real(df) else "synthetic",
        n_applicants=len(df),
        champion=m.get("champion_model", "—"),
        roc_auc=champ.get("roc_auc", 0.0),
        pr_auc=champ.get("pr_auc", 0.0),
    )
    st.markdown(
        '<div class="sidebar-foot">EDA · ML scoring · SHAP · rules · talk-to-data</div>',
        unsafe_allow_html=True,
    )
    return choice


_ensure_pipeline_ready()
_df = data.dataset()

with st.sidebar:
    _choice = _sidebar(_df)

PAGES[_choice]()
