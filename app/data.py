"""Cached loaders shared by every UI section.

Streamlit's cache is process-wide, so sections can call these freely without
reloading the dataset or model. All heavy lifting lives in `src/`.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.loader import load_train_df
from src.ml.explain import RiskExplainer
from src.ml.predict import RiskModel
from src.utils.config import METRICS_PATH, N_SYNTHETIC_ROWS, RULES_PATH
from src.utils.helpers import load_json


@st.cache_data(show_spinner="Loading dataset…")
def dataset() -> pd.DataFrame:
    return load_train_df()


@st.cache_resource(show_spinner="Loading model…")
def risk_model() -> RiskModel:
    return RiskModel()


@st.cache_resource(show_spinner="Preparing explainer…")
def explainer(_model: RiskModel) -> RiskExplainer:
    return RiskExplainer(_model)


@st.cache_data(show_spinner=False)
def rules() -> list[dict]:
    return load_json(RULES_PATH) if RULES_PATH.exists() else []


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return load_json(METRICS_PATH) if METRICS_PATH.exists() else {}


def data_is_real(df: pd.DataFrame) -> bool:
    """Heuristic for the sidebar badge: the real Kaggle set is ~307K rows, the
    synthetic fallback defaults to N_SYNTHETIC_ROWS."""
    return len(df) > N_SYNTHETIC_ROWS * 1.5
