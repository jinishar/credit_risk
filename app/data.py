"""Cached loaders shared by every UI section.

Streamlit's cache is process-wide and survives reruns, so sections can call
these freely without reloading the dataset/model or recomputing SHAP. All heavy
lifting lives in `src/`; this module only adds the cache boundary.
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


@st.cache_resource(show_spinner="Connecting to the analytics database…")
def duckdb_connection():
    """One DuckDB connection with the `applications` table loaded, shared by
    every chat session. Built once per server rather than once per session, so
    the CSV is read into DuckDB a single time and concurrent users don't
    contend for a write-mode file lock. Queries run on per-call cursors."""
    from src.data.loader import get_duckdb_connection

    return get_duckdb_connection()


@st.cache_data(show_spinner=False)
def rules() -> list[dict]:
    return load_json(RULES_PATH) if RULES_PATH.exists() else []


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return load_json(METRICS_PATH) if METRICS_PATH.exists() else {}


@st.cache_data(show_spinner=False)
def score_applicant(applicant: pd.DataFrame) -> pd.DataFrame:
    """Probability + risk band for one applicant; cached on the applicant so
    reruns that don't change the inputs don't re-run the pipeline."""
    return risk_model().score(applicant)


@st.cache_data(show_spinner=False)
def transformed_features(applicant: pd.DataFrame) -> pd.DataFrame:
    """The fully-engineered feature row for one applicant (used by the rules
    section), cached on the applicant."""
    return risk_model().transform_features(applicant)


@st.cache_data(show_spinner="Explaining this score…")
def applicant_explanation(applicant: pd.DataFrame) -> dict:
    """Per-applicant SHAP breakdown, cached on the applicant."""
    model = risk_model()
    return explainer(model).explain_one(applicant)


@st.cache_data(show_spinner="Computing SHAP on a 1,000-applicant sample…")
def global_importance(n: int = 1000) -> pd.DataFrame:
    """Global mean-|SHAP| importance. The model and dataset never change during
    a session, so this is computed once and reused across reruns and sections."""
    model = risk_model()
    df = dataset()
    sample = df.drop(columns=["TARGET"]).sample(min(n, len(df)), random_state=0)
    return explainer(model).global_importance(sample)


def data_is_real(df: pd.DataFrame) -> bool:
    """Heuristic for the sidebar badge: the real Kaggle set is ~307K rows, the
    synthetic fallback defaults to N_SYNTHETIC_ROWS."""
    return len(df) > N_SYNTHETIC_ROWS * 1.5
