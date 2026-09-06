"""Multi-section Streamlit UI: EDA / Risk Prediction / Explainability / Business Rules / Chatbot.

Single entrypoint for the whole platform (per the assignment's UI
requirement). Run with `streamlit run app/streamlit_app.py`, or via
docker-compose (see Dockerfile).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.data.loader import load_train_df
from src.ml.explain import RiskExplainer
from src.ml.predict import RiskModel
from src.rules.rule_engine import rules_for_applicant
from src.talk_to_data.query_runner import ChatSession
from src.talk_to_data.prompt_templates import CANONICAL_PATTERNS
from src.utils.config import GEMINI_API_KEY, MODEL_PATH, RULES_PATH
from src.utils.docker_utils import model_artifacts_present
from src.utils.helpers import load_json

st.set_page_config(page_title="Credit Risk Intelligence Platform", layout="wide")


@st.cache_data(show_spinner="Loading dataset...")
def _load_data() -> pd.DataFrame:
    return load_train_df()


@st.cache_resource(show_spinner="Loading model...")
def _load_model() -> RiskModel:
    return RiskModel()


@st.cache_resource(show_spinner="Preparing explainer...")
def _load_explainer(_model: RiskModel) -> RiskExplainer:
    return RiskExplainer(_model)


@st.cache_data(show_spinner=False)
def _load_rules() -> list[dict]:
    return load_json(RULES_PATH) if RULES_PATH.exists() else []


def _ensure_pipeline_ready() -> None:
    if model_artifacts_present():
        return
    st.warning("No trained model found yet - running the training pipeline once. This takes under a minute.")
    with st.spinner("Training model, computing SHAP background and deriving rules..."):
        from src.ml.train import train
        from src.rules.rule_engine import derive_and_save

        train()
        derive_and_save()
    st.cache_resource.clear()
    st.rerun()


st.title("AI-Powered Credit Risk Intelligence Platform")
st.caption(
    "Home Credit Default Risk | EDA · ML risk scoring · SHAP explainability · "
    "business rules · talk-to-data chatbot"
)

_ensure_pipeline_ready()

df = _load_data()
model = _load_model()

tab_eda, tab_predict, tab_explain, tab_rules, tab_chat = st.tabs(
    ["\U0001F4CA EDA", "\U0001F3AF Risk Prediction", "\U0001F50D Explainability", "\U0001F4CB Business Rules", "\U0001F4AC Chatbot"]
)

# ---------------------------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Features", df.shape[1] - 2)
    c3.metric("Default rate", f"{df['TARGET'].mean() * 100:.1f}%")
    c4.metric("Imbalance ratio", f"{(df['TARGET'] == 0).sum() / max((df['TARGET'] == 1).sum(), 1):.1f} : 1")

    st.subheader("Data Quality")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    st.bar_chart(missing[missing > 0].head(15))
    if "DAYS_EMPLOYED" in df.columns:
        anomaly_rate = (df["DAYS_EMPLOYED"] == 365243).mean() * 100
        st.info(f"Data quality note: `DAYS_EMPLOYED == 365243` is an anomalous placeholder meaning "
                f"'not currently employed', present in {anomaly_rate:.1f}% of applicants.")

    st.subheader("Business Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Default rate by education level**")
        st.bar_chart(df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values() * 100)
        st.markdown("**Default rate by family status**")
        st.bar_chart(df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().sort_values() * 100)
    with col2:
        age_years = -df["DAYS_BIRTH"] / 365.25
        age_bucket = pd.cut(age_years, bins=[18, 25, 35, 45, 55, 65, 100],
                             labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
        st.markdown("**Default rate by age bucket**")
        st.bar_chart(df.groupby(age_bucket, observed=True)["TARGET"].mean() * 100)
        st.markdown("**External credit score vs default**")
        ext_source_rate = df.groupby(pd.qcut(df["EXT_SOURCE_2"], 10, duplicates="drop"), observed=True)["TARGET"].mean()
        ext_source_rate.index = ext_source_rate.index.astype(str)  # Interval index isn't chart-JSON-serializable
        st.line_chart(ext_source_rate)

# ---------------------------------------------------------------------------- PREDICTION
with tab_predict:
    st.subheader("Score an Applicant")
    st.caption("Pick a sample applicant from the dataset, or provide your own values.")

    mode = st.radio("Applicant source", ["Sample from dataset", "Manual entry"], horizontal=True)

    if mode == "Sample from dataset":
        idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
        applicant = df.drop(columns=["TARGET"]).iloc[[idx]]
    else:
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Annual income", value=150000.0)
        credit = c2.number_input("Requested credit amount", value=500000.0)
        annuity = c3.number_input("Annuity", value=25000.0)
        education = c1.selectbox("Education", sorted(df["NAME_EDUCATION_TYPE"].dropna().unique()))
        family = c2.selectbox("Family status", sorted(df["NAME_FAMILY_STATUS"].dropna().unique()))
        housing = c3.selectbox("Housing type", sorted(df["NAME_HOUSING_TYPE"].dropna().unique()))
        ext2 = c1.slider("External credit score (EXT_SOURCE_2)", 0.0, 1.0, 0.5)

        applicant = df.drop(columns=["TARGET"]).iloc[[0]].copy()
        applicant["AMT_INCOME_TOTAL"] = income
        applicant["AMT_CREDIT"] = credit
        applicant["AMT_ANNUITY"] = annuity
        applicant["NAME_EDUCATION_TYPE"] = education
        applicant["NAME_FAMILY_STATUS"] = family
        applicant["NAME_HOUSING_TYPE"] = housing
        applicant["EXT_SOURCE_2"] = ext2

    scored = model.score(applicant)
    proba = float(scored["default_probability"].iloc[0])
    band = scored["risk_band"].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Predicted default probability", f"{proba * 100:.1f}%")
    band_color = {"Low": "green", "Medium": "orange", "High": "red"}[band]
    c2.markdown(f"### Risk band: :{band_color}[{band}]")

    st.session_state["current_applicant"] = applicant
    st.session_state["current_proba"] = proba
    st.session_state["current_band"] = band

# ------------------------------------------------------------------------- EXPLAINABILITY
with tab_explain:
    st.subheader("Why did the model predict this?")
    if "current_applicant" not in st.session_state:
        st.info("Score an applicant in the Risk Prediction tab first.")
    else:
        explainer = _load_explainer(model)
        applicant = st.session_state["current_applicant"]
        explanation = explainer.explain_one(applicant)

        st.metric("Predicted default probability", f"{explanation['predicted_probability'] * 100:.1f}%")
        st.markdown("**Top factors driving this prediction:**")
        contrib_df = pd.DataFrame(explanation["top_features"])
        contrib_df["impact"] = contrib_df["shap_value"].abs()
        for _, row in contrib_df.iterrows():
            icon = "\U0001F53A" if row["direction"] == "increases risk" else "\U0001F53B"
            st.write(f"{icon} **{row['feature']}** (value: {row['value']}) - {row['direction']}")
        st.bar_chart(contrib_df.set_index("feature")["shap_value"])

        st.subheader("Global Feature Importance")
        with st.spinner("Computing global SHAP importance on a sample..."):
            global_imp = explainer.global_importance(df.drop(columns=["TARGET"]).sample(min(1000, len(df))))
        st.bar_chart(global_imp.set_index("feature")["mean_abs_shap"])

# ------------------------------------------------------------------------------- RULES
with tab_rules:
    st.subheader("Business Rules Derived from the Model")
    rules = _load_rules()
    tree_rules = [r for r in rules if r["type"] == "surrogate_tree"]
    bin_rules = [r for r in rules if r["type"] == "threshold_bin"]

    st.markdown("**Policy rules (surrogate decision tree)**")
    for r in tree_rules:
        conditions = " AND ".join(r["conditions"])
        st.write(f"- IF {conditions} → predicted default rate **{r['predicted_default_rate'] * 100:.1f}%** "
                 f"({r['n_samples']} applicants)")

    st.markdown("**Threshold bins (observed default rate by key feature)**")
    if bin_rules:
        bin_df = pd.DataFrame(bin_rules)
        st.dataframe(bin_df[["feature", "range", "observed_default_rate", "n_samples"]], hide_index=True)

    if "current_applicant" in st.session_state:
        st.markdown("**Rules triggered by the currently scored applicant**")
        X = model.transform_features(st.session_state["current_applicant"])
        triggered = rules_for_applicant(rules, X.iloc[0])
        if triggered:
            for r in triggered:
                st.write(f"- {' AND '.join(r['conditions'])}")
        else:
            st.write("No surrogate-tree rule matched exactly (applicant falls in a low-risk default path).")

# ------------------------------------------------------------------------------- CHATBOT
with tab_chat:
    st.subheader("Talk to the Data")
    if not GEMINI_API_KEY:
        st.warning(
            "GEMINI_API_KEY is not set - free-form questions are disabled, but the 5 "
            "guaranteed sample questions below still work end-to-end."
        )

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession()
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    st.caption("Try one of the guaranteed sample questions, or type your own:")
    sample_cols = st.columns(len(CANONICAL_PATTERNS))
    for col, pattern in zip(sample_cols, CANONICAL_PATTERNS):
        if col.button(pattern["question"], key=f"sample_{pattern['question']}"):
            st.session_state["pending_question"] = pattern["question"]

    question = st.chat_input("Ask a question about the applicant data...")
    question = question or st.session_state.pop("pending_question", None)

    if question:
        with st.spinner("Thinking..."):
            turn = st.session_state.chat_session.ask(question)
        st.session_state.chat_log.append(turn)

    def _md_safe(text: str) -> str:
        # `$` pairs are read as LaTeX math by Streamlit's markdown; dollar amounts
        # in the narration ("$169,078 ... $165,612") would render as italic math.
        return (text or "").replace("$", r"\$")

    for turn in reversed(st.session_state.chat_log):
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            if turn.error:
                st.error(_md_safe(turn.narrative))
            else:
                st.write(_md_safe(turn.narrative))
                with st.expander("SQL used"):
                    st.code(turn.sql, language="sql")
                if not turn.result.empty:
                    st.dataframe(turn.result, hide_index=True)
