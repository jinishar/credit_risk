"""Bridges the ML model's opaque predictions to business-readable credit policy rules.

Two complementary techniques, both required by the assignment's "Bridge ML
insights and credit policy with rules" business goal:

1. Surrogate decision tree: a shallow (depth<=3) decision tree is trained to
   mimic the champion model's predicted probabilities on a sample of
   applicants. Because it's shallow, every path from root to leaf translates
   directly into an IF/THEN policy rule a credit analyst can read, review and
   challenge -- unlike the LightGBM ensemble itself.
2. Threshold bins: the model's dominant driver (EXT_SOURCE_MEAN - the
   averaged external credit score, which carries ~8x the mean-abs SHAP of any
   other feature) is cut into quantile bins and the *actual observed* default
   rate for each bin is reported, giving a simple, auditable "IF feature in
   range X THEN observed default rate is Y%" statement grounded directly in
   the labelled data rather than in the surrogate tree's approximation.

Output is a JSON-serializable list of rules consumed by the UI's Rules tab
and by the chatbot when a user asks "why" questions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree

from src.ml.predict import RiskModel
from src.utils.config import RULES_PATH
from src.utils.helpers import save_json
from src.utils.logger import get_logger

log = get_logger(__name__)

SURROGATE_MAX_DEPTH = 3


def _tree_to_rules(tree: DecisionTreeRegressor, feature_names: list[str]) -> list[dict]:
    """Extracts every root-to-leaf path of a shallow sklearn tree as a readable rule."""
    tree_ = tree.tree_
    rules: list[dict] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.3g}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.3g}"])
        else:
            avg_risk = float(tree_.value[node][0][0])
            n_samples = int(tree_.n_node_samples[node])
            rules.append({
                "type": "surrogate_tree",
                "conditions": conditions,
                "predicted_default_rate": round(avg_risk, 4),
                "n_samples": n_samples,
            })

    recurse(0, [])
    return rules


def _threshold_bin_rules(df: pd.DataFrame, feature: str, y_true: pd.Series, n_bins: int = 5) -> list[dict]:
    values = pd.to_numeric(df[feature], errors="coerce")
    bins = pd.qcut(values, n_bins, duplicates="drop")
    grouped = y_true.groupby(bins, observed=True).agg(["mean", "count"])
    rules = []
    for interval, row in grouped.iterrows():
        rules.append({
            "type": "threshold_bin",
            "feature": feature,
            "range": f"({interval.left:.3g}, {interval.right:.3g}]",
            "observed_default_rate": round(float(row["mean"]), 4),
            "n_samples": int(row["count"]),
        })
    return sorted(rules, key=lambda r: r["observed_default_rate"], reverse=True)


def derive_rules(
    risk_model: RiskModel,
    applicants_df: pd.DataFrame,
    y_true: pd.Series | None = None,
    sample_size: int = 20_000,
) -> list[dict]:
    sample = applicants_df.sample(min(sample_size, len(applicants_df)), random_state=42)
    X = risk_model.transform_features(sample)
    predicted_proba = risk_model.model.predict_proba(X)[:, 1]

    # Surrogate tree mimics the *model* (that is its job), so it is fit on predictions.
    surrogate = DecisionTreeRegressor(max_depth=SURROGATE_MAX_DEPTH, random_state=42)
    surrogate.fit(X, predicted_proba)
    tree_rules = _tree_to_rules(surrogate, list(X.columns))
    tree_rules.sort(key=lambda r: r["predicted_default_rate"], reverse=True)

    # Threshold bins report the real observed default rate where labels are available.
    for candidate in ("EXT_SOURCE_MEAN", "CREDIT_INCOME_RATIO"):
        if candidate in X.columns:
            key_feature = candidate
            break
    else:
        key_feature = X.columns[0]
    if y_true is not None:
        bin_target = pd.to_numeric(y_true.loc[sample.index], errors="coerce")
    else:
        bin_target = pd.Series(predicted_proba, index=X.index)
    bin_rules = _threshold_bin_rules(X, key_feature, bin_target)

    rules = tree_rules + bin_rules
    log.info("Derived %d surrogate-tree rules and %d threshold-bin rules", len(tree_rules), len(bin_rules))
    return rules


def rules_for_applicant(rules: list[dict], applicant_features: pd.Series) -> list[dict]:
    """Returns which surrogate-tree rules a given (already feature-engineered) applicant triggers."""
    triggered = []
    for rule in rules:
        if rule["type"] != "surrogate_tree":
            continue
        if all(_condition_holds(cond, applicant_features) for cond in rule["conditions"]):
            triggered.append(rule)
    return triggered


def _condition_holds(condition: str, features: pd.Series) -> bool:
    for op in ("<=", ">"):
        if op in condition:
            name, threshold = condition.split(op)
            name, threshold = name.strip(), float(threshold.strip())
            value = features.get(name, np.nan)
            if pd.isna(value):
                return False
            return (value <= threshold) if op == "<=" else (value > threshold)
    return False


def derive_and_save(risk_model: RiskModel | None = None) -> list[dict]:
    from src.data.loader import load_train_df

    risk_model = risk_model or RiskModel()
    df = load_train_df()
    y_true = df["TARGET"] if "TARGET" in df.columns else None
    rules = derive_rules(risk_model, df.drop(columns=["TARGET"], errors="ignore"), y_true=y_true)
    save_json(rules, RULES_PATH)
    log.info("Saved %d rules to %s", len(rules), RULES_PATH)
    return rules


if __name__ == "__main__":
    for r in derive_and_save()[:10]:
        print(r)
