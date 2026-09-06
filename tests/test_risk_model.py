"""Guards on the calibrated risk score.

The champion is a LightGBM model wrapped in `CalibratedClassifierCV(isotonic,
cv='prefit')`. Regression test for a scikit-learn version-drift bug: since 1.7,
`CalibratedClassifierCV.predict_proba` feeds the isotonic fit (trained on
[0, 1] probabilities) the base estimator's `decision_function` margins instead,
and the out-of-bounds clip collapsed ~99% of applicants to exactly 0.0.

Skipped when the model artifacts haven't been trained yet.
"""
import pytest

from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()),
    reason="model artifacts not trained (run `python -m src.ml.train`)",
)


@pytest.fixture(scope="module")
def model():
    from src.ml.predict import RiskModel

    return RiskModel()


@pytest.fixture(scope="module")
def data():
    from src.data.loader import load_train_df

    return load_train_df()


@pytest.fixture(scope="module")
def scored(model, data):
    return model.score(data.drop(columns=["TARGET"]).sample(1000, random_state=0))


def test_calibrated_scores_track_the_true_base_rate(scored, data):
    mean_p = scored["default_probability"].mean()
    true_rate = data["TARGET"].mean()
    # isotonic calibration anchors the mean to the empirical default rate (~8%)
    assert true_rate * 0.5 < mean_p < true_rate * 2.0


def test_scores_do_not_collapse_to_zero(scored):
    zero_share = (scored["default_probability"] == 0).mean()
    assert zero_share < 0.10  # a few genuinely near-zero applicants are fine, not most


def test_low_and_medium_bands_both_appear_in_a_random_sample(scored):
    assert {"Low", "Medium"} <= set(scored["risk_band"])


def test_the_riskiest_applicants_can_reach_the_high_band(model, data):
    features = data.drop(columns=["TARGET"])
    riskiest = features.nsmallest(200, "EXT_SOURCE_2")  # weakest external credit score
    assert (model.score(riskiest)["risk_band"] == "High").any()
