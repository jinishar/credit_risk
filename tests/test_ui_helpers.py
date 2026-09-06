"""Unit tests for the pure UI helpers (app/theme.py, app/ui.py, app/charts.py).

The Streamlit `render()` functions are view glue and are smoke-tested separately;
these cover the logic that decides colours, formats values and reshapes data.
"""
import pandas as pd

from app import charts, theme, ui


# --- theme.band_color ---------------------------------------------------------

def test_band_color_returns_the_palette_entry_for_each_known_band():
    for band in ("Low", "Medium", "High"):
        assert theme.band_color(band) == theme.BAND_COLORS[band]


def test_band_color_falls_back_to_muted_for_an_unknown_band():
    assert theme.band_color("not-a-band") == theme.MUTED


# --- ui.band_pill ------------------------------------------------------------

def test_band_pill_shows_the_band_label_and_the_probability_as_a_percentage():
    html = ui.band_pill("High", 0.732)
    assert "High" in html
    assert "73%" in html


def test_band_pill_is_coloured_by_the_band():
    assert theme.band_color("Low") in ui.band_pill("Low", 0.04)


# --- ui.kpi_card -----------------------------------------------------------

def test_kpi_card_includes_its_label_and_value():
    html = ui.kpi_card("Applicants", "307,511")
    assert "Applicants" in html
    assert "307,511" in html


def test_kpi_card_only_renders_the_sub_line_when_one_is_given():
    assert "kpi-card__sub" not in ui.kpi_card("Default rate", "8.1%")
    assert "kpi-card__sub" in ui.kpi_card("Default rate", "8.1%", sub="11.4:1 imbalance")


# --- charts.shap_factors_frame --------------------------------------------

def test_shap_factors_frame_orders_smallest_impact_first_for_a_horizontal_bar():
    explanation = {
        "top_features": [
            {"feature": "EXT_SOURCE_MEAN", "shap_value": 1.2, "direction": "increases risk"},
            {"feature": "AGE_YEARS", "shap_value": -0.3, "direction": "decreases risk"},
        ]
    }
    frame = charts.shap_factors_frame(explanation)
    # Plotly draws horizontal bars bottom-to-top, so the biggest driver must be last.
    assert list(frame["feature"]) == ["AGE_YEARS", "EXT_SOURCE_MEAN"]


def test_shap_factors_frame_tags_each_row_with_a_direction_colour():
    explanation = {"top_features": [{"feature": "X", "shap_value": 0.5, "direction": "increases risk"}]}
    frame = charts.shap_factors_frame(explanation)
    assert frame.loc[0, "color"] == theme.SHAP_UP


# --- ui.rule_card ----------------------------------------------------------

def test_rule_card_shows_each_condition_the_rate_and_the_sample_size():
    html = ui.rule_card(["EXT_SOURCE_MEAN <= 0.25", "EXT_SOURCE_3 <= 0.226"], 0.341, 384)
    assert "EXT_SOURCE_MEAN <= 0.25" in html
    assert "EXT_SOURCE_3 <= 0.226" in html
    assert "34.1%" in html
    assert "384 applicants" in html


def test_rule_card_joins_multiple_conditions_with_and():
    html = ui.rule_card(["a > 1", "b < 2"], 0.1, 10)
    assert "AND" in html


# --- theme.plotly_layout -------------------------------------------------

def test_plotly_layout_uses_the_categorical_colorway_on_a_transparent_surface():
    layout = theme.plotly_layout()
    assert layout["colorway"] == theme.CATEGORICAL
    assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"


def test_plotly_layout_lets_overrides_win():
    assert theme.plotly_layout(height=260)["height"] == 260
