"""Visual design tokens for the Streamlit UI — dark analytical theme.

Single source of truth for colour so every section renders consistently, and
so `.streamlit/config.toml` (Streamlit's own chrome) can mirror the same values.

The categorical palette and diverging pair are the dataviz-skill reference
instance, stepped for a dark surface and validated as a set against
`SURFACE` (#161d2b): worst adjacent CVD ΔE 8.4, normal-vision ΔE 19.3, every
slot ≥ 3:1 contrast. Colour is assigned by the job it does — categorical =
identity, sequential = magnitude, diverging = signed contribution, status =
risk band — never by rank, never cycled past slot 8.
"""
from __future__ import annotations

# --- surfaces / ink ------------------------------------------------------
BG = "#0b1220"        # app page plane, behind everything
SURFACE = "#161d2b"   # cards, sidebar, chart surface (palette is validated here)
SURFACE_2 = "#1e2739"  # raised / hover state
LINE = "#2a3446"      # hairline borders
GRID = "rgba(147,161,184,0.12)"  # chart gridlines — recessive
INK = "#e8edf6"       # primary text
MUTED = "#93a1b8"     # secondary text, axis labels
FAINT = "#5b6880"     # tertiary text, disabled
PRIMARY = "#3987e5"   # accent / active nav (= categorical slot 1)

# --- categorical palette (fixed order, never cycled) --------------------
CATEGORICAL = [
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
]

# --- diverging pair (signed contribution, neutral midpoint) -------------
SHAP_UP = "#e66767"    # feature increases predicted risk
SHAP_DOWN = "#3987e5"  # feature decreases predicted risk
DIVERGING_MID = "#383835"

# --- status: risk band (reserved, always shipped with a label) ---------
BAND_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
BAND_BG = {
    "Low": "rgba(12,163,12,0.16)",
    "Medium": "rgba(250,178,25,0.16)",
    "High": "rgba(208,59,59,0.18)",
}

FONT_STACK = '"Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif'


def band_color(band: str) -> str:
    """Hex colour for a risk-band label; muted grey for anything unrecognised."""
    return BAND_COLORS.get(band, MUTED)


def band_bg(band: str) -> str:
    return BAND_BG.get(band, "rgba(147,161,184,0.12)")


def plotly_layout(**overrides) -> dict:
    """Base layout for every figure — transparent surfaces (the card behind
    supplies the colour), recessive grid, light ink. `overrides` win."""
    layout = dict(
        font=dict(family=FONT_STACK, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis=dict(gridcolor=GRID, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
        yaxis=dict(gridcolor=GRID, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
        colorway=CATEGORICAL,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=LINE,
                        font=dict(color=INK, family=FONT_STACK)),
    )
    layout.update(overrides)
    return layout
