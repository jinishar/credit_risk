"""Plotly figure builders, themed via `app.theme`.

One rule from the dataviz skill drives the colour here: **one series → one
colour**. Variety on the page comes from using a *different* hue and a
*different form* per insight, never from ramping a single series by value.
Diverging bar is the one signed encoding (SHAP: red raises risk, blue lowers).
`shap_factors_frame` (data reshaping) is unit-tested; the builders are thin.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app import theme

_CORNER = 4  # rounded data-ends, per the dataviz mark spec


def shap_factors_frame(explanation: dict) -> pd.DataFrame:
    """`RiskExplainer.explain_one()` -> frame for a horizontal diverging bar:
    smallest-impact row first (Plotly draws bottom-to-top), with a direction colour."""
    frame = pd.DataFrame(explanation["top_features"]).iloc[::-1].reset_index(drop=True)
    frame["color"] = frame["direction"].map(
        {"increases risk": theme.SHAP_UP, "decreases risk": theme.SHAP_DOWN}
    )
    return frame


def _fig(traces, *, title: str = "", height: int | None = None, **layout) -> go.Figure:
    """Wrap traces in the shared dark layout. A title reserves top margin;
    without one the plot sits flush so panels don't carry dead space."""
    merged = theme.plotly_layout(**layout)
    merged.setdefault("margin", dict(l=8, r=8, t=44 if title else 10, b=8))
    if title:
        merged["title"] = dict(text=title, font=dict(size=14, color=theme.INK),
                               x=0, xanchor="left")
    if height is not None:
        merged["height"] = height
    fig = go.Figure(traces)
    fig.update_layout(**merged)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def ratio_bar(repaid: int, defaulted: int, *, title: str = "") -> go.Figure:
    """A single 100%-stacked horizontal bar — the class imbalance at a glance."""
    total = repaid + defaulted or 1
    return _fig(
        [
            go.Bar(y=[""], x=[repaid / total * 100], orientation="h", name="Repaid",
                   marker_color=theme.CATEGORICAL[2], text=f"Repaid {repaid / total:.1%}",
                   textposition="inside", insidetextanchor="middle",
                   textfont=dict(color="#ffffff")),
            go.Bar(y=[""], x=[defaulted / total * 100], orientation="h", name="Defaulted",
                   marker_color=theme.SHAP_UP, text=f"Defaulted {defaulted / total:.1%}",
                   textposition="inside", insidetextanchor="middle",
                   textfont=dict(color="#ffffff")),
        ],
        title=title, barmode="stack", showlegend=False, height=88,
        xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False),
        margin=dict(l=6, r=6, t=10, b=6), bargap=0.5,
    )


def hbar(labels, values, *, color: str, title: str = "", suffix: str = "",
         height: int = 300) -> go.Figure:
    return _fig(
        go.Bar(
            x=list(values), y=list(labels), orientation="h",
            marker=dict(color=color, cornerradius=_CORNER,
                        line=dict(color=theme.SURFACE, width=1)),
            text=[f"{v:.1f}{suffix}" for v in values], textposition="auto",
            textfont=dict(color=theme.INK),
            hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>",
        ),
        title=title, height=height, showlegend=False,
        xaxis=dict(title="", gridcolor=theme.GRID), yaxis=dict(title=""),
    )


def area(labels, values, *, color: str, title: str = "", suffix: str = "",
        height: int = 300) -> go.Figure:
    return _fig(
        go.Scatter(
            x=list(labels), y=list(values), mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=8, color=color, line=dict(color=theme.SURFACE, width=1)),
            fill="tozeroy", fillcolor=_rgba(color, 0.16),
            hovertemplate="%{x}: %{y:.1f}" + suffix + "<extra></extra>",
        ),
        title=title, height=height, showlegend=False,
    )


def grouped_bar(categories, series: dict[str, list[float]], *, title: str = "",
                suffix: str = "", height: int = 300) -> go.Figure:
    traces = [
        go.Bar(name=name, x=list(categories), y=list(vals),
               marker=dict(color=theme.CATEGORICAL[i], cornerradius=_CORNER),
               text=[f"{v:.1f}{suffix}" for v in vals], textposition="outside",
               textfont=dict(color=theme.MUTED))
        for i, (name, vals) in enumerate(series.items())
    ]
    return _fig(traces, title=title, height=height, barmode="group")


def diverging_bar(frame: pd.DataFrame, *, title: str = "", height: int = 320) -> go.Figure:
    return _fig(
        go.Bar(
            x=frame["shap_value"], y=frame["feature"], orientation="h",
            marker=dict(color=frame["color"], cornerradius=_CORNER,
                        line=dict(color=theme.SURFACE, width=1)),
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ),
        title=title, height=height, showlegend=False,
        xaxis=dict(title="", gridcolor=theme.GRID, zerolinecolor=theme.MUTED, zerolinewidth=1.5),
        yaxis=dict(title=""),
    )


def gauge(probability: float, band: str) -> go.Figure:
    """A clean half-donut: a recessive full track with the value drawn as a
    single band-coloured arc. Low/Medium/High context lives in the pill beside
    it, so the gauge itself stays uncluttered."""
    colour = theme.band_color(band)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix="%", font=dict(size=44, color=theme.INK)),
            gauge=dict(
                shape="angular",
                axis=dict(range=[0, 100], tickvals=[0, 50, 100],
                          tickfont=dict(color=theme.MUTED, size=11), ticklen=4),
                bar=dict(color=colour, thickness=0.30),
                bgcolor=theme.SURFACE_2,
                bordercolor="rgba(0,0,0,0)", borderwidth=0,
            ),
        )
    )
    fig.update_layout(**theme.plotly_layout(height=270, margin=dict(l=36, r=36, t=54, b=8)))
    return fig
