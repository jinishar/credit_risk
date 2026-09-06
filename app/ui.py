"""Reusable presentational components for the Streamlit UI.

Pure string builders (`band_pill`, `kpi_card`, `rule_card`, ...) are
unit-tested; the `st.*`-calling helpers (`kpi_row`, `panel`, `chart`,
`section_header`) are thin view glue covered by the app smoke test. All colour
comes from `app.theme`; all CSS from `app.styles`.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from app import styles, theme


def inject_css() -> None:
    st.markdown(styles.stylesheet(), unsafe_allow_html=True)


# --------------------------------------------------------------- headers

def section_header(title: str, caption: str | None = None, kicker: str | None = None) -> None:
    parts = ['<div class="app-header">']
    if kicker:
        parts.append(f'<div class="app-header__kicker">{kicker}</div>')
    parts.append(f"<h1>{title}</h1>")
    if caption:
        parts.append(f"<p>{caption}</p>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ----------------------------------------------------------------- cards

def kpi_card(
    label: str,
    value: str,
    sub: str | None = None,
    accent: str | None = None,
) -> str:
    """One metric card. `sub` renders a caption line only when given; `accent`
    recolours the left rule."""
    sub_html = f'<div class="kpi-card__sub">{sub}</div>' if sub else ""
    style = f' style="--accent:{accent}"' if accent else ""
    return (
        f'<div class="kpi-card"{style}><div class="kpi-card__label">{label}</div>'
        f'<div class="kpi-card__value">{value}</div>{sub_html}</div>'
    )


def kpi_row(cards: list[str]) -> None:
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def stat_chip(label: str, value: str) -> str:
    return f'<span class="stat-chip"><b>{label}</b>{value}</span>'


def band_pill(band: str, probability: float) -> str:
    """A coloured pill — a dot plus `<band> · <probability as %>` — outlined and
    tinted in the band's colour."""
    colour = theme.band_color(band)
    return (
        f'<span class="band-pill" style="color:{colour};background:{theme.band_bg(band)}">'
        f'<span class="band-pill__dot"></span>{band} &middot; {probability:.0%}</span>'
    )


def callout(html: str) -> None:
    """A quiet left-accented note — data-quality caveats, threshold definitions."""
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def _rate_tone(rate: float) -> str:
    """Risk-band colour for a predicted default rate — used to tint rule cards
    so the policy list is scannable at a glance."""
    if rate >= 0.5:
        return theme.BAND_COLORS["High"]
    if rate >= 0.2:
        return theme.BAND_COLORS["Medium"]
    return theme.BAND_COLORS["Low"]


def rule_card(conditions: list[str], rate: float, n_samples: int) -> str:
    """One derived policy rule: `IF <conds>` with its predicted default rate and
    supporting sample size. The left rule is tinted by the rate's risk band."""
    cond = " <b>AND</b> ".join(conditions)
    tone = _rate_tone(rate)
    return (
        f'<div class="rule-card" style="border-left-color:{tone}">'
        f'<span class="rule-card__rate" style="color:{tone}">{rate * 100:.1f}%</span>'
        f'<div class="rule-card__cond"><b>IF</b> {cond}</div>'
        f'<div class="rule-card__meta">{n_samples:,} applicants</div>'
        "</div>"
    )


# ---------------------------------------------------------------- layout

@contextmanager
def panel(title: str | None = None, caption: str | None = None) -> Iterator[None]:
    """A bordered card that groups a chart with its heading. Use as a context
    manager: `with ui.panel("Title", "caption"): st.plotly_chart(...)`."""
    with st.container(border=True):
        if title or caption:
            head = ['<div class="panel__head">']
            if title:
                head.append(f'<div class="panel__title">{title}</div>')
            if caption:
                head.append(f'<div class="panel__caption">{caption}</div>')
            head.append("</div>")
            st.markdown("".join(head), unsafe_allow_html=True)
        yield


def chart(fig) -> None:
    """Render a Plotly figure full-width with the modebar hidden. Figure size is
    set by the builder in `app.charts`."""
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------- sidebar

def sidebar_brand() -> None:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand__mark">📊</div>'
        '<div><div class="sidebar-brand__name">Credit Risk Platform</div>'
        '<div class="sidebar-brand__sub">Home Credit Default Risk</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def sidebar_model_card(
    *, source: str, n_applicants: int, champion: str, roc_auc: float, pr_auc: float
) -> None:
    """The live model summary in the sidebar. Caller supplies the values —
    this only formats them."""
    st.markdown(
        '<div class="sidebar-model">'
        f'<b>Data</b> &nbsp;{source} &middot; {n_applicants:,} applicants<br>'
        f'<b>Champion</b> &nbsp;{champion}<br>'
        f'<b>ROC-AUC</b> &nbsp;{roc_auc:.3f} &nbsp;&nbsp; <b>PR-AUC</b> &nbsp;{pr_auc:.3f}<br>'
        '<b>Probabilities</b> &nbsp;isotonic-calibrated'
        "</div>",
        unsafe_allow_html=True,
    )
