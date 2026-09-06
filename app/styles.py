"""The app's stylesheet, built from `app.theme` tokens.

Kept out of `app.ui` so that module stays about components. Every colour comes
from `theme` — nothing is hard-coded here — so the palette has one home. The
rules are scoped to our own class names plus a short list of stable
`data-testid` hooks; a Streamlit version bump can only degrade the styling to
plain, never break the layout.
"""
from __future__ import annotations

from app import theme

_NAV_ICONS = ("📊", "🎯", "🔬", "⚖️", "💬")


def _nav_icon_rules() -> str:
    return "\n".join(
        f'  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type({i})::before {{'
        f' content: "{icon}"; }}'
        for i, icon in enumerate(_NAV_ICONS, start=1)
    )


def stylesheet() -> str:
    t = theme
    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* ---------- app shell ---------- */
  .stApp {{
    background:
      radial-gradient(1100px 620px at 88% -8%, rgba(57,135,229,0.10), transparent 60%),
      radial-gradient(900px 600px at 0% 0%, rgba(144,133,233,0.06), transparent 55%),
      {t.BG};
    color: {t.INK};
    font-family: {t.FONT_STACK};
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"] {{ right: 1rem; }}
  [data-testid="stAppDeployButton"] {{ display: none; }}
  #MainMenu, footer {{ visibility: hidden; }}

  .block-container, [data-testid="stAppViewBlockContainer"] {{
    padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1200px;
  }}

  h1, h2, h3, h4 {{ color: {t.INK}; font-weight: 650; letter-spacing: -0.01em; }}
  a {{ color: {t.PRIMARY}; }}
  ::selection {{ background: rgba(57,135,229,0.35); }}

  /* ---------- section header ---------- */
  .app-header {{ margin: 0 0 1.4rem; }}
  .app-header__kicker {{
    color: {t.PRIMARY}; font-size: .74rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .12em;
  }}
  .app-header h1 {{ font-size: 1.6rem; margin: .25rem 0 0; }}
  .app-header p {{ color: {t.MUTED}; margin: .35rem 0 0; font-size: .92rem; max-width: 82ch; }}

  /* ---------- KPI cards ---------- */
  .kpi-row {{ display: flex; gap: .9rem; flex-wrap: wrap; margin: .2rem 0 1.5rem; }}
  .kpi-card {{
    flex: 1 1 168px; background: {t.SURFACE}; border: 1px solid {t.LINE};
    border-radius: 14px; padding: .95rem 1.1rem; position: relative; overflow: hidden;
    transition: border-color .15s ease, transform .15s ease;
  }}
  .kpi-card::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent, {t.PRIMARY});
  }}
  .kpi-card:hover {{ border-color: {t.FAINT}; transform: translateY(-1px); }}
  .kpi-card__label {{
    color: {t.MUTED}; font-size: .72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .07em;
  }}
  .kpi-card__value {{
    color: {t.INK}; font-size: 1.7rem; font-weight: 680; margin-top: .3rem;
    font-variant-numeric: tabular-nums; line-height: 1.1;
  }}
  .kpi-card__sub {{ color: {t.MUTED}; font-size: .78rem; margin-top: .25rem; }}

  /* ---------- panel (chart grouping = a bordered st.container) ---------- */
  [data-testid="stVerticalBlockBorderWrapper"] {{
    background: {t.SURFACE}; border: 1px solid {t.LINE} !important;
    border-radius: 16px; padding: 1.1rem 1.2rem;
  }}
  .panel__head {{ margin-bottom: .5rem; }}
  .panel__title {{ color: {t.INK}; font-size: 1rem; font-weight: 640; }}
  .panel__caption {{ color: {t.MUTED}; font-size: .82rem; margin-top: .2rem; }}

  /* ---------- band pill ---------- */
  .band-pill {{
    display: inline-flex; align-items: center; gap: .5rem; padding: .45rem 1rem;
    border-radius: 999px; font-weight: 650; font-size: 1rem; border: 1px solid currentColor;
  }}
  .band-pill__dot {{
    width: .55rem; height: .55rem; border-radius: 999px; background: currentColor;
  }}

  /* ---------- stat chips ---------- */
  .stat-chip {{
    display: inline-block; background: {t.SURFACE_2}; border: 1px solid {t.LINE};
    border-radius: 8px; padding: .35rem .65rem; margin: .25rem .35rem .25rem 0;
    font-size: .82rem; color: {t.INK};
  }}
  .stat-chip b {{ color: {t.MUTED}; font-weight: 500; margin-right: .35rem; }}

  /* ---------- callout ---------- */
  .callout {{
    background: {t.SURFACE_2}; border: 1px solid {t.LINE}; border-left: 3px solid {t.PRIMARY};
    border-radius: 10px; padding: .7rem .9rem; margin: .6rem 0; color: {t.MUTED};
    font-size: .85rem; line-height: 1.5;
  }}
  .callout code {{ color: {t.INK}; background: rgba(147,161,184,0.14); padding: 0 .3rem; border-radius: 4px; }}

  /* ---------- rule cards ---------- */
  .rule-card {{
    background: {t.SURFACE}; border: 1px solid {t.LINE}; border-left: 3px solid {t.PRIMARY};
    border-radius: 12px; padding: .75rem .95rem; margin-bottom: .6rem;
  }}
  .rule-card__cond {{ color: {t.INK}; font-size: .9rem; line-height: 1.45; }}
  .rule-card__cond b {{ color: {t.PRIMARY}; font-weight: 600; }}
  .rule-card__rate {{ float: right; font-weight: 700; color: {t.INK}; }}
  .rule-card__meta {{ color: {t.MUTED}; font-size: .78rem; margin-top: .2rem; }}

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] {{
    background: {t.SURFACE}; border-right: 1px solid {t.LINE};
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
  section[data-testid="stSidebar"] hr {{ border-color: {t.LINE}; margin: 1rem 0; }}

  .sidebar-brand {{ display: flex; align-items: center; gap: .7rem; margin-bottom: 1.2rem; }}
  .sidebar-brand__mark {{
    width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0;
    background: linear-gradient(135deg, {t.PRIMARY}, {t.CATEGORICAL[6]});
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; box-shadow: 0 2px 10px rgba(57,135,229,0.35);
  }}
  .sidebar-brand__name {{ color: {t.INK}; font-weight: 660; font-size: .98rem; line-height: 1.2; }}
  .sidebar-brand__sub {{ color: {t.MUTED}; font-size: .74rem; }}

  /* radio -> nav rail */
  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: .25rem; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    display: flex; align-items: center; gap: .6rem; width: 100%;
    padding: .55rem .7rem; border-radius: 10px; cursor: pointer;
    color: {t.MUTED}; font-weight: 500; transition: background .12s ease, color .12s ease;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background: {t.SURFACE_2}; color: {t.INK};
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display: none; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label::before {{
    font-size: .95rem; width: 1.2rem; text-align: center;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
    background: rgba(57,135,229,0.14); color: {t.INK}; font-weight: 600;
    box-shadow: inset 3px 0 0 {t.PRIMARY};
  }}
{_nav_icon_rules()}

  .sidebar-model {{
    background: {t.SURFACE_2}; border: 1px solid {t.LINE}; border-radius: 12px;
    padding: .8rem .9rem; font-size: .8rem; color: {t.MUTED}; line-height: 1.7;
  }}
  .sidebar-model b {{ color: {t.INK}; font-weight: 600; }}
  .sidebar-foot {{ color: {t.FAINT}; font-size: .72rem; margin-top: 1rem; }}

  /* ---------- widgets ---------- */
  [data-testid="stMetric"] {{
    background: {t.SURFACE}; border: 1px solid {t.LINE}; border-radius: 14px; padding: .9rem 1.1rem;
  }}
  [data-testid="stMetricLabel"] p {{ color: {t.MUTED}; font-size: .72rem;
    text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }}
  [data-testid="stMetricValue"] {{ color: {t.INK}; font-variant-numeric: tabular-nums; }}

  [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
  [data-testid="stHorizontalBlock"] [class*="stColumn"] > div,
  [data-testid="stHorizontalBlock"] [data-testid="column"] > div {{ height: 100%; }}
  .stButton {{ height: 100%; }}
  .stButton > button {{
    height: 100%; background: {t.SURFACE_2}; color: {t.INK}; border: 1px solid {t.LINE};
    border-radius: 10px; font-weight: 500; font-size: .84rem; white-space: normal;
    transition: border-color .12s ease, background .12s ease;
  }}
  .stButton > button:hover {{ border-color: {t.PRIMARY}; color: {t.INK}; background: {t.SURFACE}; }}

  [data-testid="stExpander"] {{ border: 1px solid {t.LINE}; border-radius: 10px; background: {t.SURFACE}; }}
  [data-testid="stExpander"] summary {{ color: {t.MUTED}; }}

  [data-testid="stChatInput"] {{ background: {t.SURFACE}; border: 1px solid {t.LINE}; border-radius: 12px; }}
  [data-testid="stChatInput"] textarea {{ color: {t.INK}; }}

  [data-testid="stDataFrame"] {{ border: 1px solid {t.LINE}; border-radius: 10px; }}

  /* ---------- scrollbars ---------- */
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-track {{ background: {t.BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {t.LINE}; border-radius: 6px; border: 2px solid {t.BG}; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {t.FAINT}; }}
</style>
"""
