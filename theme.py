"""Shared visual tokens for the Streamlit UI: a technical-instrument
aesthetic (near-white/near-black, zero border radius, hairline borders,
monospace numbers) used by both app.py and analytics.py. Not copied from
any specific product - an original palette/type system built to match this
project's own calibrated-probability output.

Theme-aware (light and dark) - each mode gets its own step from the same
hue ramp, not a naive CSS invert. See
docs/superpowers/plans/2026-09-22-theme-support.md for the design."""

from functools import lru_cache

TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#FEFEFE", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "accent": "#3355FF", "field_bg": "#1B1F2A", "field_ink": "#F2F4F8",
    },
    "dark": {
        "surface": "#0E1117", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "accent": "#3355FF", "field_bg": "#262A36", "field_ink": "#F2F4F8",
    },
}

# Deliberately identical in both modes - clears contrast against both
# #FEFEFE and #0E1117 (verified: 3.56:1 / 5.26:1), so chart axis text stays
# legible even if theme detection lags by one interaction.
MUTED = "#898781"

# One shared accent for both modes (clears 3:1 against both surfaces:
# 5.36:1 light, 3.50:1 dark, verified) - do not introduce a second. Call
# sites that build a figure without threading `mode` through (e.g.
# analytics.py, which has no theme awareness of its own) can reference this
# directly instead of theme.TOKENS[mode]["accent"].
ACCENT = "#3355FF"

# Validated (light, dark) pairs for a chart that ever needs more than the
# single shared accent. Fixed order, never cycled/reassigned past slot 3.
SERIES_COLORS: dict[str, list[str]] = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}

FONT_SANS = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600;700&display=swap"
)


def resolve_initial_theme(detected: str | None) -> str:
    """Best-effort seed for the FIRST run only. st.context.theme.type can be
    absent (pre-Streamlit-1.46), wrong on the first script run
    (streamlit#11920), and does not trigger a rerun on an OS-level theme
    change (streamlit#15287) - so this is only ever used to pick a starting
    value; st.session_state owns the truth after that."""
    return "dark" if detected == "dark" else "light"


@lru_cache(maxsize=2)
def build_plotly_template(mode: str) -> dict:
    """Pure, cached (per mode) - the go.layout.Template layout dict. Ticks/
    axis titles/axis lines use MUTED (identical both modes, no Python
    dependency); gridlines use the mode's own --grid value (Plotly bakes
    colors at render time, so this one genuinely needs Python to know the
    theme). paper/plot bgcolor are transparent so the page background - a
    pure-CSS layer - always shows through regardless of what Python
    believes the theme is."""
    tokens = TOKENS[mode]
    axis = dict(
        color=MUTED, linecolor=MUTED, gridcolor=tokens["grid"], zerolinecolor=tokens["grid"],
        showline=True, ticks="outside", tickcolor=MUTED,
        tickfont=dict(color=MUTED, size=11, family=FONT_MONO),
        title=dict(font=dict(color=MUTED, size=12, family=FONT_MONO)),
    )
    return dict(
        font=dict(family=FONT_MONO, size=12, color=tokens["ink"]),
        title=dict(font=dict(size=13, color=tokens["ink"], family=FONT_MONO)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[tokens["accent"]],
        xaxis=axis, yaxis=axis,
        legend=dict(font=dict(color=tokens["ink"], size=11, family=FONT_MONO)),
        margin=dict(l=48, r=16, t=36, b=40),
    )


def register_plotly_template(mode: str) -> None:
    """Call whenever the resolved theme changes (app.py does this once per
    script run, driven by st.session_state - not per figure)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    name = f"instrument-{mode}"
    if name not in pio.templates:
        pio.templates[name] = go.layout.Template(layout=build_plotly_template(mode))
    pio.templates.default = name


def apply_theme(fig):
    """Kept for call sites that want explicit margin control after
    construction - color/font correctness comes from the registered default
    template (register_plotly_template()), not from this call."""
    fig.update_layout(margin={"t": 36, "l": 8, "r": 8, "b": 8})
    return fig


def inject_css() -> str:
    """Layer 1: pure CSS, no Python involvement in the color values
    themselves. Every color below the token block is var(--token) - the
    :root/@media/[data-theme] declarations are the only place a raw hex
    literal is allowed to appear."""
    return f"""
<style>
@import url('{GOOGLE_FONTS_URL}');

:root {{
    --surface: #FEFEFE; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
    --accent: #3355FF; --field-bg: #1B1F2A; --field-ink: #F2F4F8;
}}
@media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
        --surface: #0E1117; --ink: #ffffff; --ink2: #c3c2b7;
        --muted: #898781; --grid: #2c2c2a; --axis: #383835;
        --accent: #3355FF; --field-bg: #262A36; --field-ink: #F2F4F8;
    }}
}}
:root[data-theme="dark"] {{
    --surface: #0E1117; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --accent: #3355FF; --field-bg: #262A36; --field-ink: #F2F4F8;
}}

html, body, .stApp {{
    background-color: var(--surface) !important;
    color: var(--ink) !important;
    font-family: '{FONT_SANS}', sans-serif;
}}

#MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}

.block-container {{ padding-top: 1rem; max-width: 900px; }}

h1, h2, h3, h4 {{
    font-family: '{FONT_SANS}', sans-serif;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
}}

p, span, label, div {{ color: var(--ink); }}

/* the sidebar is a separate DOM subtree that paints its own background
(Streamlit's native secondaryBackgroundColor) on top of the page - the
generic html/body/.stApp rule above does not reach it, verified live (it
stayed light while the rest of the page went dark until this rule was
added). */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
    background: var(--surface) !important;
}}

/* status strip */
.status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px 20px;
    padding: 6px 0 10px 0;
    border-bottom: 1px solid var(--axis);
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 20px;
}}
.status-strip b {{ color: var(--ink); font-weight: 600; }}

/* zero radius everywhere, hairline instead of shadow */
.stButton>button, .stDownloadButton>button, input, select, textarea,
div[data-baseweb="select"] > div, div[data-baseweb="input"],
div[data-baseweb="base-input"], div[data-testid="stVerticalBlockBorderWrapper"],
.stTextInput input, .stNumberInput input {{
    border-radius: 0 !important;
    box-shadow: none !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: var(--axis) !important;
}}

/* buttons */
.stButton>button, .stDownloadButton>button {{
    background: var(--ink);
    color: var(--surface);
    border: 1px solid var(--ink);
    font-family: '{FONT_MONO}', monospace;
    font-weight: 500;
    font-size: 13px;
    padding: 6px 18px;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background: var(--accent);
    border-color: var(--accent);
    color: var(--surface);
}}
/* the blanket `p, span, label, div {{ color: var(--ink) }}` rule above would
otherwise override the inherited surface text color on the button's inner
label element - force it back explicitly. */
.stButton>button *, .stDownloadButton>button * {{
    color: var(--surface) !important;
}}
.stButton>button:hover *, .stDownloadButton>button:hover * {{
    color: var(--surface) !important;
}}

/* inputs - a consistent dark "console field" look in both modes (field-bg/
field-ink shift shade between modes but stay dark-bg/light-ink in both -
this is intentional, not a light/dark mismatch). */
input, select, textarea, div[data-baseweb="select"] > div,
div[data-baseweb="input"], div[data-baseweb="base-input"] {{
    background: var(--field-bg) !important;
    color: var(--field-ink) !important;
    border: 1px solid var(--axis) !important;
    font-family: '{FONT_MONO}', monospace !important;
    font-size: 13px !important;
}}
label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stSlider label {{
    font-family: '{FONT_MONO}', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

/* BaseWeb-style floating portals (the selectbox dropdown and the date
picker calendar): these render outside this app's normal component tree
with Streamlit's own native theme colors by default, not this stylesheet's
tokens - confirmed live (dropdown background rgb(255,255,255), option text
rgb(49,51,63) prior to this rule existing). Selectors verified against the
real DOM (div[data-testid="stSelectboxVirtualDropdown"] > ... >
div[role="option"], not the BaseWeb li[role="option"]/popover markup this
Streamlit version has moved away from). */
div[data-testid="stSelectboxVirtualDropdown"] {{
    background: var(--field-bg) !important;
    border: 1px solid var(--axis) !important;
}}
div[data-testid="stSelectboxVirtualDropdown"] div[role="option"] {{
    color: var(--field-ink) !important;
    background: var(--field-bg) !important;
}}
div[data-testid="stSelectboxVirtualDropdown"] div[role="option"][aria-selected="true"],
div[data-testid="stSelectboxVirtualDropdown"] div[role="option"]:hover {{
    background: var(--surface) !important;
    color: var(--ink) !important;
}}
div[data-testid="stDateInputCalendar"] {{
    background: var(--field-bg) !important;
    color: var(--field-ink) !important;
    border: 1px solid var(--axis) !important;
}}
div[data-testid="stDateInputCalendar"] * {{
    color: var(--field-ink) !important;
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid var(--axis);
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 13px;
    color: var(--muted);
    background: transparent;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: var(--ink) !important;
    border-bottom: 2px solid var(--accent) !important;
}}

/* metric / dense data panels */
.readout-panel {{
    border: 1px solid var(--axis);
    padding: 16px 20px;
    margin-bottom: 16px;
}}
.readout-label {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}}
.headline-number {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 64px;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.02em;
    color: var(--ink);
}}
.readout-sub {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
}}
.decision-stated {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    margin-top: 12px;
}}
.decision-abstain {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 15px;
    font-weight: 600;
    color: var(--muted);
    margin-top: 12px;
}}

hr {{ border-color: var(--axis); }}

/* AI advisor: flat blocks, no chat bubbles */
.qa-block {{
    border: 1px solid var(--axis);
    padding: 12px 16px;
    margin-bottom: 10px;
}}
.qa-label {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}}
.qa-question {{ font-family: '{FONT_SANS}', sans-serif; font-size: 14px; }}
.qa-answer {{ font-family: '{FONT_SANS}', sans-serif; font-size: 13px; line-height: 1.5; }}
.qa-answer code {{ font-family: '{FONT_MONO}', monospace; font-size: 12px; }}
</style>
"""
