"""Shared visual tokens for the Streamlit UI: a technical-instrument
aesthetic (near-white/near-black, zero border radius, hairline borders,
monospace numbers) used by both app.py and analytics.py. Not copied from
any specific product - an original palette/type system built to match this
project's own calibrated-probability output."""

BG = "#FEFEFE"
FG = "#1E1E1E"
BORDER = "#D8D8D2"
MUTED = "#6B6B63"
ACCENT = "#2E4BFF"

FONT_SANS = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600;700&display=swap"
)

PLOTLY_LAYOUT = {
    "font": {"family": f"{FONT_MONO}, monospace", "size": 12, "color": FG},
    "paper_bgcolor": BG,
    "plot_bgcolor": BG,
    "xaxis": {"gridcolor": BORDER, "zerolinecolor": BORDER, "linecolor": BORDER},
    "yaxis": {"gridcolor": BORDER, "zerolinecolor": BORDER, "linecolor": BORDER},
    "margin": {"t": 36, "l": 8, "r": 8, "b": 8},
    "title": {"font": {"family": f"{FONT_MONO}, monospace", "size": 13, "color": FG}},
    "legend": {"font": {"family": f"{FONT_MONO}, monospace", "size": 11}},
}


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def inject_css() -> str:
    return f"""
<style>
@import url('{GOOGLE_FONTS_URL}');

html, body, .stApp {{
    background-color: {BG} !important;
    color: {FG} !important;
    font-family: '{FONT_SANS}', sans-serif;
}}

#MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}

.block-container {{ padding-top: 1rem; max-width: 900px; }}

h1, h2, h3, h4 {{
    font-family: '{FONT_SANS}', sans-serif;
    font-weight: 600;
    color: {FG};
    letter-spacing: -0.01em;
}}

p, span, label, div {{ color: {FG}; }}

/* status strip */
.status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px 20px;
    padding: 6px 0 10px 0;
    border-bottom: 1px solid {BORDER};
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
    color: {MUTED};
    margin-bottom: 20px;
}}
.status-strip b {{ color: {FG}; font-weight: 600; }}

/* zero radius everywhere, hairline instead of shadow */
.stButton>button, .stDownloadButton>button, input, select, textarea,
div[data-baseweb="select"] > div, div[data-baseweb="input"],
div[data-baseweb="base-input"], div[data-testid="stVerticalBlockBorderWrapper"],
.stTextInput input, .stNumberInput input {{
    border-radius: 0 !important;
    box-shadow: none !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: {BORDER} !important;
}}

/* buttons */
.stButton>button, .stDownloadButton>button {{
    background: {FG};
    color: {BG};
    border: 1px solid {FG};
    font-family: '{FONT_MONO}', monospace;
    font-weight: 500;
    font-size: 13px;
    padding: 6px 18px;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: {BG};
}}
/* the blanket `p, span, label, div {{ color: FG }}` rule below would
otherwise override the inherited BG text color on the button's inner
label element - force it back explicitly. */
.stButton>button *, .stDownloadButton>button * {{
    color: {BG} !important;
}}
.stButton>button:hover *, .stDownloadButton>button:hover * {{
    color: {BG} !important;
}}

/* inputs */
input, select, textarea, div[data-baseweb="select"] > div {{
    border: 1px solid {BORDER} !important;
    font-family: '{FONT_MONO}', monospace !important;
    font-size: 13px !important;
}}
label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stSlider label {{
    font-family: '{FONT_MONO}', monospace !important;
    font-size: 11px !important;
    color: {MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid {BORDER};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 13px;
    color: {MUTED};
    background: transparent;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: {FG} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* metric / dense data panels */
.readout-panel {{
    border: 1px solid {BORDER};
    padding: 16px 20px;
    margin-bottom: 16px;
}}
.readout-label {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
    color: {MUTED};
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
    color: {FG};
}}
.readout-sub {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 13px;
    color: {MUTED};
    margin-top: 6px;
}}
.decision-stated {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 15px;
    font-weight: 600;
    color: {ACCENT};
    margin-top: 12px;
}}
.decision-abstain {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 15px;
    font-weight: 600;
    color: {MUTED};
    margin-top: 12px;
}}

hr {{ border-color: {BORDER}; }}

/* AI advisor: flat blocks, no chat bubbles */
.qa-block {{
    border: 1px solid {BORDER};
    padding: 12px 16px;
    margin-bottom: 10px;
}}
.qa-label {{
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}}
.qa-question {{ font-family: '{FONT_SANS}', sans-serif; font-size: 14px; }}
.qa-answer {{ font-family: '{FONT_SANS}', sans-serif; font-size: 13px; line-height: 1.5; }}
.qa-answer code {{ font-family: '{FONT_MONO}', monospace; font-size: 12px; }}
</style>
"""
