# Light/Dark Theme Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Streamlit app render correctly in both light and dark viewer themes - page chrome, Plotly charts, and the BaseWeb selectbox/date-picker popovers - without depending solely on the unreliable `st.context.theme.type` signal.

**Architecture:** Three independent layers that must never contradict each other:
1. Pure CSS (`:root` custom properties, declared three times: light default, `@media (prefers-color-scheme: dark)`, `[data-theme="dark"]`) - correct on first paint with zero Python involvement.
2. Plotly figures made theme-agnostic via transparent `paper_bgcolor`/`plot_bgcolor` (page background shows through) plus a single muted tick/axis color (`#898781`) that is identical in both modes.
3. An app-owned sidebar toggle, seeded best-effort from `st.context.theme.type`, that drives both the Plotly template (rebuilt per theme, cached) and a `data-theme` attribute stamped onto the top-level `<html>` via a same-origin iframe script (verified working - see "Verified real state" below), so CSS and Plotly can never disagree once the toggle has run once.

**Tech Stack:** Streamlit 1.64.0 (confirms `[theme.light]`/`[theme.dark]` config.toml sections and `st.context.theme` are both supported), Plotly, existing `theme.py`/`app.py`/`analytics.py`.

**Spec:** User-pasted "principal AI engineer" theme spec (this conversation, 2026-09-22) - exact palette hex values, the three-declaration CSS guard pattern, and the acceptance checklist are given verbatim there and are not repeated in full here; this plan implements them literally.

## Verified real state - corrections/confirmations before writing code

- **Streamlit version is 1.64.0.** `streamlit.config` exposes `theme.light.*`/`theme.dark.*`/`theme.sidebar.*` config options (confirmed via `_create_theme_options` calls in the installed package) and `st.context.theme` exists. No version gate needed.
- **No `.streamlit/config.toml` exists yet** - this is a new file, not an edit.
- **The spec's "existing accent #3355FF" claim is not quite right**: the app's actual current accent (`theme.ACCENT`) is `#2E4BFF`, not `#3355FF`. The spec's CSS block hardcodes `--accent: #3355FF` in all three declarations, which is an explicit instruction, not a diagnosis of current state - implementing it literally changes the accent color. Contrast verified independently (WCAG relative-luminance formula, not trusted blindly): `#3355FF` vs `#FEFEFE` = 5.36:1, vs `#0E1117` = 3.50:1 - both clear 3:1, the spec's claim holds.
- **Muted `#898781` contrast verified**: vs `#FEFEFE` = 3.56:1, vs `#0E1117` = 5.26:1. Both clear 3:1 (appropriate for de-emphasized axis text, not body text).
- **No `template=` argument exists anywhere** in `app.py`/`analytics.py` (grepped project-wide, excluding `.venv`) - the only match for `template` is `texttemplate=` in `analytics.py` (bar-label text formatting, unrelated). Nothing to remove for that audit step.
- **No hardcoded hex values exist** in `app.py`, `analytics.py`, or `advisor_ai.py` today - all color usage already routes through `theme.py` constants. The only hex values to touch are inside `theme.py` itself.
- **The BaseWeb selectbox popover does not use the selectors the spec assumes.** This Streamlit version renders the open dropdown as `div[data-testid="stSelectboxVirtualDropdown"]` containing `div[role="listbox"]` > `div[role="option"]` rows - not `li[role="option"]` or `div[data-baseweb="popover"]` (verified live via Playwright DOM walk). Measured current computed style: dropdown background `rgb(255,255,255)`, option text `rgb(49,51,63)` - both Streamlit's own native light-theme defaults, completely independent of this app's custom CSS today. This confirms the spec's warning is real: without explicit rules targeting `stSelectboxVirtualDropdown`, the dropdown stays light regardless of what the rest of the page does.
- **The date-input calendar popup has the same exposure**, via `div[data-testid="stDateInputCalendar"]` (verified live) - not called out in the spec by name, but it is the same class of bug (a BaseWeb-style floating portal that doesn't inherit this app's custom CSS), so it gets the same token-based treatment.
- **The `window.parent.document.documentElement.setAttribute(...)` stamping mechanism was prototyped and confirmed working**: a same-origin iframe (which is what `st.components.v1.html` renders under the hood) can reach into the parent document and set an attribute on `<html>` that the parent page's own CSS then reacts to. Confirmed live: after running the script, `document.documentElement.getAttribute('data-theme')` on the *parent* page returned `"dark"`. This is the mechanism Layer 3 uses to make the `[data-theme="dark"]` CSS selector actually fire from Python-driven state.

## Global Constraints

- Palette hex values are exact and copied verbatim from the spec - do not adjust, round, or "improve" any of them.
- `--field-bg`/`--field-ink` differ between light-default and dark (`#1B1F2A`/`#F2F4F8` light, `#262A36`/`#F2F4F8` dark) - this is intentional (a consistent dark "console field" look for all editable inputs regardless of overall page mode), not a spec typo. Do not unify them.
- Series colors beyond the single shared accent use the three validated pairs in fixed order (never cycled/reassigned): slot 1 `#2a78d6`/`#3987e5`, slot 2 `#eb6834`/`#d95926`, slot 3 `#1baf7a`/`#199e70` (light/dark). No current chart needs more than the accent + one semantic secondary (SHAP contribution bars already use accent/ink for increase/decrease, which is a semantic encoding, not a "series" - leave that as-is). Ship the validated list as a `SERIES_COLORS` constant for future use; do not build unused plumbing around it.
- Do not add a `template=` argument to any individual `px.*`/`go.*` call - the registered default template is the single source of truth.
- Rebuild the Plotly template once per theme change (cache via `functools.lru_cache`), not on every figure.

---

## Task 1: `theme.py` - tokens, CSS (Layer 1), Plotly template builder (Layer 2/3), series colors

**Files:**
- Modify: `theme.py` (near-total rewrite)
- Test: `tests/test_theme.py` (new)

**Interfaces:**
- Produces: `TOKENS: dict[str, dict[str, str]]` (keys `"light"`/`"dark"`, each an 8-key dict: `surface, ink, ink2, muted, grid, axis, accent, field_bg, field_ink`); `SERIES_COLORS: dict[str, list[str]]`; `MUTED = "#898781"` (module-level constant, same both modes); `FONT_SANS`, `FONT_MONO`, `GOOGLE_FONTS_URL` (unchanged from current file); `inject_css() -> str` (theme-independent - no mode argument, per Layer 1's "no Python involvement"); `build_plotly_template(mode: str) -> dict` (pure function, cached, returns a `go.layout.Template`-compatible layout dict); `register_plotly_template(mode: str) -> None` (registers + sets default, calls the cached builder); `apply_theme(fig)` (unchanged signature/behavior from current file - margin-only).
- Consumes: nothing (this is the base module).

- [x] **Step 1: Write `tests/test_theme.py` - contrast ratios and pure-function behavior**

```python
import pytest

import theme


def _relative_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(c1: str, c2: str) -> float:
    l1, l2 = _relative_luminance(c1), _relative_luminance(c2)
    l1, l2 = max(l1, l2), min(l1, l2)
    return (l1 + 0.05) / (l2 + 0.05)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_accent_clears_3to1_against_its_surface(mode):
    tokens = theme.TOKENS[mode]
    assert _contrast(tokens["accent"], tokens["surface"]) >= 3.0


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_muted_clears_3to1_against_its_surface(mode):
    tokens = theme.TOKENS[mode]
    assert _contrast(theme.MUTED, tokens["surface"]) >= 3.0


def test_muted_is_identical_in_both_modes():
    assert theme.TOKENS["light"]["muted"] == theme.TOKENS["dark"]["muted"] == theme.MUTED


def test_accent_is_identical_in_both_modes():
    assert theme.TOKENS["light"]["accent"] == theme.TOKENS["dark"]["accent"]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_ink_clears_body_text_contrast(mode):
    tokens = theme.TOKENS[mode]
    assert _contrast(tokens["ink"], tokens["surface"]) >= 7.0  # AAA for normal text


def test_series_colors_have_three_slots_for_both_modes():
    assert len(theme.SERIES_COLORS["light"]) == 3
    assert len(theme.SERIES_COLORS["dark"]) == 3


def test_build_plotly_template_is_transparent():
    tpl = theme.build_plotly_template("dark")
    assert tpl["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert tpl["plot_bgcolor"] == "rgba(0,0,0,0)"


def test_build_plotly_template_axis_uses_muted_not_mode_ink():
    light_tpl = theme.build_plotly_template("light")
    dark_tpl = theme.build_plotly_template("dark")
    assert light_tpl["xaxis"]["tickfont"]["color"] == theme.MUTED
    assert dark_tpl["xaxis"]["tickfont"]["color"] == theme.MUTED
    assert light_tpl["xaxis"]["linecolor"] == theme.MUTED


def test_build_plotly_template_gridcolor_differs_by_mode():
    light_tpl = theme.build_plotly_template("light")
    dark_tpl = theme.build_plotly_template("dark")
    assert light_tpl["xaxis"]["gridcolor"] == theme.TOKENS["light"]["grid"]
    assert dark_tpl["xaxis"]["gridcolor"] == theme.TOKENS["dark"]["grid"]
    assert light_tpl["xaxis"]["gridcolor"] != dark_tpl["xaxis"]["gridcolor"]


def test_build_plotly_template_colorway_is_shared_accent():
    tpl = theme.build_plotly_template("light")
    assert tpl["colorway"] == [theme.TOKENS["light"]["accent"]]


def test_build_plotly_template_is_cached_not_rebuilt():
    a = theme.build_plotly_template("light")
    b = theme.build_plotly_template("light")
    assert a is b  # lru_cache returns the identical object, proving no rebuild


def test_inject_css_takes_no_mode_argument():
    import inspect
    assert list(inspect.signature(theme.inject_css).parameters) == []


def test_inject_css_has_no_raw_hex_outside_token_block():
    css = theme.inject_css()
    # Isolate the :root/media/data-theme token declarations, then check
    # everything AFTER them for stray hex literals.
    marker = ":root[data-theme=\"dark\"]"
    after_token_block = css.split(marker, 1)[1]
    after_token_block = after_token_block.split("}", 1)[1]  # skip past the closing brace of that block
    import re
    assert re.search(r"#[0-9A-Fa-f]{3,6}\b", after_token_block) is None


def test_inject_css_defines_all_three_declarations():
    css = theme.inject_css()
    assert ":root {" in css.replace("\n", " ") or ":root{" in css.replace(" ", "")
    assert "@media (prefers-color-scheme: dark)" in css
    assert '[data-theme="dark"]' in css


def test_inject_css_styles_the_selectbox_dropdown_portal():
    css = theme.inject_css()
    assert "stSelectboxVirtualDropdown" in css


def test_inject_css_styles_the_date_calendar_portal():
    css = theme.inject_css()
    assert "stDateInputCalendar" in css
```

- [x] **Step 2: Run the new tests to confirm they fail**

Run: `pytest tests/test_theme.py -v`
Expected: FAIL - `theme.TOKENS`/`theme.MUTED`/`theme.SERIES_COLORS`/`theme.build_plotly_template` don't exist yet, `inject_css()` still takes no positional args by luck but has no dropdown/calendar CSS, so the last two tests fail.

- [x] **Step 3: Rewrite `theme.py`**

```python
"""Shared visual tokens for the Streamlit UI. Palette is theme-aware (light
and dark), selected per-mode from the same hue ramp - not a naive CSS invert.
See docs/superpowers/plans/2026-09-22-theme-support.md for the design."""

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
# #FEFEFE and #0E1117, so chart axis text stays legible even if theme
# detection lags by one interaction (see Layer 2 in the plan doc).
MUTED = "#898781"

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
.stButton>button *, .stDownloadButton>button * {{
    color: var(--surface) !important;
}}
.stButton>button:hover *, .stDownloadButton>button:hover * {{
    color: var(--surface) !important;
}}

/* inputs - a consistent dark "console field" look in both modes */
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

/* BaseWeb-style floating portals: this app's own CSS does not reach these
by default (they render with Streamlit's native theme colors, independent
of this stylesheet) - verified live, see the plan's "Verified real state". */
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
```

- [x] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_theme.py -v`
Expected: PASS, all tests.

- [x] **Step 5: Commit**

```bash
git add theme.py tests/test_theme.py
git commit -m "feat(theme): rewrite theme.py with light/dark token palette, transparent Plotly template, and dropdown/calendar portal styling"
```

---

## Task 2: `.streamlit/config.toml` - base theme sections

**Files:**
- Create: `.streamlit/config.toml`

**Interfaces:** None (config file, no Python).

- [x] **Step 1: Create the config file**

```toml
[theme]
base = "light"
font = "monospace"

[theme.light]
backgroundColor = "#FEFEFE"
secondaryBackgroundColor = "#F2F1EC"
textColor = "#0b0b0b"
primaryColor = "#3355FF"

[theme.dark]
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#171A21"
textColor = "#ffffff"
primaryColor = "#3355FF"
```

(`secondaryBackgroundColor` values are new - not given explicitly in the spec's token block, since that block only defines the CSS custom properties this app's own stylesheet uses, not Streamlit's native theme keys. Chosen as a small step between `backgroundColor` and `surface`/`grid` in each mode, for Streamlit's own native chrome - e.g. the sidebar background - that this app's CSS does not override.)

- [x] **Step 2: Verify Streamlit accepts the config without error**

Run: `.venv/bin/python3 -c "import streamlit.config as config; config.get_config_options()"` and separately start the app (`streamlit run app.py --server.headless true`) and confirm no config-parsing error in the log.

- [x] **Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat(theme): add Streamlit native light/dark base theme config"
```

---

## Task 3: `app.py` - sidebar toggle, theme resolution/seeding, document stamping

**Files:**
- Modify: `app.py`
- Test: `tests/test_theme.py` (extend with the pure seeding-logic test)

**Interfaces:**
- Consumes: `theme.TOKENS`, `theme.register_plotly_template(mode)`, `theme.inject_css()` (Task 1).
- Produces: a module-level `resolve_initial_theme(detected: str | None) -> str` pure function in `theme.py` (moved here rather than app.py so it's unit-testable without a Streamlit runtime); `st.session_state["app_theme"]` (`"light"`/`"dark"`) as the single source of truth used by every later part of `app.py`.

- [x] **Step 1: Add the seeding function to `theme.py` and its test**

Add to `theme.py`:
```python
def resolve_initial_theme(detected: str | None) -> str:
    """Best-effort seed for the FIRST run only. st.context.theme.type can be
    absent (pre-1.46), wrong on the first script run (streamlit#11920), and
    does not trigger a rerun on an OS-level theme change (streamlit#15287) -
    so this is only ever used to pick a starting value; st.session_state
    owns the truth after that."""
    return "dark" if detected == "dark" else "light"
```

Add to `tests/test_theme.py`:
```python
@pytest.mark.parametrize("detected,expected", [("dark", "dark"), ("light", "light"), (None, "light"), ("", "light")])
def test_resolve_initial_theme(detected, expected):
    assert theme.resolve_initial_theme(detected) == expected
```

Run: `pytest tests/test_theme.py -v` - expect the new test to pass immediately (simple function), confirm no regressions in the rest of the file.

- [x] **Step 2: Wire the sidebar control and document-stamping into `app.py`**

Replace the current:
```python
st.markdown(theme.inject_css(), unsafe_allow_html=True)
theme.register_plotly_template()
```
with:
```python
import streamlit.components.v1 as components

if "app_theme" not in st.session_state:
    try:
        detected = st.context.theme.type
    except Exception:
        detected = None
    st.session_state["app_theme"] = theme.resolve_initial_theme(detected)

st.markdown(theme.inject_css(), unsafe_allow_html=True)

with st.sidebar:
    theme_label = st.radio(
        "Theme", ["Light", "Dark"],
        index=0 if st.session_state["app_theme"] == "light" else 1,
        horizontal=True, key="app_theme_radio",
    )
st.session_state["app_theme"] = "dark" if theme_label == "Dark" else "light"

# Stamp [data-theme] onto the top-level <html> from Python-driven state, so
# the CSS's :root[data-theme="dark"] selector (Layer 1's explicit-override
# clause) actually fires - prototyped and confirmed working via a
# same-origin iframe reaching window.parent.document (see the plan's
# "Verified real state"). st.markdown-injected <script> tags do not execute
# (React's dangerouslySetInnerHTML does not run scripts) - components.html
# is the only reliable path for this in Streamlit.
components.html(
    f"<script>window.parent.document.documentElement.setAttribute('data-theme', '{st.session_state['app_theme']}');</script>",
    height=0,
)

theme.register_plotly_template(st.session_state["app_theme"])
```

Note: `st.radio`'s `index=` only seeds the FIRST render for a given `key`; on reruns Streamlit reads the widget's own tracked state for that `key`, so this does not fight the user's later clicks - confirmed against Streamlit's documented controlled-widget behavior, verify live in Step 4 below by clicking the toggle twice in a row and confirming it doesn't snap back.

- [x] **Step 3: Run the full test suite to confirm no regressions**

Run: `pytest -q`
Expected: all previously-passing tests still pass, new theme tests pass.

- [x] **Step 4: Live-verify the toggle doesn't fight itself**

Start the app (`streamlit run app.py --server.headless true --server.port <port>`), drive it with Playwright: click the sidebar Dark radio option, screenshot, click Light, screenshot, click Dark again - confirm each click actually lands (the radio's visual selected state matches the click) and the page chrome + a chart's background genuinely flip each time. Stop the server after.

- [x] **Step 5: Commit**

```bash
git add app.py theme.py tests/test_theme.py
git commit -m "feat(theme): add sidebar light/dark toggle, best-effort st.context.theme seeding, and document data-theme stamping"
```

---

## Task 4: Live verification against every acceptance bullet + README screenshots

**Files:**
- Modify: `README.md` (new screenshot references + a short "Theming" note documenting the Streamlit-menu-theme interaction)
- Create: `docs/screenshots/dark/*.png` (four new screenshots, one per existing light-mode screenshot)

No test-writing in this task - it is Playwright-driven live verification plus documentation, matching the acceptance checklist bullet-for-bullet.

- [x] **Step 1: OS dark, app never touched**

Launch a fresh Playwright browser context with `color_scheme="dark"` (Playwright's OS-preference emulation - no in-app interaction at all, no session_state, no clicks), load the app, screenshot immediately after first paint. Confirm: page background is dark, a chart (e.g. after a default Predict click) is not stuck light. Note: since `st.context.theme.type` is only read once `st.session_state["app_theme"]` doesn't exist (first run) and the OS is emulated dark, this exercises the seeding path end-to-end, not just the pure function in isolation.

- [x] **Step 2: Toggle the in-app control both directions**

From a light-OS context, click Dark in the sidebar - screenshot immediately (one interaction, no intermediate reload) and confirm chrome AND a chart's gridlines/axis both changed in that same screenshot (no frame where one has flipped and the other hasn't). Click Light - repeat.

- [x] **Step 3: Toggle the Streamlit-menu theme and document what happens**

Open Streamlit's own hamburger menu (`Settings > Theme`), switch its theme independent of this app's sidebar control. Screenshot. Expected (write up whichever is actually observed, do not assume): Streamlit's own native chrome (anything not covered by this app's CSS overrides, e.g. the sidebar's own background via `secondaryBackgroundColor`) may follow the menu's choice while this app's custom-styled surfaces and charts stay wherever `st.session_state["app_theme"]` last left them, since the menu toggle doesn't touch `st.context.theme.type` reliably (the two GitHub issues cited in the spec) and doesn't trigger this app's own sidebar radio. Document the actual observed behavior in README, and confirm clicking this app's own sidebar toggle (even re-selecting the already-active option, to force a rerun) re-stamps `data-theme` and rebuilds the Plotly template, resolving any mismatch - screenshot that recovery step too.

- [x] **Step 4: Every selectbox, both modes**

In both light and dark (in-app toggle), open Country/Region/Primary category, screenshot each open state, confirm every visible option is legible against `--field-bg`/`--field-ink`.

- [x] **Step 5: Every chart's axis labels, both modes**

In both modes, screenshot every Plotly chart (Predictor: SHAP bar, PDP/sensitivity, gauge; Analytics: country/category/funding-band/age-band) and pixel-sample the axis-label regions the same way as the prior correctness-pass session (crop generously, `argmin` over the full region - not a guessed narrow row, which gave a false alarm last time). Confirm ticks/titles read as `MUTED` (`#898781` = `rgb(137,135,129)`) in both modes, and gridlines differ by mode (`#e1e0d9` light / `#2c2c2a` dark).

- [x] **Step 6: Screenshot every tab in both modes for the README**

Re-use the existing `docs/screenshots/*.png` light-mode set (already current from the prior session's Task 10) and add a `docs/screenshots/dark/` set: `predictor_form.png`, `predictor_result.png`, `analytics.png`, `advisor.png`, captured the same way (375px width, same interaction sequence) but with the sidebar toggle set to Dark.

- [x] **Step 7: Update README**

Add dark-mode screenshots alongside the light ones (or a toggled/paired presentation - whichever reads cleanly given the existing README structure) and a short "Theming" subsection documenting: the three-layer architecture in one sentence each, and the Streamlit-menu-theme interaction finding from Step 3 verbatim (real observed behavior, not assumed).

- [x] **Step 8: Full suite + lint, then commit**

Run: `pytest -q && ruff check src/ tests/ api.py theme.py app.py analytics.py`
Expected: all pass, clean.

```bash
git add README.md docs/screenshots/
git commit -m "docs: add dark-mode screenshots and theming notes to README"
```

---

## Self-review notes

- Spec coverage: Layer 1 (Task 1 Step 3's `:root`/`@media`/`[data-theme]` block + var()-only rules below it), Layer 2 (Task 1's `build_plotly_template` - transparent bg, shared MUTED axis color), Layer 3 (Task 3 - sidebar toggle, seeding, session_state, document stamping, per-theme template rebuild+cache), palette rules (Task 1's exact TOKENS/SERIES_COLORS), config.toml check (Task 2, version confirmed to support it), `template=` audit (confirmed clean in "Verified real state", no task needed), hardcoded-hex grep (confirmed clean outside theme.py in "Verified real state"), all 6 acceptance bullets (Task 4, one sub-step each).
- No placeholders: every step carries full code or an exact command.
- Type consistency: `theme.TOKENS[mode]` dict keys (`surface, ink, ink2, muted, grid, axis, accent, field_bg, field_ink`) are used identically in `build_plotly_template` (Task 1) and referenced by name in Task 3's commentary; `st.session_state["app_theme"]` is written once in Task 3 Step 2 and is the only place any later task should read theme state from.
- Correction carried forward: the spec's "existing accent #3355FF" framing doesn't match the actual current `theme.ACCENT` (`#2E4BFF`) - implementing the spec's literal CSS still changes the accent color; this is noted in "Verified real state" and will be called out in the final report rather than silently absorbed into a commit message as if nothing changed.
