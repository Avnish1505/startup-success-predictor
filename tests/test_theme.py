import inspect
import re

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
    assert list(inspect.signature(theme.inject_css).parameters) == []


def test_inject_css_has_no_raw_hex_outside_token_block():
    css = theme.inject_css()
    marker = ':root[data-theme="dark"]'
    after_token_block = css.split(marker, 1)[1]
    after_token_block = after_token_block.split("}", 1)[1]
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


@pytest.mark.parametrize("detected,expected", [("dark", "dark"), ("light", "light"), (None, "light"), ("", "light")])
def test_resolve_initial_theme(detected, expected):
    assert theme.resolve_initial_theme(detected) == expected
