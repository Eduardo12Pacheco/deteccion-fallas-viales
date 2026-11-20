"""Accessibility and color-token checks for the local web UI (S13).

The browser UI is checked without a browser: the CSS custom properties in
``static/css/tokens.css`` are parsed and every documented text/background and
component/background pair is evaluated with the WCAG 2.x relative-luminance
formula. Class colors are additionally compared with the CIEDE2000 color
difference so the four damage classes stay distinguishable in both themes.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS = REPO_ROOT / "src" / "rdv" / "entrypoints" / "web" / "static" / "css" / "tokens.css"

_BLOCK = re.compile(r"([^{}]+)\{([^{}]*)\}")
_VARIABLE = re.compile(r"--([a-z0-9-]+)\s*:\s*([^;]+);")
_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")

TEXT_ON_BACKGROUND = (
    ("text", "bg"),
    ("text", "surface"),
    ("text", "surface-2"),
    ("text-muted", "bg"),
    ("text-muted", "surface"),
    ("text-muted", "surface-2"),
    ("accent", "surface"),
    ("accent", "bg"),
    ("danger", "surface"),
    ("ok", "surface"),
    ("accent-contrast", "accent"),
)
COMPONENT_ON_BACKGROUND = (
    ("border", "bg"),
    ("border", "surface"),
    ("border", "surface-2"),
    ("focus", "bg"),
    ("focus", "surface"),
    ("accent", "surface"),
    ("accent", "bg"),
)
CLASS_COLORS = ("c-d00", "c-d10", "c-d20", "c-d40")


def _palettes() -> dict[str, dict[str, str]]:
    text = re.sub(r"/\*.*?\*/", "", TOKENS.read_text(encoding="utf-8"), flags=re.DOTALL)
    palettes: dict[str, dict[str, str]] = {}
    for selector, body in _BLOCK.findall(text):
        selector = " ".join(selector.split())
        if selector == ":root":
            palettes.setdefault("light", {}).update(_VARIABLE.findall(body))
        elif 'data-theme="dark"' in selector:
            # Both the explicit toggle and the prefers-color-scheme fallback
            # declare the dark palette; either copy is representative.
            palettes.setdefault("dark", {}).update(_VARIABLE.findall(body))
    if "light" in palettes and "dark" in palettes:
        # Class accents are declared once on :root and inherited by the dark
        # theme, exactly like the browser computes them.
        palettes["dark"] = {**palettes["light"], **palettes["dark"]}
    return palettes


def _channel(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _rgb(color: str) -> tuple[float, float, float]:
    match = _HEX.match(color.strip())
    assert match, f"token must be a 6-digit hex color, got {color!r}"
    raw = match.group(1)
    return tuple(int(raw[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def _luminance(color: str) -> float:
    red, green, blue = (_channel(channel) for channel in _rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio between two hex colors."""
    first = _luminance(foreground)
    second = _luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _lab(color: str) -> tuple[float, float, float]:
    red, green, blue = (_channel(channel) for channel in _rgb(color))
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    def f(value: float) -> float:
        return value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def ciede2000(first: str, second: str) -> float:
    """CIEDE2000 color difference (kL = kC = kH = 1)."""
    l1, a1, b1 = _lab(first)
    l2, a2, b2 = _lab(second)
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    delta_l = l2 - l1
    delta_c = c2p - c1p
    if c1p * c2p == 0:
        delta_hp = 0.0
    elif abs(h2p - h1p) <= 180:
        delta_hp = h2p - h1p
    elif h2p <= h1p:
        delta_hp = h2p - h1p + 360
    else:
        delta_hp = h2p - h1p - 360
    delta_big_h = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_hp) / 2)
    l_bar = (l1 + l2) / 2
    c_bar_p = (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        h_bar = (h1p + h2p + 360) / 2
    else:
        h_bar = (h1p + h2p - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    delta_theta = 30 * math.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * math.sqrt(c_bar_p**7 / (c_bar_p**7 + 25**7))
    s_l = 1 + (0.015 * (l_bar - 50) ** 2) / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = -math.sin(math.radians(2 * delta_theta)) * r_c
    return math.sqrt(
        (delta_l / s_l) ** 2
        + (delta_c / s_c) ** 2
        + (delta_big_h / s_h) ** 2
        + r_t * (delta_c / s_c) * (delta_big_h / s_h)
    )


def test_tokens_file_exists_and_declares_both_themes() -> None:
    palettes = _palettes()
    assert set(palettes) == {"light", "dark"}
    for mode, palette in palettes.items():
        missing = {
            "bg",
            "surface",
            "surface-2",
            "text",
            "text-muted",
            "border",
            "accent",
            "accent-contrast",
            "focus",
            "ok",
            "danger",
            *CLASS_COLORS,
        } - set(palette)
        assert not missing, f"{mode} palette is missing {sorted(missing)}"


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize(("foreground", "background"), TEXT_ON_BACKGROUND)
def test_body_text_contrast_meets_wcag_aa(mode: str, foreground: str, background: str) -> None:
    palette = _palettes()[mode]
    ratio = contrast_ratio(palette[foreground], palette[background])
    assert ratio >= 4.5, f"{mode}: {foreground} on {background} = {ratio:.2f}:1"


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize(("foreground", "background"), COMPONENT_ON_BACKGROUND)
def test_ui_component_contrast_meets_wcag_non_text(
    mode: str, foreground: str, background: str
) -> None:
    palette = _palettes()[mode]
    ratio = contrast_ratio(palette[foreground], palette[background])
    assert ratio >= 3.0, f"{mode}: {foreground} on {background} = {ratio:.2f}:1"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_damage_class_colors_stay_distinguishable(mode: str) -> None:
    palette = _palettes()[mode]
    for code in CLASS_COLORS:
        for background in ("bg", "surface"):
            delta = ciede2000(palette[code], palette[background])
            assert delta >= 20, f"{mode}: {code} vs {background} deltaE={delta:.1f}"
    for left_index, left in enumerate(CLASS_COLORS):
        for right in CLASS_COLORS[left_index + 1 :]:
            delta = ciede2000(palette[left], palette[right])
            assert delta >= 20, f"{mode}: {left} vs {right} deltaE={delta:.1f}"


def test_contrast_formula_known_values() -> None:
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)
    assert ciede2000("#ffffff", "#ffffff") == pytest.approx(0.0)
    assert ciede2000("#000000", "#ffffff") > 90
