"""Theme toggle and persistence (S13)."""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.e2e.ui.conftest import open_app

pytestmark = [pytest.mark.ui, pytest.mark.enable_socket]


def test_theme_toggle_persists_across_reload(page, app_url: str) -> None:
    open_app(page, app_url)
    html = page.locator("html")
    expect(html).to_have_attribute("data-theme", "light")
    light_background = page.evaluate("getComputedStyle(document.body).backgroundColor")

    page.locator('[data-testid="theme-toggle"]').click()
    expect(html).to_have_attribute("data-theme", "dark")
    expect(page.locator('[data-testid="theme-toggle"]')).to_have_attribute("aria-pressed", "true")
    dark_background = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert dark_background != light_background

    page.reload(wait_until="networkidle")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == dark_background

    page.locator('[data-testid="theme-toggle"]').click()
    expect(page.locator("html")).to_have_attribute("data-theme", "light")


def test_shortcut_t_toggles_the_theme(page, app_url: str) -> None:
    open_app(page, app_url)
    page.locator("h1").click()
    page.keyboard.press("t")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
