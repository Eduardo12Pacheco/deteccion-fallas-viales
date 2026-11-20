"""Keyboard-only operation of the viewer (S13)."""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.e2e.ui.conftest import open_app, overlay_alpha_count, wait_for_result

pytestmark = [pytest.mark.ui, pytest.mark.enable_socket]


def test_keyboard_shortcut_opens_the_picker_and_completes_the_flow(
    page, app_url: str, fixture_image: Path
) -> None:
    open_app(page, app_url)
    with page.expect_file_chooser() as chooser:
        page.keyboard.press("o")
    chooser.value.set_files(str(fixture_image))
    wait_for_result(page)
    expect(page.locator('[data-testid="result-view"]')).to_be_visible()
    assert overlay_alpha_count(page) > 0


def test_visible_focus_ring_on_header_controls(page, app_url: str) -> None:
    open_app(page, app_url)
    page.keyboard.press("Tab")  # skip link
    page.keyboard.press("Tab")  # theme toggle
    focused = page.evaluate("document.activeElement.id")
    assert focused == "theme-toggle"
    outline = page.evaluate(
        """() => {
            const style = getComputedStyle(document.activeElement);
            return { width: style.outlineWidth, style: style.outlineStyle };
        }"""
    )
    assert outline["style"] != "none"
    assert outline["width"] == "2px"
