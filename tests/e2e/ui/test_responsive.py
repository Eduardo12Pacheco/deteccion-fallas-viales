"""Responsive layout and reduced-motion behavior (S13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.ui.conftest import drop_file, open_app, wait_for_result

pytestmark = [pytest.mark.ui, pytest.mark.enable_socket]


def test_360px_viewport_has_no_horizontal_scroll(page, app_url: str, fixture_image: Path) -> None:
    page.set_viewport_size({"width": 360, "height": 740})
    open_app(page, app_url)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    drop_file(page, fixture_image)
    wait_for_result(page)
    page.wait_for_timeout(200)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_reduced_motion_runs_no_animations(page, app_url: str) -> None:
    page.emulate_media(reduced_motion="reduce")
    open_app(page, app_url)
    running = page.evaluate(
        "document.getAnimations().filter((animation) => animation.playState === 'running').length"
    )
    assert running == 0
    duration = page.evaluate(
        "getComputedStyle(document.querySelector('.dropzone')).transitionDuration"
    )
    assert duration in {"0.01ms", "0s"}
