"""Video analysis flow, timeline and frame stepping in the browser (S12)."""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.ui.conftest import open_app, wait_for_result
from tests.support.synthetic_video import write_moving_rectangle_video

pytestmark = [pytest.mark.ui, pytest.mark.enable_socket]


def _upload_video(page: Page, clip: Path) -> None:
    page.set_input_files("#file-input", str(clip))


def test_video_progress_timeline_and_arrow_keys(page, app_url: str, tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    open_app(page, app_url)
    _upload_video(page, clip)
    wait_for_result(page)

    expect(page.locator('[data-testid="progress-value"]')).to_have_text("100%")
    expect(page.locator('[data-testid="video-controls"]')).to_be_visible()
    bars = page.locator('[data-testid="timeline-bar"]')
    expect(bars).not_to_have_count(0)
    assert bars.count() == 20

    expect(page.locator('[data-testid="frame-position"]')).to_contain_text("Frame 1 of 20")
    page.locator("h1").click()
    page.keyboard.press("ArrowRight")
    expect(page.locator('[data-testid="frame-position"]')).to_contain_text("Frame 2 of 20")
    page.keyboard.press("ArrowLeft")
    expect(page.locator('[data-testid="frame-position"]')).to_contain_text("Frame 1 of 20")

    bars.nth(4).click()
    expect(page.locator('[data-testid="frame-position"]')).to_contain_text("Frame 5 of 20")
    expect(bars.nth(4)).to_have_attribute("aria-current", "true")


def test_video_summary_shows_unique_instances(page, app_url: str, tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    open_app(page, app_url)
    _upload_video(page, clip)
    wait_for_result(page)

    expect(page.locator("#tile-extra-label")).to_have_text("Unique instances")
    expect(page.locator("#tile-extra")).not_to_have_text("0")
    expect(page.locator("#summary-scope")).to_contain_text("20 frames")
