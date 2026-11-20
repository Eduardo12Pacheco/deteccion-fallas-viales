"""Image analysis flow in the real browser viewer (S12/S13)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from playwright.sync_api import expect

from tests.e2e.ui.conftest import drop_file, open_app, overlay_alpha_count, wait_for_result

pytestmark = [pytest.mark.ui, pytest.mark.enable_socket]


def test_drop_image_renders_overlay_and_filters_without_a_new_request(
    page, app_url: str, fixture_image: Path
) -> None:
    posts: list[str] = []
    page.on(
        "request",
        lambda request: (
            posts.append(request.url)
            if request.method == "POST" and "/api/analyses" in request.url
            else None
        ),
    )
    open_app(page, app_url)
    expect(page.locator('[data-testid="dropzone"]')).to_be_visible()
    expect(page.locator('[data-testid="empty-state"]')).to_be_hidden()

    drop_file(page, fixture_image)
    wait_for_result(page)

    assert overlay_alpha_count(page) > 0
    analysis_id = page.locator('[data-testid="result-view"]').get_attribute("data-analysis-id")
    assert analysis_id

    document = httpx.get(f"{app_url}/api/analyses/{analysis_id}", timeout=10.0).json()["result"][
        "document"
    ]
    threshold = float(page.locator('[data-testid="confidence-slider"]').input_value())
    expected_rows = sum(
        1
        for frame in document["frames"]
        for detection in frame["detections"]
        if float(detection["score"]) >= threshold
    )
    assert expected_rows > 0
    assert page.locator('[data-testid="detection-row"]').count() == expected_rows
    expect(page.locator('[data-testid="tile-total"]')).to_have_text(str(expected_rows))
    expect(page.locator('[data-testid="empty-state"]')).to_be_hidden()

    posts_before = len(posts)
    slider = page.locator('[data-testid="confidence-slider"]')
    slider.focus()
    page.keyboard.press("End")
    expect(slider).to_have_value("1")
    expect(page.locator('[data-testid="empty-state"]')).to_be_visible()
    expect(page.locator('[data-testid="empty-title"]')).to_have_text(
        "No detections match the filters"
    )
    assert page.locator('[data-testid="detection-row"]').count() == 0
    assert overlay_alpha_count(page) == 0
    page.wait_for_timeout(700)
    assert len(posts) == posts_before

    page.keyboard.press("Home")
    expect(page.locator('[data-testid="detection-row"]')).to_have_count(expected_rows)


def test_sorting_reorders_the_table_without_touching_the_canvas(
    page, app_url: str, fixture_image: Path
) -> None:
    open_app(page, app_url)
    drop_file(page, fixture_image)
    wait_for_result(page)

    rows = page.locator('[data-testid="detection-row"]')
    scores = [
        float(rows.nth(index).locator("td").nth(2).inner_text()) for index in range(rows.count())
    ]
    assert scores == sorted(scores, reverse=True)

    page.locator('th button[data-sort="score"]').click()
    ascending = [
        float(rows.nth(index).locator("td").nth(2).inner_text()) for index in range(rows.count())
    ]
    assert ascending == sorted(ascending)
    assert overlay_alpha_count(page) > 0
