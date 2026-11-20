"""Error and empty states in the browser viewer (S13)."""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.e2e.ui.conftest import open_app

pytestmark = [pytest.mark.ui, pytest.mark.enable_socket]


def test_text_upload_shows_an_accessible_error(page, app_url: str, tmp_path: Path) -> None:
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("this is not an image or a video\n", encoding="utf-8")

    open_app(page, app_url)
    page.set_input_files("#file-input", str(not_an_image))

    alert = page.get_by_role("alert")
    expect(alert).to_be_visible()
    expect(alert).to_contain_text("Unsupported or corrupt media file")
    expect(page.locator('[data-testid="result-view"]')).to_be_hidden()
    expect(page.locator('[data-testid="progress-panel"]')).to_be_hidden()
    expect(page.locator('[data-testid="dropzone"]')).to_be_visible()
    expect(page.locator('[data-testid="status"]')).to_have_text("Upload failed.")


def test_error_is_replaced_by_the_next_successful_upload(
    page, app_url: str, tmp_path: Path, fixture_image: Path
) -> None:
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("nope\n", encoding="utf-8")
    open_app(page, app_url)
    page.set_input_files("#file-input", str(not_an_image))
    expect(page.get_by_role("alert")).to_be_visible()

    page.set_input_files("#file-input", str(fixture_image))
    expect(page.locator('[data-testid="result-view"]')).to_be_visible(timeout=45_000)
    expect(page.get_by_role("alert")).to_be_hidden()
