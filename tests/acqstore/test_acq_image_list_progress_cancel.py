"""Tests for AcqImageList cooperative load progress and cancellation."""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image.acq_image_list import AcqImageList, LoadCancelled, PathKind


class FakeAcqImage:
    """Small file object for AcqImageList load tests.

    Args:
        path: Source path.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.file_id = path


def test_load_safe_reports_progress_for_file(tmp_path: Path) -> None:
    """load_safe should report discovery and loaded progress."""
    source = tmp_path / 'a.tif'
    source.write_text('x', encoding='utf-8')
    progress: list[tuple[int, int, str]] = []

    result = AcqImageList.load_safe(
        str(source),
        kind=PathKind.FILE,
        file_factory=FakeAcqImage,
        progress_callback=lambda completed, total, message: progress.append((completed, total, message)),
    )

    assert len(result.acq_image_list) == 1
    assert progress[0][0:2] == (0, 1)
    assert progress[-1][0:2] == (1, 1)


def test_load_safe_cancels_before_loading_file(tmp_path: Path) -> None:
    """load_safe should raise LoadCancelled when cancellation is requested."""
    source = tmp_path / 'a.tif'
    source.write_text('x', encoding='utf-8')

    try:
        AcqImageList.load_safe(
            str(source),
            kind=PathKind.FILE,
            file_factory=FakeAcqImage,
            should_cancel=lambda: True,
        )
    except LoadCancelled:
        pass
    else:
        raise AssertionError('Expected LoadCancelled')
