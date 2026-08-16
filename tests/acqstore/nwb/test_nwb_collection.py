"""Tests for lazy multi-AcqImage storage in one NWB file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pynwb")
from pynwb.validation import validate

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList, PathKind
from acqstore.nwb_io import load_nwb_collection, save_nwb_collection


def _make_image(
    *,
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    source_id: str,
    load_images: bool = True,
) -> AcqImage:
    """Create one deterministic collection member.

    Args:
        axes: AcqStore axes, either YX or CYX.
        shape: Pixel shape corresponding to ``axes``.
        source_id: Logical source identity.
        load_images: Whether the AcqPixels wrapper starts loaded.

    Returns:
        Deterministic uint16 AcqImage.
    """
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    return AcqImage.from_array(
        data,
        axes=axes,
        source_id=source_id,
        axis_spacing={"Y": 0.002, "X": 0.25},
        axis_units={"Y": "s" if axes == ("Y", "X") else "um", "X": "um"},
        load_images=load_images,
    )


def _make_collection(images: list[AcqImage]) -> AcqImageList:
    """Assemble an in-memory AcqImageList using its established internal layout.

    Args:
        images: Ordered AcqImages to include.

    Returns:
        In-memory AcqImageList for serializer tests.
    """
    collection = AcqImageList.__new__(AcqImageList)
    collection.path = "memory://nwb-collection-test"
    collection.source_root_path = None
    collection.file_list = [image.file_id for image in images]
    collection._files = list(images)
    collection._files_by_id = {image.file_id: image for image in images}
    collection._attach_analysis_pools()
    return collection


def test_collection_load_is_lazy_and_ids_are_unique(tmp_path: Path) -> None:
    """Different shapes should share one NWB without eager data or ID collisions."""
    first = _make_image(axes=("C", "Y", "X"), shape=(2, 32, 48), source_id="first")
    second = _make_image(axes=("Y", "X"), shape=(300, 11), source_id="second")
    collection = _make_collection([first, second])
    path = tmp_path / "collection.nwb"

    save_nwb_collection(collection, path)
    loaded = load_nwb_collection(path)
    members = list(loaded)

    assert len(members) == 2
    assert len({member.file_id for member in members}) == 2
    assert all(member.path == str(path.resolve()) for member in members)
    assert all(member.images_loaded is False for member in members)

    members[0].load_images()
    assert members[0].images_loaded is True
    assert members[1].images_loaded is False
    assert members[0].pixels.shape == (2, 32, 48)

    members[0].unload_images()
    members[1].load_images()
    assert members[0].images_loaded is False
    assert members[1].pixels.shape == (300, 11)


def test_collection_export_restores_each_lazy_member(tmp_path: Path) -> None:
    """Collection export should not leave every source member resident in memory."""
    first = _make_image(
        axes=("C", "Y", "X"),
        shape=(2, 24, 31),
        source_id="first",
        load_images=False,
    )
    second = _make_image(
        axes=("Y", "X"),
        shape=(80, 9),
        source_id="second",
        load_images=False,
    )
    collection = _make_collection([first, second])
    path = tmp_path / "lazy-export.nwb"

    assert first.images_loaded is False
    assert second.images_loaded is False
    save_nwb_collection(collection, path)

    assert first.images_loaded is False
    assert second.images_loaded is False
    loaded = load_nwb_collection(path)
    assert all(member.images_loaded is False for member in loaded)


def test_reexport_of_lazy_nwb_collection_preserves_lazy_state(tmp_path: Path) -> None:
    """Re-export should load/write/unload one NWB member at a time."""
    original = _make_collection(
        [
            _make_image(axes=("C", "Y", "X"), shape=(2, 16, 20), source_id="first"),
            _make_image(axes=("Y", "X"), shape=(40, 7), source_id="second"),
        ]
    )
    first_path = tmp_path / "first.nwb"
    second_path = tmp_path / "second.nwb"
    save_nwb_collection(original, first_path)

    lazy = load_nwb_collection(first_path)
    assert all(member.images_loaded is False for member in lazy)

    save_nwb_collection(lazy, second_path)

    assert all(member.images_loaded is False for member in lazy)
    copied = load_nwb_collection(second_path, load_images=True)
    assert [member.pixels.shape for member in copied] == [(2, 16, 20), (40, 7)]


def test_collection_convenience_api(tmp_path: Path) -> None:
    """AcqImageList convenience wrappers should delegate to canonical NWB I/O."""
    original = _make_collection(
        [_make_image(axes=("Y", "X"), shape=(8, 12), source_id="first")]
    )
    path = tmp_path / "convenience.nwb"

    original.save_as_nwb(path)
    loaded = AcqImageList.from_nwb(path)

    assert len(loaded) == 1
    assert loaded.get_files()[0].images_loaded is False


def test_load_safe_file_expands_native_nwb_collection(tmp_path: Path) -> None:
    """One selected NWB file should expand into all of its logical images."""
    original = _make_collection(
        [
            _make_image(axes=("Y", "X"), shape=(8, 12), source_id="first"),
            _make_image(axes=("C", "Y", "X"), shape=(2, 9, 13), source_id="second"),
        ]
    )
    path = tmp_path / "selected-collection.nwb"
    save_nwb_collection(original, path)
    progress: list[tuple[int, int, str]] = []

    result = AcqImageList.load_safe(
        str(path),
        kind=PathKind.FILE,
        load_images=False,
        load_analysis_csv=False,
        progress_callback=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )

    loaded = list(result.acq_image_list)
    assert result.warnings == ()
    assert result.discovered_count == 2
    assert [image.images.header.shape for image in loaded] == [(8, 12), (2, 9, 13)]
    assert all(not image.images_loaded for image in loaded)
    assert result.acq_image_list.file_list == [image.file_id for image in loaded]
    assert [image.file_id for image in loaded] == [
        f"{path.resolve()}#acqimage_0000",
        f"{path.resolve()}#acqimage_0001",
    ]
    assert progress[0][:2] == (0, 1)
    assert progress[-1][:2] == (1, 1)
    assert "loaded 2 image(s)" in progress[-1][2]


def test_load_safe_folder_flattens_multiple_nwb_collections(tmp_path: Path) -> None:
    """Multiple physical NWB files should become one ordered logical list."""
    first_path = tmp_path / "a-first.nwb"
    second_path = tmp_path / "b-second.nwb"
    save_nwb_collection(
        _make_collection(
            [
                _make_image(axes=("Y", "X"), shape=(4, 5), source_id="a0"),
                _make_image(axes=("Y", "X"), shape=(6, 7), source_id="a1"),
            ]
        ),
        first_path,
    )
    save_nwb_collection(
        _make_collection(
            [
                _make_image(axes=("Y", "X"), shape=(8, 9), source_id="b0"),
                _make_image(axes=("C", "Y", "X"), shape=(2, 10, 11), source_id="b1"),
                _make_image(axes=("Y", "X"), shape=(12, 13), source_id="b2"),
            ]
        ),
        second_path,
    )
    progress: list[tuple[int, int, str]] = []

    result = AcqImageList.load_safe(
        str(tmp_path),
        kind=PathKind.FOLDER,
        folder_depth=1,
        load_images=False,
        load_analysis_csv=False,
        progress_callback=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )

    loaded = list(result.acq_image_list)
    assert result.warnings == ()
    assert result.discovered_count == 5
    assert len(loaded) == 5
    assert [image.images.header.shape for image in loaded] == [
        (4, 5),
        (6, 7),
        (8, 9),
        (2, 10, 11),
        (12, 13),
    ]
    assert [image.file_id for image in loaded[:2]] == [
        f"{first_path.resolve()}#acqimage_0000",
        f"{first_path.resolve()}#acqimage_0001",
    ]
    assert [image.file_id for image in loaded[2:]] == [
        f"{second_path.resolve()}#acqimage_0000",
        f"{second_path.resolve()}#acqimage_0001",
        f"{second_path.resolve()}#acqimage_0002",
    ]
    assert result.acq_image_list.file_list == [image.file_id for image in loaded]
    assert all(not image.images_loaded for image in loaded)
    assert progress[-1][:2] == (2, 2)
    assert "loaded 5 image(s)" in progress[-1][2]


def test_load_safe_folder_keeps_valid_nwb_members_when_another_fails(
    tmp_path: Path,
) -> None:
    """A malformed NWB container should warn without discarding valid peers."""
    bad_path = tmp_path / "a-bad.nwb"
    bad_path.write_bytes(b"not an HDF5 file")
    good_path = tmp_path / "b-good.nwb"
    save_nwb_collection(
        _make_collection(
            [
                _make_image(axes=("Y", "X"), shape=(4, 5), source_id="good-0"),
                _make_image(axes=("Y", "X"), shape=(6, 7), source_id="good-1"),
            ]
        ),
        good_path,
    )

    result = AcqImageList.load_safe(
        str(tmp_path),
        kind=PathKind.FOLDER,
        folder_depth=1,
        load_images=False,
        load_analysis_csv=False,
    )

    assert len(result.acq_image_list) == 2
    assert result.discovered_count == 3
    assert len(result.warnings) == 1
    assert result.warnings[0].path == str(bad_path.resolve())
    assert "Failed to load file" in result.warnings[0].message


def test_load_safe_nwb_candidate_respects_injected_file_factory(
    tmp_path: Path,
) -> None:
    """Test factories should retain one-candidate behavior even for .nwb paths."""
    path = tmp_path / "factory.nwb"
    path.write_bytes(b"factory controls loading")
    calls: list[str] = []

    def factory(candidate: str) -> AcqImage:
        calls.append(candidate)
        return AcqImage.from_array(
            np.zeros((3, 4), dtype=np.uint16),
            axes=("Y", "X"),
            source_id=candidate,
        )

    result = AcqImageList.load_safe(
        str(path),
        kind=PathKind.FILE,
        file_factory=factory,
    )

    assert calls == [str(path.resolve())]
    assert len(result.acq_image_list) == 1
    assert result.discovered_count == 1
    assert result.warnings == ()


def test_collection_rejects_unsupported_member_axes(tmp_path: Path) -> None:
    """A ZYX member should fail rather than partially changing its semantics."""
    supported = _make_image(axes=("Y", "X"), shape=(8, 12), source_id="supported")
    unsupported = AcqImage.from_array(
        np.zeros((3, 8, 12), dtype=np.uint16),
        axes=("Z", "Y", "X"),
        source_id="unsupported",
    )
    collection = _make_collection([supported, unsupported])

    with pytest.raises(ValueError, match="YX and CYX"):
        save_nwb_collection(collection, tmp_path / "unsupported.nwb")


def test_collection_saved_file_passes_pynwb_validation(tmp_path: Path) -> None:
    """A collection with independent shapes should pass PyNWB validation."""
    collection = _make_collection(
        [
            _make_image(axes=("C", "Y", "X"), shape=(2, 16, 20), source_id="first"),
            _make_image(axes=("Y", "X"), shape=(40, 7), source_id="second"),
        ]
    )
    path = tmp_path / "validated_collection.nwb"

    save_nwb_collection(collection, path)

    assert validate(path=path) == []
