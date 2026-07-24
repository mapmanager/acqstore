"""Unit tests for ROI v1 models."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.acq_image.roi import (
    BaseROI,
    ImageBounds,
    LineEndpoints,
    LineROI,
    RectROI,
    RectRoiBounds,
    RoiSet,
    RoiType,
    clamp_line_endpoints_to_image,
    clamp_rect_bounds_to_image,
    roi_from_dict,
)


def test_image_bounds_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ImageBounds(width=0, height=10)

    with pytest.raises(ValueError):
        ImageBounds(width=10, height=0)

    with pytest.raises(ValueError):
        ImageBounds(width=10, height=10, num_slices=0)


def test_rect_bounds_from_image_bounds() -> None:
    bounds = RectRoiBounds.from_image_bounds(ImageBounds(width=20, height=10))
    assert bounds == RectRoiBounds(dim0_start=0, dim0_stop=10, dim1_start=0, dim1_stop=20)


def test_rect_bounds_clamped_to_image_bounds() -> None:
    bounds = RectRoiBounds(dim0_start=-5, dim0_stop=100, dim1_start=-2, dim1_stop=50)
    clamped = bounds.clamped_to(ImageBounds(width=20, height=10))
    assert clamped == RectRoiBounds(dim0_start=0, dim0_stop=10, dim1_start=0, dim1_stop=20)


def test_rect_bounds_clamped_to_image_bounds_fixes_inverted_edges() -> None:
    bounds = RectRoiBounds(dim0_start=8, dim0_stop=2, dim1_start=9, dim1_stop=1)
    clamped = bounds.clamped_to(ImageBounds(width=20, height=10))
    assert clamped == RectRoiBounds(dim0_start=2, dim0_stop=8, dim1_start=1, dim1_stop=9)


def test_line_endpoints_clamped_to_image_bounds() -> None:
    endpoints = LineEndpoints(row0=-5, col0=100, row1=20, col1=-2)
    clamped = endpoints.clamped_to(ImageBounds(width=20, height=10))
    assert clamped == LineEndpoints(row0=0, col0=19, row1=9, col1=0)


def test_base_roi_to_dict_raises() -> None:
    roi = BaseROI(roi_id=1)
    with pytest.raises(NotImplementedError):
        roi.to_dict()


def test_rect_roi_to_dict_uses_enum_value() -> None:
    roi = RectROI(
        roi_id=1,
        name="rect",
        note="note",
        bounds=RectRoiBounds(1, 5, 2, 7),
    )
    assert roi.to_dict() == {
        "roi_id": 1,
        "roi_type": RoiType.RECTROI.value,
        "version": "1.0",
        "name": "rect",
        "note": "note",
        "data": {
            "row_start": 1,
            "row_stop": 5,
            "col_start": 2,
            "col_stop": 7,
        },
    }


def test_line_roi_to_dict_uses_enum_value() -> None:
    roi = LineROI(
        roi_id=2,
        name="line",
        note="note",
        endpoints=LineEndpoints(row0=1, col0=2, row1=3, col1=4),
    )
    assert roi.to_dict() == {
        "roi_id": 2,
        "roi_type": RoiType.LINESEGMENTROI.value,
        "version": "1.0",
        "name": "line",
        "note": "note",
        "data": {
            "row0": 1,
            "col0": 2,
            "row1": 3,
            "col1": 4,
        },
    }


def test_rect_roi_round_trip() -> None:
    roi = RectROI(roi_id=1, name="r", note="n", bounds=RectRoiBounds(1, 2, 3, 4))
    restored = RectROI.from_dict(roi.to_dict())
    assert restored == roi


def test_line_roi_round_trip() -> None:
    roi = LineROI(roi_id=1, name="l", note="n", endpoints=LineEndpoints(1, 2, 3, 4))
    restored = LineROI.from_dict(roi.to_dict())
    assert restored == roi


def test_roi_from_dict_dispatches_rect() -> None:
    roi = RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4))
    restored = roi_from_dict(roi.to_dict())
    assert isinstance(restored, RectROI)
    assert restored == roi


def test_roi_from_dict_dispatches_line() -> None:
    roi = LineROI(roi_id=1, endpoints=LineEndpoints(1, 2, 3, 4))
    restored = roi_from_dict(roi.to_dict())
    assert isinstance(restored, LineROI)
    assert restored == roi


def test_roi_from_dict_requires_roi_type() -> None:
    with pytest.raises(ValueError, match="roi_type"):
        roi_from_dict({})


def test_rect_from_dict_rejects_unexpected_key() -> None:
    data = RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4)).to_dict()
    data["extra"] = True
    with pytest.raises(ValueError, match="unexpected"):
        RectROI.from_dict(data)


def test_rect_from_dict_rejects_missing_key() -> None:
    data = RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4)).to_dict()
    del data["name"]
    with pytest.raises(ValueError, match="missing"):
        RectROI.from_dict(data)


def test_rect_from_dict_rejects_wrong_roi_type() -> None:
    data = RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4)).to_dict()
    data["roi_type"] = RoiType.LINESEGMENTROI.value
    with pytest.raises(NotImplementedError):
        RectROI.from_dict(data)


def test_rect_from_dict_rejects_bad_version() -> None:
    data = RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4)).to_dict()
    data["version"] = "2.0"
    with pytest.raises(ValueError, match="Unsupported"):
        RectROI.from_dict(data)


def test_line_from_dict_rejects_unexpected_key() -> None:
    data = LineROI(roi_id=1, endpoints=LineEndpoints(1, 2, 3, 4)).to_dict()
    data["extra"] = True
    with pytest.raises(ValueError, match="unexpected"):
        LineROI.from_dict(data)


def test_line_from_dict_rejects_missing_key() -> None:
    data = LineROI(roi_id=1, endpoints=LineEndpoints(1, 2, 3, 4)).to_dict()
    del data["name"]
    with pytest.raises(ValueError, match="missing"):
        LineROI.from_dict(data)


def test_line_from_dict_rejects_wrong_roi_type() -> None:
    data = LineROI(roi_id=1, endpoints=LineEndpoints(1, 2, 3, 4)).to_dict()
    data["roi_type"] = RoiType.RECTROI.value
    with pytest.raises(NotImplementedError):
        LineROI.from_dict(data)


def test_line_from_dict_rejects_bad_version() -> None:
    data = LineROI(roi_id=1, endpoints=LineEndpoints(1, 2, 3, 4)).to_dict()
    data["version"] = "2.0"
    with pytest.raises(ValueError, match="Unsupported"):
        LineROI.from_dict(data)


def test_create_rect_roi_defaults_to_full_image_and_sets_dirty() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    roi = rois.create_rect_roi()
    assert roi.bounds == RectRoiBounds(0, 10, 0, 20)
    assert rois.is_dirty()


def test_create_rect_roi_clamps_bounds() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    roi = rois.create_rect_roi(RectRoiBounds(-1, 99, -2, 88))
    assert roi.bounds == RectRoiBounds(0, 10, 0, 20)


def test_create_line_roi_clamps_endpoints() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    roi = rois.create_line_roi(LineEndpoints(-1, 99, 99, -2))
    assert roi.endpoints == LineEndpoints(0, 19, 9, 0)


def test_roi_ids_are_assigned_sequentially() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    r1 = rois.create_rect_roi()
    r2 = rois.create_line_roi(LineEndpoints(1, 2, 3, 4))
    assert (r1.roi_id, r2.roi_id) == (1, 2)
    assert rois.get_roi_ids() == [1, 2]


def test_edit_rect_roi_updates_fields_and_sets_dirty() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    roi = rois.create_rect_roi()
    rois.set_clean()

    edited = rois.edit_rect_roi(
        roi.roi_id,
        bounds=RectRoiBounds(1, 2, 3, 4),
        name="new",
        note="note",
    )

    assert edited.bounds == RectRoiBounds(1, 2, 3, 4)
    assert edited.name == "new"
    assert edited.note == "note"
    assert rois.is_dirty()


def test_edit_line_roi_updates_fields_and_sets_dirty() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    roi = rois.create_line_roi(LineEndpoints(1, 2, 3, 4))
    rois.set_clean()

    edited = rois.edit_line_roi(
        roi.roi_id,
        endpoints=LineEndpoints(5, 6, 7, 8),
        name="new",
        note="note",
    )

    assert edited.endpoints == LineEndpoints(5, 6, 7, 8)
    assert edited.name == "new"
    assert edited.note == "note"
    assert rois.is_dirty()


def test_edit_wrong_roi_type_raises() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    rect = rois.create_rect_roi()
    line = rois.create_line_roi(LineEndpoints(1, 2, 3, 4))

    with pytest.raises(TypeError):
        rois.edit_line_roi(rect.roi_id, endpoints=LineEndpoints(1, 2, 3, 4))

    with pytest.raises(TypeError):
        rois.edit_rect_roi(line.roi_id, bounds=RectRoiBounds(1, 2, 3, 4))


def test_edit_missing_roi_raises() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    with pytest.raises(ValueError):
        rois.edit_rect_roi(999, bounds=RectRoiBounds(1, 2, 3, 4))

    with pytest.raises(ValueError):
        rois.edit_line_roi(999, endpoints=LineEndpoints(1, 2, 3, 4))


def test_delete_sets_dirty_and_removes_roi() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    roi = rois.create_rect_roi()
    rois.set_clean()

    rois.delete(roi.roi_id)

    assert rois.get(roi.roi_id) is None
    assert rois.num_rois == 0
    assert rois.is_dirty()


def test_delete_missing_roi_raises() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    with pytest.raises(ValueError):
        rois.delete(999)


def test_clear_resets_ids() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    rois.create_rect_roi()
    rois.create_line_roi(LineEndpoints(1, 2, 3, 4))

    assert rois.clear() == 2
    assert rois.num_rois == 0

    roi = rois.create_rect_roi()
    assert roi.roi_id == 1


def test_clear_empty_does_not_set_dirty() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    rois.set_clean()

    assert rois.clear() == 0
    assert not rois.is_dirty()


def test_iter_and_as_list_preserve_creation_order() -> None:
    rois = RoiSet(ImageBounds(width=20, height=10))
    r1 = rois.create_rect_roi()
    r2 = rois.create_line_roi(LineEndpoints(1, 2, 3, 4))

    assert list(rois) == [r1, r2]
    assert rois.as_list() == [r1, r2]


def test_roiset_round_trip_mixed_rois() -> None:
    bounds = ImageBounds(width=20, height=10)
    rois = RoiSet(bounds)
    rect = rois.create_rect_roi(RectRoiBounds(1, 5, 2, 7), name="rect")
    line = rois.create_line_roi(LineEndpoints(1, 2, 3, 4), name="line")

    payload = rois.to_list()

    restored = RoiSet(bounds)
    restored.from_list(payload)

    assert not restored.is_dirty()
    assert restored.get_roi_ids() == [rect.roi_id, line.roi_id]
    assert restored.to_list() == payload


def test_from_list_replaces_existing_rois() -> None:
    bounds = ImageBounds(width=20, height=10)
    rois = RoiSet(bounds)
    rois.create_rect_roi()

    payload = [
        LineROI(roi_id=10, endpoints=LineEndpoints(1, 2, 3, 4)).to_dict(),
    ]

    rois.from_list(payload)

    assert rois.get_roi_ids() == [10]
    new_roi = rois.create_rect_roi()
    assert new_roi.roi_id == 11


def test_from_list_rejects_duplicate_ids() -> None:
    bounds = ImageBounds(width=20, height=10)
    payload = [
        RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4)).to_dict(),
        LineROI(roi_id=1, endpoints=LineEndpoints(1, 2, 3, 4)).to_dict(),
    ]

    rois = RoiSet(bounds)

    with pytest.raises(ValueError, match="Duplicate"):
        rois.from_list(payload)


def test_from_list_clamps_loaded_rois() -> None:
    bounds = ImageBounds(width=20, height=10)
    payload = [
        RectROI(roi_id=1, bounds=RectRoiBounds(-1, 99, -2, 88)).to_dict(),
        LineROI(roi_id=2, endpoints=LineEndpoints(-1, 99, 99, -2)).to_dict(),
    ]

    rois = RoiSet(bounds)
    rois.from_list(payload)

    rect = rois.get(1)
    line = rois.get(2)

    assert isinstance(rect, RectROI)
    assert isinstance(line, LineROI)
    assert rect.bounds == RectRoiBounds(0, 10, 0, 20)
    assert line.endpoints == LineEndpoints(0, 19, 9, 0)


def test_clamp_rect_bounds_to_image_rejects_non_2d_array() -> None:
    img = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        clamp_rect_bounds_to_image(RectRoiBounds(1, 2, 3, 4), img)


def test_clamp_line_endpoints_to_image_rejects_non_2d_array() -> None:
    img = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        clamp_line_endpoints_to_image(LineEndpoints(1, 2, 3, 4), img)


def test_unknown_roi_type_rejected() -> None:
    data = RectROI(roi_id=1, bounds=RectRoiBounds(1, 2, 3, 4)).to_dict()
    data["roi_type"] = "polygonroi"

    with pytest.raises(ValueError, match="Unsupported ROI type"):
        roi_from_dict(data)


def test_roi_set_has_roi() -> None:
    """RoiSet.has_roi should report ROI presence."""
    roi_set = RoiSet(ImageBounds(width=10, height=20, num_slices=1))
    roi = roi_set.create_rect_roi()

    assert roi_set.has_roi(roi.roi_id)
    assert not roi_set.has_roi(roi.roi_id + 100)
