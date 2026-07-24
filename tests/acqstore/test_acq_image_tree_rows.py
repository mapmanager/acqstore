"""Tests for AcqStore acquisition tree-row APIs."""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import RadonVelocityAnalysis
from acqstore.acq_image.tree_rows import (
    ACQ_TREE_ANALYSIS_CHANNEL_FIELD,
    ACQ_TREE_ANALYSIS_NAME_FIELD,
    ACQ_TREE_ANALYSIS_ROI_ID_FIELD,
    ACQ_TREE_PATH_FIELD,
    ACQ_TREE_ROW_ID_FIELD,
    ACQ_TREE_ROW_TYPE_ANALYSIS,
    ACQ_TREE_ROW_TYPE_FIELD,
    ACQ_TREE_ROW_TYPE_FILE,
    build_analysis_tree_row_id,
)
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader
from acqstore.schema import ACQ_FILE_LIST_SCHEMA
import numpy as np


class _FakeImages:
    """Small image-loader test double for schema-row generation."""

    reference_image = None

    def __init__(self, *, num_channels: int = 2, path: str = '/tmp/sample.tif') -> None:
        self.num_channels = num_channels
        self.default_channel = 0
        self.header = ImageHeader(
            path=path,
            shape=(100, 200, num_channels),
            dims=('C', 'Y', 'X'),
            sizes={'C': num_channels, 'Y': 100, 'X': 200},
            dtype=np.dtype('uint16'),
            num_channels=num_channels,
            num_scenes=1,
            physical_units=(1.0, 1.0, 1.0),
            physical_units_labels=('Pixels', 'Pixels', 'Pixels'),
        )

    @property
    def has_reference_image(self) -> bool:
        """Match :class:`BaseFileLoader.has_reference_image` contract."""
        return self.reference_image is not None


class _FakeRois:
    """Small ROI-set test double for schema-row generation."""

    def __init__(self, *, num_rois: int = 3) -> None:
        self.num_rois = num_rois

    def is_dirty(self) -> bool:
        """Return clean state for tests."""
        return False

    def get_roi_ids(self) -> list[int]:
        """Return deterministic ROI identifiers."""
        return list(range(1, self.num_rois + 1))


class _FakeMetadata:
    """Small metadata-section test double for schema-row generation."""

    def __init__(self, *, genotype: str = 'wt', condition: str = '') -> None:
        self.genotype = genotype
        self.condition = condition

    def is_dirty(self) -> bool:
        """Return clean state for tests."""
        return False


class _FakeTreeAcqImage:
    """Test double for AcqImageList tree-row flattening."""

    def __init__(self, path: str) -> None:
        self.file_id = str(path)

    def get_tree_rows(self) -> list[dict[str, object]]:
        """Return deterministic rows for list-order assertions."""
        return [
            {
                ACQ_TREE_ROW_ID_FIELD: self.file_id,
                ACQ_TREE_PATH_FIELD: [self.file_id],
                ACQ_TREE_ROW_TYPE_FIELD: ACQ_TREE_ROW_TYPE_FILE,
            },
            {
                ACQ_TREE_ROW_ID_FIELD: f'{self.file_id}::child',
                ACQ_TREE_PATH_FIELD: [self.file_id, f'{self.file_id}::child'],
                ACQ_TREE_ROW_TYPE_FIELD: ACQ_TREE_ROW_TYPE_ANALYSIS,
            },
        ]


def _make_acq_image(path: Path, *, with_analysis: bool = True) -> AcqImage:
    """Build an AcqImage without invoking file loaders.

    Args:
        path: Source path to assign to the image.
        with_analysis: Whether to attach analysis rows.

    Returns:
        AcqImage instance with enough runtime state for tree-row tests.
    """
    acq_image = AcqImage.__new__(AcqImage)
    acq_image.path = str(path.resolve())
    acq_image._accept = True
    acq_image._pixels = None
    acq_image._images = _FakeImages(num_channels=2, path=str(path.resolve()))
    acq_image._experimental_metadata = _FakeMetadata(genotype='wt', condition='control')
    acq_image._image_header_metadata = _FakeMetadata(genotype='')
    acq_image._reference_image_metadata = None
    acq_image._image_contrast_dirty = False
    acq_image._rois = _FakeRois(num_rois=3)
    acq_image._acq_analysis_set = AcqAnalysisSet(acq_image.path)
    if with_analysis:
        acq_image._acq_analysis_set.add(RadonVelocityAnalysis(channel=0, roi_id=1))
        acq_image._acq_analysis_set.add(RadonVelocityAnalysis(channel=1, roi_id=2))
        acq_image._acq_analysis_set.set_clean()
    return acq_image


def test_acq_image_tree_rows_includes_file_row_and_analysis_rows(tmp_path: Path) -> None:
    """AcqImage returns parent row first, followed by analysis child rows."""
    acq_image = _make_acq_image(tmp_path / 'sample.tif')

    rows = acq_image.get_tree_rows()

    assert [row[ACQ_TREE_ROW_TYPE_FIELD] for row in rows] == [
        ACQ_TREE_ROW_TYPE_FILE,
        ACQ_TREE_ROW_TYPE_ANALYSIS,
        ACQ_TREE_ROW_TYPE_ANALYSIS,
    ]
    assert rows[0][ACQ_TREE_PATH_FIELD] == [acq_image.file_id]
    assert rows[1][ACQ_TREE_PATH_FIELD] == [acq_image.file_id, rows[1][ACQ_TREE_ROW_ID_FIELD]]


def test_acq_image_tree_rows_file_row_carries_schema_keys_with_values(tmp_path: Path) -> None:
    """The parent file tree row includes the unchanged file-list schema values."""
    acq_image = _make_acq_image(tmp_path / 'sample.tif')

    schema_row = acq_image.get_schema_row()
    file_row = acq_image.get_tree_rows()[0]

    for key, value in schema_row.items():
        assert file_row[key] == value
    assert file_row['condition'] == 'control'
    assert file_row['dims'] == 'C:2 Y:100 X:200'
    assert file_row[ACQ_TREE_ROW_ID_FIELD] == acq_image.file_id
    assert file_row[ACQ_TREE_ROW_TYPE_FIELD] == ACQ_TREE_ROW_TYPE_FILE
    assert file_row[ACQ_TREE_ANALYSIS_NAME_FIELD] is None
    assert file_row[ACQ_TREE_ANALYSIS_CHANNEL_FIELD] is None
    assert file_row[ACQ_TREE_ANALYSIS_ROI_ID_FIELD] is None


def test_acq_image_tree_rows_analysis_row_keys_match_contract(tmp_path: Path) -> None:
    """Analysis rows carry tree contract fields and schema keys.

    The ``name``, ``num_channels`` and ``num_rois`` schema fields are
    overloaded on analysis rows to carry display values
    (``analysis_name``, ``channel``, ``roi_id``) rather than counts; the
    remaining schema keys are set to ``None`` because they are not
    meaningful for analysis children.
    """
    acq_image = _make_acq_image(tmp_path / 'sample.tif')

    analysis_row = acq_image.get_tree_rows()[1]

    assert analysis_row[ACQ_TREE_ROW_TYPE_FIELD] == ACQ_TREE_ROW_TYPE_ANALYSIS
    assert analysis_row[ACQ_TREE_ANALYSIS_NAME_FIELD] == 'radon_velocity'
    assert analysis_row[ACQ_TREE_ANALYSIS_CHANNEL_FIELD] == 0
    assert analysis_row[ACQ_TREE_ANALYSIS_ROI_ID_FIELD] == 1
    assert analysis_row['name'] == 'radon_velocity'
    assert analysis_row['num_channels'] == 0
    assert analysis_row['num_rois'] == 1
    display_overloaded = {'name', 'num_channels', 'num_rois'}
    for key in ACQ_FILE_LIST_SCHEMA.field_names():
        assert key in analysis_row
        if key in display_overloaded:
            continue
        assert analysis_row[key] is None


def test_acq_image_tree_rows_analysis_row_id_is_stable_and_unique(tmp_path: Path) -> None:
    """Analysis tree ids encode file id plus analysis identity."""
    acq_image = _make_acq_image(tmp_path / 'sample.tif')

    rows = acq_image.get_tree_rows()
    ids = [row[ACQ_TREE_ROW_ID_FIELD] for row in rows]

    assert len(ids) == len(set(ids))
    assert rows[1][ACQ_TREE_ROW_ID_FIELD] == build_analysis_tree_row_id(
        acq_image.file_id,
        'radon_velocity',
        0,
        1,
    )


def test_acq_image_list_tree_rows_preserves_display_order(tmp_path: Path) -> None:
    """AcqImageList flattens each file subtree in file display order."""
    (tmp_path / 'b.tif').write_text('')
    (tmp_path / 'a.tif').write_text('')

    acq_list = AcqImageList(str(tmp_path), file_factory=_FakeTreeAcqImage)
    rows = acq_list.get_tree_rows()

    assert [Path(str(row[ACQ_TREE_PATH_FIELD][0])).name for row in rows] == [
        'a.tif',
        'a.tif',
        'b.tif',
        'b.tif',
    ]


def test_acq_image_list_tree_rows_empty_when_no_files() -> None:
    """An empty AcqImageList object returns no tree rows."""
    acq_list = AcqImageList.__new__(AcqImageList)
    acq_list.path = ''
    acq_list.file_list = []
    acq_list._files = []
    acq_list._files_by_id = {}

    assert acq_list.get_tree_rows() == []


def test_existing_get_schema_row_unchanged_after_tree_row_addition(tmp_path: Path) -> None:
    """Tree-row generation does not mutate or extend the existing schema row."""
    acq_image = _make_acq_image(tmp_path / 'sample.tif')

    before = acq_image.get_schema_row()
    acq_image.get_tree_rows()
    after = acq_image.get_schema_row()

    assert after == before
    assert list(after.keys()) == list(ACQ_FILE_LIST_SCHEMA.field_names())
