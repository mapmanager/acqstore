"""Unit tests for AcqImageList list/protocol behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from acqstore.acq_image.acq_image import parent_grandparent_folder_names
from acqstore.acq_image.acq_image_list import AcqImageList, LoadErrorType, SaveEvent, _build_file_list
from acqstore.schema import ACQ_FILE_LIST_SCHEMA
from acqstore.acq_image.supported_import_extensions import get_allowed_import_extensions


class _FakeAcqImage:
    """Test double for AcqImage that avoids file IO."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.file_id = self.path
        self._dirty = self.path.endswith('dirty.tif')
        self._saved = 0
        self._default_roi = 1 if self.path.endswith('roi.tif') else None

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def save(self) -> None:
        self._saved += 1
        self._dirty = False

    def get_default_channel(self) -> int | None:
        return 0

    def get_default_roi(self) -> int | None:
        return self._default_roi

    def get_schema_row(self) -> dict[str, object]:
        parent, grandparent = parent_grandparent_folder_names(
            self.path,
            loaded_from_stream=False,
        )
        return {
            'name': Path(self.path).name,
            'saved': not self._dirty,
            'path': self.path,
            'loaded': '',
            'reference_image': '',
            'file_size': '',
            'parent': parent,
            'grandparent': grandparent,
            'condition': '',
            'genotype': '',
            'num_channels': 1,
            'dims': '',
            'num_rois': 0 if self._default_roi is None else 1,
            'accept': True,
        }

    @property
    def saved_count(self) -> int:
        return self._saved


def test_builds_sorted_files_for_directory(tmp_path: Path) -> None:
    (tmp_path / 'b.tif').write_text('')
    (tmp_path / 'a.tif').write_text('')
    (tmp_path / 'ignore.txt').write_text('')

    file_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)

    assert [Path(item.file_id).name for item in file_list.get_files()] == ['a.tif', 'b.tif']


def test_directory_walk_case_insensitive_includes_czi_excludes_tiff(tmp_path: Path) -> None:
    (tmp_path / 'a.CZI').write_text('')
    (tmp_path / 'b.TIF').write_text('')
    (tmp_path / 'c.oir').write_text('')
    (tmp_path / 'ignored.tiff').write_text('')
    (tmp_path / 'skip.txt').write_text('')

    file_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)

    assert [Path(p).name for p in file_list.file_list] == ['a.CZI', 'b.TIF', 'c.oir']


def test_lookup_and_default_selection(tmp_path: Path) -> None:
    file_path = tmp_path / 'sample_roi.tif'
    file_path.write_text('')

    file_list = AcqImageList(str(file_path), file_factory=_FakeAcqImage)

    default_file_id, channel, roi_id = file_list.get_default_selection()
    assert default_file_id == str(file_path.resolve())
    assert channel == 0
    assert roi_id == 1
    assert file_list.has_file_id(default_file_id)
    assert file_list.get_file_by_id(default_file_id) is not None
    assert file_list.get_file_by_index(0).file_id == default_file_id


def test_iter_save_all_yields_progress_and_honors_cancel(tmp_path: Path) -> None:
    first = tmp_path / 'first_dirty.tif'
    second = tmp_path / 'second_dirty.tif'
    first.write_text('')
    second.write_text('')

    file_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)

    cancelled_once = {'called': False}

    def should_cancel() -> bool:
        if cancelled_once['called']:
            return True
        cancelled_once['called'] = True
        return False

    events = list(file_list.iter_save_all(should_cancel=should_cancel))
    assert [event.event for event in events] == [SaveEvent.SAVING, SaveEvent.SAVED, SaveEvent.CANCELLED]
    assert events[-1].completed == 1


def test_get_schema_rows_match_backend_schema(tmp_path: Path) -> None:
    file_path = tmp_path / 'sample.tif'
    file_path.write_text('')
    file_list = AcqImageList(str(file_path), file_factory=_FakeAcqImage)

    rows = file_list.get_schema_rows()
    assert list(rows[0].keys()) == list(ACQ_FILE_LIST_SCHEMA.field_names())
    assert rows[0]['condition'] == ''


def test_build_file_list_folder_depth_one_skips_subdirs(tmp_path: Path) -> None:
    (tmp_path / 'root.tif').write_text('')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'nested.tif').write_text('')

    exts = get_allowed_import_extensions()
    shallow = _build_file_list(tmp_path, exts, folder_depth=1)
    assert [Path(p).name for p in shallow] == ['root.tif']

    deeper = _build_file_list(tmp_path, exts, folder_depth=2)
    assert {Path(p).name for p in deeper} == {'nested.tif', 'root.tif'}


def test_build_file_list_folder_depth_three_reaches_two_levels_down(tmp_path: Path) -> None:
    (tmp_path / 'a.tif').write_text('')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'b.tif').write_text('')
    deep = sub / 'deep'
    deep.mkdir()
    (deep / 'c.tif').write_text('')

    exts = get_allowed_import_extensions()
    two = _build_file_list(tmp_path, exts, folder_depth=2)
    assert {Path(p).name for p in two} == {'a.tif', 'b.tif'}

    three = _build_file_list(tmp_path, exts, folder_depth=3)
    assert {Path(p).name for p in three} == {'a.tif', 'b.tif', 'c.tif'}


def test_build_file_list_rejects_non_positive_depth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='folder_depth'):
        _build_file_list(tmp_path, ['tif'], folder_depth=0)


def test_acq_image_list_rejects_non_positive_folder_depth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='folder_depth'):
        AcqImageList(str(tmp_path), file_factory=_FakeAcqImage, folder_depth=0)


def test_load_safe_csv_manifest_parent_resolution_and_warnings(tmp_path: Path) -> None:
    csv_dir = tmp_path / 'csvroot'
    csv_dir.mkdir()
    valid = csv_dir / 'ok.tif'
    valid.write_text('')
    csv_path = csv_dir / 'list.csv'
    csv_path.write_text(
        '_rel_path\n'
        'ok.tif\n'
        '   \n'
        'missing.tif\n',
        encoding='utf-8',
    )

    result = AcqImageList.load_safe(
        str(csv_path),
        kind='csv',
        file_factory=_FakeAcqImage,
    )
    assert len(result.acq_image_list) == 1
    assert Path(result.acq_image_list.get_files()[0].file_id).name == 'ok.tif'
    assert len(result.warnings) == 2
    assert any('blank _rel_path' in warning.message for warning in result.warnings)
    assert any('does not exist' in warning.message for warning in result.warnings)


def test_load_safe_csv_missing_required_column_warns(tmp_path: Path) -> None:
    csv_path = tmp_path / 'bad.csv'
    csv_path.write_text('path\nfoo.tif\n', encoding='utf-8')
    result = AcqImageList.load_safe(str(csv_path), kind='csv', file_factory=_FakeAcqImage)
    assert len(result.acq_image_list) == 0
    assert len(result.warnings) == 1
    assert 'missing required column "_rel_path"' in result.warnings[0].message


def test_load_safe_file_missing_returns_warning_not_exception(tmp_path: Path) -> None:
    missing = tmp_path / 'missing.tif'
    result = AcqImageList.load_safe(str(missing), kind='file', file_factory=_FakeAcqImage)
    assert len(result.acq_image_list) == 0
    assert len(result.warnings) == 1
    assert 'does not exist' in result.warnings[0].message


def test_load_safe_csv_duplicate_manifest_path_warns_and_skips_duplicate(tmp_path: Path) -> None:
    csv_dir = tmp_path / 'csvroot'
    csv_dir.mkdir()
    valid = csv_dir / 'ok.tif'
    valid.write_text('')
    csv_path = csv_dir / 'list.csv'
    csv_path.write_text('_rel_path\nok.tif\nok.tif\n', encoding='utf-8')

    result = AcqImageList.load_safe(str(csv_path), kind='csv', file_factory=_FakeAcqImage)

    assert len(result.acq_image_list) == 1
    assert len(result.warnings) == 1
    assert result.warnings[0].error_type is LoadErrorType.CSV_ERROR
    assert 'duplicate _rel_path' in result.warnings[0].message


def test_load_safe_csv_absolute_rel_path_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / 'list.csv'
    absolute = tmp_path / 'ok.tif'
    absolute.write_text('')
    csv_path.write_text(f'_rel_path\n{absolute}\n', encoding='utf-8')

    result = AcqImageList.load_safe(str(csv_path), kind='csv', file_factory=_FakeAcqImage)

    assert len(result.acq_image_list) == 0
    assert len(result.warnings) == 1
    assert '_rel_path must be relative' in result.warnings[0].message


def test_load_safe_csv_parent_escape_rejected(tmp_path: Path) -> None:
    csv_dir = tmp_path / 'csvroot'
    csv_dir.mkdir()
    outside = tmp_path / 'outside.tif'
    outside.write_text('')
    csv_path = csv_dir / 'list.csv'
    csv_path.write_text('_rel_path\n../outside.tif\n', encoding='utf-8')

    result = AcqImageList.load_safe(str(csv_path), kind='csv', file_factory=_FakeAcqImage)

    assert len(result.acq_image_list) == 0
    assert len(result.warnings) == 1
    assert 'escapes manifest root' in result.warnings[0].message


def test_load_safe_csv_missing_file_structured_warning(tmp_path: Path) -> None:
    csv_path = tmp_path / 'list.csv'
    csv_path.write_text('_rel_path\nmissing.tif\n', encoding='utf-8')

    result = AcqImageList.load_safe(str(csv_path), kind='csv', file_factory=_FakeAcqImage)

    assert len(result.acq_image_list) == 0
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.error_type is LoadErrorType.MISSING_FILE
    assert warning.rel_path == 'missing.tif'
    assert warning.resolved_path is not None


def test_load_safe_loader_exception_structured_warning(tmp_path: Path) -> None:
    file_path = tmp_path / 'bad.tif'
    file_path.write_text('')

    def failing_factory(path: str) -> _FakeAcqImage:
        raise RuntimeError(f'boom {Path(path).name}')

    result = AcqImageList.load_safe(str(file_path), kind='file', file_factory=failing_factory)

    assert len(result.acq_image_list) == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].error_type is LoadErrorType.LOADER_ERROR
    assert 'boom bad.tif' in result.warnings[0].message


def test_source_root_path_set_for_folder_file_and_csv(tmp_path: Path) -> None:
    file_path = tmp_path / 'root_file.tif'
    file_path.write_text('')
    folder_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)
    file_list = AcqImageList(str(file_path), file_factory=_FakeAcqImage)
    csv_path = tmp_path / 'list.csv'
    csv_path.write_text('_rel_path\nroot_file.tif\n', encoding='utf-8')
    csv_result = AcqImageList.load_safe(str(csv_path), kind='csv', file_factory=_FakeAcqImage)

    assert folder_list.source_root_path == str(tmp_path.resolve())
    assert file_list.source_root_path == str(tmp_path.resolve())
    assert csv_result.acq_image_list.source_root_path == str(tmp_path.resolve())


def test_to_manifest_csv_writes_manifest_path(tmp_path: Path) -> None:
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'a.tif').write_text('')
    sub = root / 'sub'
    sub.mkdir()
    (sub / 'b.tif').write_text('')
    file_list = AcqImageList(str(root), file_factory=_FakeAcqImage)
    csv_path = tmp_path / 'manifest.csv'

    file_list.to_manifest_csv(csv_path, root_path=root)

    assert csv_path.read_text(encoding='utf-8').splitlines() == [
        '_rel_path',
        'a.tif',
        'sub/b.tif',
    ]


def test_randomized_manifest_master_csv_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / 'data'
    for group in ('g1', 'g2'):
        group_dir = root / group
        group_dir.mkdir(parents=True)
        for name in ('a.tif', 'b.tif', 'c.tif'):
            (group_dir / name).write_text('')
    file_list = AcqImageList(str(root), file_factory=_FakeAcqImage)
    first = tmp_path / 'first.csv'
    second = tmp_path / 'second.csv'

    file_list.to_randomized_manifest_master_csv(first, groupby_column='parent', random_seed=123, root_path=root)
    file_list.to_randomized_manifest_master_csv(second, groupby_column='parent', random_seed=123, root_path=root)

    assert first.read_text(encoding='utf-8') == second.read_text(encoding='utf-8')
    assert first.read_text(encoding='utf-8').splitlines()[0] == '_rel_path,_group,_random_order,_source_index'


def test_randomized_manifest_csv_samples_n_per_group(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    master_csv = tmp_path / 'master.csv'
    csv_path = tmp_path / 'sampled.csv'
    master_csv.write_text(
        '_rel_path,_group,_random_order,_source_index\n'
        'g2/b.tif,g2,0,4\n'
        'g1/c.tif,g1,0,2\n'
        'g2/a.tif,g2,1,3\n'
        'g1/a.tif,g1,1,0\n'
        'g1/b.tif,g1,2,1\n',
        encoding='utf-8',
    )
    file_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)

    file_list.to_randomized_manifest_csv(
        csv_path,
        master_csv_path=master_csv,
        n_per_group=2,
    )

    assert csv_path.read_text(encoding='utf-8').splitlines() == [
        '_rel_path,_group,_random_order,_source_index',
        'g1/c.tif,g1,0,2',
        'g1/a.tif,g1,1,0',
        'g2/b.tif,g2,0,4',
        'g2/a.tif,g2,1,3',
    ]


def test_randomized_manifest_csv_unbalanced_policy(tmp_path: Path) -> None:
    master_csv = tmp_path / 'master.csv'
    csv_path = tmp_path / 'sampled.csv'
    master_csv.write_text(
        '_rel_path,_group,_random_order,_source_index\n'
        'g1/a.tif,g1,0,0\n'
        'g2/a.tif,g2,0,1\n'
        'g2/b.tif,g2,1,2\n',
        encoding='utf-8',
    )
    file_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)

    with pytest.raises(ValueError, match='fewer than n_per_group'):
        file_list.to_randomized_manifest_csv(csv_path, master_csv_path=master_csv, n_per_group=2)

    file_list.to_randomized_manifest_csv(
        csv_path,
        master_csv_path=master_csv,
        n_per_group=2,
        allow_unbalanced=True,
    )
    assert len(csv_path.read_text(encoding='utf-8').splitlines()) == 4


def test_randomized_manifest_rejects_invalid_groupby_column_and_type(tmp_path: Path) -> None:
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'a.tif').write_text('')
    file_list = AcqImageList(str(root), file_factory=_FakeAcqImage)

    with pytest.raises(KeyError):
        file_list.to_randomized_manifest_master_csv(
            tmp_path / 'bad.csv',
            groupby_column='not_a_column',
            root_path=root,
        )
    with pytest.raises(ValueError, match='not categorical-like'):
        file_list.to_randomized_manifest_master_csv(
            tmp_path / 'bad.csv',
            groupby_column='num_channels',
            root_path=root,
        )


def test_randomized_manifest_csv_rejects_non_master_csv(tmp_path: Path) -> None:
    master_csv = tmp_path / 'not_master.csv'
    csv_path = tmp_path / 'sampled.csv'
    master_csv.write_text('_rel_path\nsample.tif\n', encoding='utf-8')
    file_list = AcqImageList(str(tmp_path), file_factory=_FakeAcqImage)

    with pytest.raises(ValueError, match='missing required column'):
        file_list.to_randomized_manifest_csv(
            csv_path,
            master_csv_path=master_csv,
            n_per_group=1,
        )
