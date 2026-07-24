"""Tests for ``ExperimentMetadata`` and ``EXPERIMENT_METADATA_SCHEMA``."""

from __future__ import annotations

import pytest
import numpy as np

from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader, ReferenceImage
from acqstore.acq_image.metadata import (
    EXPERIMENT_METADATA_SCHEMA,
    IMAGE_HEADER_METADATA_SCHEMA,
    REFERENCE_IMAGE_METADATA_SCHEMA,
    ExperimentMetadata,
    ImageHeaderMetadata,
    ReferenceImageMetadata,
)
from acqstore.schema import ACQ_FILE_LIST_SCHEMA, ValueType, validate_values_for_schema


def test_experiment_metadata_age_defaults_to_empty_string() -> None:
    m = ExperimentMetadata()
    assert m.age == ''
    row = m.get_values()
    validate_values_for_schema(EXPERIMENT_METADATA_SCHEMA, row)
    assert row['age'] == ''


def test_experiment_metadata_get_values_matches_schema() -> None:
    m = ExperimentMetadata(species='mouse', depth=1.5, branch_order=2)
    row = m.get_values()
    validate_values_for_schema(EXPERIMENT_METADATA_SCHEMA, row)
    assert row['species'] == 'mouse'
    assert row['depth'] == 1.5
    assert row['branch_order'] == 2
    assert row['note'] == ''


def test_experiment_metadata_get_values_coerces_str_none_to_empty() -> None:
    m = ExperimentMetadata()
    row = m.get_values()
    for fs in EXPERIMENT_METADATA_SCHEMA.fields:
        if fs.value_type is ValueType.STR:
            assert isinstance(row[fs.name], str)


def test_experiment_metadata_update_values_updates_field() -> None:
    m = ExperimentMetadata()
    m.update_values({'species': 'rat', 'depth': 3.0})
    assert m.species == 'rat'
    assert m.depth == 3.0


def test_experiment_metadata_update_values_coerces_none_str_to_empty() -> None:
    m = ExperimentMetadata(species='x')
    m.update_values({'species': None})
    assert m.species == ''


def test_experiment_metadata_update_values_rejects_unknown_field() -> None:
    m = ExperimentMetadata()
    with pytest.raises(KeyError):
        m.update_values({'nope': 'a'})


def test_experiment_metadata_dirty_flag_tracks_changed_values() -> None:
    m = ExperimentMetadata()
    assert m.is_dirty() is False
    m.update_values({'species': 'mouse'})
    assert m.is_dirty() is True
    m.set_clean()
    assert m.is_dirty() is False
    m.update_values({'species': 'mouse'})  # no effective value change
    assert m.is_dirty() is False


def test_experiment_metadata_from_dict_coerces_str_none() -> None:
    m = ExperimentMetadata.from_dict({'species': None, 'depth': 2.0})
    assert m.species == ''
    assert m.depth == 2.0


def test_experiment_metadata_from_dict_coerces_string_numbers() -> None:
    m = ExperimentMetadata.from_dict({'depth': '75', 'branch_order': '2'})
    assert m.depth == 75.0
    assert m.branch_order == 2
    assert isinstance(m.depth, float)
    assert isinstance(m.branch_order, int)


def test_experiment_metadata_from_dict_coerces_empty_string_to_none_for_numeric() -> None:
    m = ExperimentMetadata.from_dict({'depth': '', 'branch_order': ''})
    assert m.depth is None
    assert m.branch_order is None


def test_experiment_metadata_from_dict_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        ExperimentMetadata.from_dict({'depth': 'unknown'})


def test_experiment_metadata_from_dict_rejects_non_integer_branch_order() -> None:
    with pytest.raises(ValueError, match="branch_order"):
        ExperimentMetadata.from_dict({'branch_order': '2.5'})


def test_experiment_metadata_to_dict_excludes_internal_dirty_flag() -> None:
    m = ExperimentMetadata(species='mouse')
    m.update_values({'species': 'rat'})
    payload = m.to_dict()
    assert '_is_dirty' not in payload
    assert payload['species'] == 'rat'


def test_acq_file_list_schema_core_contract_and_unique_names() -> None:
    """Guard stable consumer expectations without freezing the full field list."""
    names_tuple = ACQ_FILE_LIST_SCHEMA.field_names()
    names_set = set(names_tuple)
    assert len(names_set) == len(names_tuple)
    assert {'name', 'path', 'accept', 'saved'}.issubset(names_set)


def test_experiment_metadata_schema_defaults_match_dataclass_defaults() -> None:
    metadata = ExperimentMetadata()
    values = metadata.get_values()
    for fs in EXPERIMENT_METADATA_SCHEMA.fields:
        assert fs.default_value == values[fs.name]


def test_image_header_with_coerced_physical_calibration_normalizes_invalid_units() -> None:
    header = ImageHeader(
        path='/tmp/a.oir',
        shape=(10, 20),
        dims=('Y', 'X'),
        sizes={'Y': 10, 'X': 20},
        dtype=np.dtype('uint16'),
        num_channels=1,
        num_scenes=1,
        physical_units=('bad', -2.0),
        physical_units_labels=('', ''),
    )
    coerced = header.with_coerced_physical_calibration()
    assert coerced.physical_units == (1.0, 1.0)
    assert coerced.physical_units_labels == ('Pixels', 'Pixels')


def test_image_header_metadata_coerces_on_init_and_reads_normalized_units() -> None:
    header = ImageHeader(
        path='/tmp/a.oir',
        shape=(10, 20),
        dims=('Y', 'X'),
        sizes={'Y': 10, 'X': 20},
        dtype=np.dtype('uint16'),
        num_channels=1,
        num_scenes=1,
        physical_units=('x',),
        physical_units_labels=('',),
    )
    section = ImageHeaderMetadata(header, apply_header=lambda _h: None)
    values = section.get_values()
    assert values['physical_unit_y'] == 1.0
    assert values['physical_unit_x'] == 1.0
    assert values['physical_label_y'] == 'Pixels'


def test_image_header_metadata_get_values_and_patch_updates_yx() -> None:
    header = ImageHeader(
        path='/tmp/a.oir',
        shape=(10, 20),
        dims=('Y', 'X'),
        sizes={'Y': 10, 'X': 20},
        dtype=np.dtype('uint16'),
        num_channels=1,
        num_scenes=1,
        physical_units=(1.0, 2.0),
        physical_units_labels=('um', 'um'),
    )
    seen: list[ImageHeader] = []
    section = ImageHeaderMetadata(header, apply_header=lambda h: seen.append(h))
    values = section.get_values()
    validate_values_for_schema(IMAGE_HEADER_METADATA_SCHEMA, values)
    assert values['physical_unit_y'] == 1.0
    assert values['physical_unit_x'] == 2.0
    section.update_values({'physical_unit_y': 3.5, 'physical_label_x': 'px'})
    assert section.is_dirty() is True
    assert seen[-1].physical_units[0] == 3.5
    assert seen[-1].physical_units_labels[1] == 'px'


def test_image_header_metadata_sidecar_patch_filters_editable_keys_only() -> None:
    raw = {
        'shape': '(999, 999)',
        'physical_unit_x': 0.02,
        'physical_label_y': 'seconds',
        'unknown': 'ignored',
    }
    patch = ImageHeaderMetadata.editable_patch_from_sidecar(raw)
    assert patch == {'physical_unit_x': 0.02, 'physical_label_y': 'seconds'}


def test_image_header_metadata_apply_sidecar_calibration_updates_without_dirty() -> None:
    header = ImageHeader(
        path='/tmp/a.oir',
        shape=(10, 20),
        dims=('Y', 'X'),
        sizes={'Y': 10, 'X': 20},
        dtype=np.dtype('uint16'),
        num_channels=1,
        num_scenes=1,
        physical_units=(1.0, 1.0),
        physical_units_labels=('Pixels', 'Pixels'),
    )
    seen: list[ImageHeader] = []
    section = ImageHeaderMetadata(header, apply_header=lambda h: seen.append(h))
    section.apply_sidecar_calibration({'physical_unit_x': 0.02, 'physical_label_y': 'seconds'})
    values = section.get_values()
    assert values['physical_unit_x'] == 0.02
    assert values['physical_label_y'] == 'seconds'
    assert section.is_dirty() is False
    assert seen[-1].physical_units[1] == 0.02
    assert seen[-1].physical_units_labels[0] == 'seconds'


def test_image_header_metadata_schema_hides_num_scenes_in_ui() -> None:
    """``num_scenes`` is stored on the header but not shown in schema-driven forms."""
    fields_by_name = {field.name: field for field in IMAGE_HEADER_METADATA_SCHEMA.fields}
    assert fields_by_name['num_scenes'].visible is False


def test_metadata_section_objects_expose_expected_methods_and_attributes() -> None:
    exp = ExperimentMetadata()
    header = ImageHeader(
        path='/tmp/a.oir',
        shape=(10, 20),
        dims=('Y', 'X'),
        sizes={'Y': 10, 'X': 20},
        dtype=np.dtype('uint16'),
        num_channels=1,
        num_scenes=1,
        physical_units=(1.0, 1.0),
        physical_units_labels=('Pixels', 'Pixels'),
    )
    img = ImageHeaderMetadata(header, apply_header=lambda _h: None)
    for section in (exp, img):
        assert isinstance(section.metadata_section_id, str)
        assert isinstance(section.display_section_title, str)
        assert callable(section.get_schema)
        assert callable(section.get_values)
        assert callable(section.update_values)
        assert callable(section.is_dirty)
        assert callable(section.set_clean)


def test_reference_image_metadata_from_snapshot() -> None:
    """Reference metadata exposes spatial Y/X calibration from ReferenceImage."""
    array = np.zeros((512, 512), dtype=np.uint16)
    ref = ReferenceImage(
        array=array,
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=None,
        coord_units=(('Y', 'micrometer'), ('X', 'micrometer')),
        coord_scales=(('Y', 0.331), ('X', 0.331)),
        coords=(),
    )
    section = ReferenceImageMetadata.from_reference_image(ref)
    values = section.get_values()
    validate_values_for_schema(REFERENCE_IMAGE_METADATA_SCHEMA, values)
    assert values['shape'] == '(512, 512)'
    assert values['dims'] == "('Y', 'X')"
    assert values['dtype'] == 'uint16'
    assert values['num_channels'] == 1
    assert values['physical_unit_x'] == pytest.approx(0.331)
    assert values['physical_unit_y'] == pytest.approx(0.331)
    assert values['physical_label_x'] == 'um'
    assert values['physical_label_y'] == 'um'
    assert values['has_scan_path'] is False
    assert values['scan_path_num_points'] == 0
    assert values['line_roi'] == ''
    assert values['scan_path_x_pixels'] == []
    assert values['scan_path_y_pixels'] == []


def test_reference_image_metadata_includes_scan_path_lists() -> None:
    """Reference metadata stores scan-path coordinates as JSON-serializable lists."""
    scan_path = np.asarray([[1.0, 7.0], [2.0, 6.0]])
    ref = ReferenceImage(
        array=np.zeros((8, 8), dtype=np.uint8),
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=(1.0, 2.0, 7.0, 6.0),
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 1.0), ('X', 1.0)),
        coords=(),
        scan_path=scan_path,
    )
    values = ReferenceImageMetadata.from_reference_image(ref).get_values()
    validate_values_for_schema(REFERENCE_IMAGE_METADATA_SCHEMA, values)
    assert values['has_scan_path'] is True
    assert values['scan_path_num_points'] == 2
    assert values['line_roi'] == '(1.0, 2.0, 7.0, 6.0)'
    assert values['scan_path_x_pixels'] == [1.0, 7.0]
    assert values['scan_path_y_pixels'] == [2.0, 6.0]


def test_reference_image_metadata_rejects_edits() -> None:
    """Reference metadata is read-only in v1."""
    array = np.zeros((4, 4), dtype=np.uint8)
    ref = ReferenceImage(
        array=array,
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=None,
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 1.0), ('X', 1.0)),
        coords=(),
    )
    section = ReferenceImageMetadata.from_reference_image(ref)
    with pytest.raises(ValueError, match='not editable'):
        section.update_values({'physical_unit_x': 0.5})
