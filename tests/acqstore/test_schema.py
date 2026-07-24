"""Unit tests for acqstore semantic schema definitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from acqstore.schema import (
    ACQ_FILE_LIST_SCHEMA,
    FieldSchema,
    SchemaDefinition,
    ValueType,
    card_groups,
    validate_patch_for_schema,
    validate_values_for_schema,
)


def _minimal_valid_row_for_schema(
    schema: SchemaDefinition,
    **overrides: object,
) -> dict[str, object]:
    """Build a schema-complete values dict for tests; survives schema field churn."""
    values: dict[str, object] = {}
    for field in schema.fields:
        if field.default_value is not None:
            values[field.name] = field.default_value
            continue
        vt = field.value_type
        if vt is ValueType.STR:
            values[field.name] = 'n'
        elif vt is ValueType.PATH:
            values[field.name] = '/tmp/row.tif'
        elif vt is ValueType.INT:
            values[field.name] = 0
        elif vt is ValueType.FLOAT:
            values[field.name] = 0.0
        elif vt is ValueType.BOOL:
            values[field.name] = True
        elif vt is ValueType.ENUM:
            assert field.choices is not None
            values[field.name] = field.choices[0]
        else:
            raise AssertionError(f'Unhandled value_type in test: {field.name} {vt!r}')
    values.update(overrides)
    return values


def test_enum_field_requires_choices() -> None:
    with pytest.raises(ValueError, match='ENUM'):
        FieldSchema(
            name='status',
            display_name='Status',
            value_type=ValueType.ENUM,
        )


def test_field_schema_to_from_dict_round_trip() -> None:
    field = FieldSchema(
        name='path',
        display_name='Path',
        value_type=ValueType.PATH,
        group='File',
    )
    restored = FieldSchema.from_dict(field.to_dict())
    assert restored == field


def test_field_schema_from_dict_maps_legacy_default_key() -> None:
    restored = FieldSchema.from_dict(
        {
            'name': 'species',
            'display_name': 'Species',
            'value_type': 'str',
            'default': '',
        }
    )
    assert restored.default_value == ''


def test_field_schema_rejects_invalid_default_value_type() -> None:
    with pytest.raises(TypeError, match='default_value'):
        FieldSchema(
            name='depth',
            display_name='Depth',
            value_type=ValueType.FLOAT,
            default_value='not-a-number',
        )


def test_field_schema_enum_default_must_be_choice() -> None:
    with pytest.raises(ValueError, match='choices'):
        FieldSchema(
            name='kind',
            display_name='Kind',
            value_type=ValueType.ENUM,
            choices=('a', 'b'),
            default_value='c',
        )


def test_schema_definition_rejects_duplicate_fields() -> None:
    with pytest.raises(ValueError, match='Duplicate schema field names'):
        SchemaDefinition(
            schema_id='x',
            version=1,
            fields=(
                FieldSchema(name='a', display_name='A'),
                FieldSchema(name='a', display_name='A2'),
            ),
        )


def test_schema_definition_helpers() -> None:
    schema = ACQ_FILE_LIST_SCHEMA
    assert schema.schema_id == 'acq_file_list'
    assert schema.version == 1
    path_field = next(f for f in schema.fields if f.name == 'path')
    assert path_field.visible is False
    visible = schema.visible_fields()
    assert tuple(f.name for f in visible) == tuple(
        f.name for f in schema.fields if f.visible
    )
    assert card_groups(schema) == tuple(dict.fromkeys(f.group for f in visible))


def test_field_schema_from_dict_ignores_unknown_keys() -> None:
    restored = FieldSchema.from_dict(
        {
            'name': 'a',
            'display_name': 'A',
            'value_type': 'int',
            'visible': True,
            'editable': False,
            'new_future_key': 'ignored',
        }
    )
    assert restored.value_type is ValueType.INT


def test_validate_values_for_schema_rejects_missing_visible_field() -> None:
    with pytest.raises(KeyError, match='name'):
        validate_values_for_schema(ACQ_FILE_LIST_SCHEMA, {'path': '/tmp/a.tif'})


def test_validate_values_for_schema_rejects_extra_keys_by_default() -> None:
    row = _minimal_valid_row_for_schema(ACQ_FILE_LIST_SCHEMA)
    row['extra'] = True
    with pytest.raises(ValueError, match='extra'):
        validate_values_for_schema(ACQ_FILE_LIST_SCHEMA, row)


def test_validate_values_for_schema_allows_extra_values_when_enabled() -> None:
    row = _minimal_valid_row_for_schema(ACQ_FILE_LIST_SCHEMA)
    row['extra'] = True
    validate_values_for_schema(
        ACQ_FILE_LIST_SCHEMA,
        row,
        allow_extra_values=True,
    )


def test_acq_file_list_schema_has_validating_minimal_row() -> None:
    """Every declared field must be satisfiable; catches schema vs row drift."""
    values = _minimal_valid_row_for_schema(ACQ_FILE_LIST_SCHEMA)
    validate_values_for_schema(ACQ_FILE_LIST_SCHEMA, values)


def test_validate_patch_for_schema_rejects_unknown_or_noneditable_fields() -> None:
    with pytest.raises(KeyError, match='unknown'):
        validate_patch_for_schema(ACQ_FILE_LIST_SCHEMA, {'unknown': 1})
    with pytest.raises(ValueError, match='name'):
        validate_patch_for_schema(ACQ_FILE_LIST_SCHEMA, {'name': 'new'})


def test_schema_definition_from_dict_ignores_unknown_top_level_keys() -> None:
    restored = SchemaDefinition.from_dict(
        {
            'schema_id': 'example',
            'version': 1,
            'unknown': 'ignored',
            'fields': [
                {
                    'name': 'a',
                    'display_name': 'A',
                    'value_type': 'int',
                    'new_future_key': 'ignored',
                }
            ],
        }
    )
    assert restored.schema_id == 'example'
    assert restored.fields[0].value_type is ValueType.INT


def test_acqstore_does_not_import_cloudscope() -> None:
    src_root = Path(__file__).resolve().parents[2] / 'src' / 'acqstore'
    python_files = src_root.rglob('*.py')
    for py_file in python_files:
        text = py_file.read_text()
        assert 'from cloudscope' not in text
        assert 'import cloudscope' not in text
