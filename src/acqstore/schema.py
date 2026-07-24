"""Semantic schema definitions for AcqStore values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any


class ValueType(StrEnum):
    """Supported semantic value types."""

    STR = 'str'
    INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    PATH = 'path'
    ENUM = 'enum'


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """Semantic schema for one backend value."""

    name: str
    display_name: str
    value_type: ValueType = ValueType.STR
    description: str = ''
    unit: str | None = None
    required: bool = False
    default_value: Any | None = None
    choices: tuple[Any, ...] | None = None
    visible: bool = True
    editable: bool = False
    group: str | None = None

    def __post_init__(self) -> None:
        """Validate field consistency."""
        if not self.name:
            raise ValueError('FieldSchema.name must not be empty')
        if not self.display_name:
            raise ValueError('FieldSchema.display_name must not be empty')
        if self.value_type is ValueType.ENUM and not self.choices:
            raise ValueError(
                f'FieldSchema {self.name!r} has value_type=ENUM but no choices'
            )
        if self.default_value is not None:
            self._validate_default_value()

    @property
    def default(self) -> Any | None:
        """Backward-compatible alias for older callsites."""
        return self.default_value

    def _validate_default_value(self) -> None:
        """Validate that default value is compatible with ``value_type``."""
        value = self.default_value
        if self.value_type in {ValueType.STR, ValueType.PATH}:
            if not isinstance(value, str):
                raise TypeError(f'FieldSchema {self.name!r} default_value must be str')
            return
        if self.value_type is ValueType.INT:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'FieldSchema {self.name!r} default_value must be int')
            return
        if self.value_type is ValueType.FLOAT:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f'FieldSchema {self.name!r} default_value must be float-compatible')
            return
        if self.value_type is ValueType.BOOL:
            if not isinstance(value, bool):
                raise TypeError(f'FieldSchema {self.name!r} default_value must be bool')
            return
        if self.value_type is ValueType.ENUM:
            assert self.choices is not None
            if value not in self.choices:
                raise ValueError(
                    f'FieldSchema {self.name!r} default_value must be one of choices: {self.choices!r}'
                )
            return

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'value_type': self.value_type.value,
            'description': self.description,
            'unit': self.unit,
            'required': self.required,
            'default_value': self.default_value,
            'choices': list(self.choices) if self.choices is not None else None,
            'visible': self.visible,
            'editable': self.editable,
            'group': self.group,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSchema:
        """Create a field schema from a dictionary.

        Unknown keys are ignored for forward compatibility.
        """
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        if 'value_type' in filtered:
            filtered['value_type'] = ValueType(str(filtered['value_type']))
        if 'default_value' not in filtered and 'default' in data:
            filtered['default_value'] = data['default']
        if 'choices' in filtered and filtered['choices'] is not None:
            filtered['choices'] = tuple(filtered['choices'])
        return cls(**filtered)


@dataclass(frozen=True, slots=True)
class SchemaDefinition:
    """Versioned collection of field schemas."""

    schema_id: str
    version: int
    fields: tuple[FieldSchema, ...]

    def __post_init__(self) -> None:
        """Validate schema metadata and field names."""
        if not self.schema_id:
            raise ValueError('SchemaDefinition.schema_id must not be empty')
        if self.version < 1:
            raise ValueError('SchemaDefinition.version must be >= 1')
        validate_schema_field_names(self)

    def visible_fields(self) -> tuple[FieldSchema, ...]:
        """Return visible fields in schema order."""
        return tuple(field for field in self.fields if field.visible)

    def field_names(self) -> tuple[str, ...]:
        """Return field names in schema order."""
        return tuple(field.name for field in self.fields)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable schema dictionary."""
        return {
            'schema_id': self.schema_id,
            'version': self.version,
            'fields': [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaDefinition:
        """Create a schema definition from a dictionary.

        Unknown top-level keys are ignored for forward compatibility.
        """
        return cls(
            schema_id=str(data['schema_id']),
            version=int(data['version']),
            fields=tuple(
                FieldSchema.from_dict(field_data)
                for field_data in data['fields']
            ),
        )


def validate_schema_field_names(schema: SchemaDefinition) -> None:
    """Validate that schema field names are unique."""
    names = schema.field_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f'Duplicate schema field names: {duplicates}')


def validate_values_for_schema(
    schema: SchemaDefinition,
    values: Mapping[str, object],
    *,
    require_visible_fields: bool = True,
    allow_extra_values: bool = False,
) -> None:
    """Validate values against a schema definition."""
    validate_schema_field_names(schema)
    required_names = (
        {field.name for field in schema.visible_fields()}
        if require_visible_fields
        else set(schema.field_names())
    )
    value_names = set(values.keys())
    missing = sorted(required_names - value_names)
    if missing:
        raise KeyError(f'Values missing schema keys: {missing}')

    if not allow_extra_values:
        schema_names = set(schema.field_names())
        extra = sorted(value_names - schema_names)
        if extra:
            raise ValueError(f'Values contain keys not declared by schema: {extra}')


def validate_patch_for_schema(schema: SchemaDefinition, patch: Mapping[str, object]) -> None:
    """Validate edit patches against declared editable schema fields."""
    fields_by_name = {field.name: field for field in schema.fields}
    for name in patch:
        if name not in fields_by_name:
            raise KeyError(f'Unknown schema field: {name}')
        if not fields_by_name[name].editable:
            raise ValueError(f'Field is not editable: {name}')


def card_groups(schema: SchemaDefinition) -> tuple[str | None, ...]:
    """Return visible card groups in first-seen order."""
    groups: list[str | None] = []
    for field in schema.visible_fields():
        if field.group not in groups:
            groups.append(field.group)
    return tuple(groups)


ACQ_FILE_LIST_SCHEMA = SchemaDefinition(
    schema_id='acq_file_list',
    version=1,
    fields=(
        FieldSchema(
            name='name',
            display_name='Name',
            value_type=ValueType.STR,
            description='File name.',
            group='File',
        ),
        FieldSchema(
            name='saved',
            display_name='Saved',
            value_type=ValueType.BOOL,
            description='Checked if file is saved.',
            group='File',
        ),
        FieldSchema(
            name='path',
            display_name='Path',
            value_type=ValueType.PATH,
            visible=False,  # do not show path in the table
            description='Absolute file path.',
            group='File',
        ),
        FieldSchema(
            name='loaded',
            display_name='Loaded',
            value_type=ValueType.STR,
            description='Image and analysis loaded.',
            default_value='',
            group='File',
        ),
        FieldSchema(
            name='reference_image',
            display_name='Reference Image',
            value_type=ValueType.STR,
            description='Reference image available.',
            default_value='',
            group='Image',
        ),
        FieldSchema(
            name='file_size',
            display_name='File Size',
            value_type=ValueType.STR,
            description='File size on disk.',
            default_value='',
            group='File',
        ),
        FieldSchema(
            name='parent',
            display_name='Parent',
            value_type=ValueType.STR,
            description='Immediate parent folder name.',
            default_value='',
            group='File',
        ),
        FieldSchema(
            name='grandparent',
            display_name='Grandparent',
            value_type=ValueType.STR,
            description='Grandparent folder name.',
            default_value='',
            group='File',
        ),
        FieldSchema(
            name='condition',
            display_name='Condition',
            value_type=ValueType.STR,
            description='Experimental condition or treatment.',
            default_value='',
            visible=False,
            group='Experiment',
        ),
        FieldSchema(
            name='genotype',
            display_name='Genotype',
            value_type=ValueType.STR,
            description='Genotype or strain label.',
            default_value='',
            visible=False,
            group='Experiment',
        ),
        FieldSchema(
            name='num_channels',
            display_name='Channels',
            value_type=ValueType.INT,
            description='Number of image channels.',
            group='Image',
        ),
        FieldSchema(
            name='dims',
            display_name='Dims',
            value_type=ValueType.STR,
            description='Image axis sizes',
            default_value='',
            group='Image',
        ),
        FieldSchema(
            name='num_rois',
            display_name='ROIs',
            value_type=ValueType.INT,
            description='Number of ROIs.',
            group='ROI',
        ),
        FieldSchema(
            name='accept',
            display_name='Accept',
            value_type=ValueType.BOOL,
            editable=False,
            description='Acceptance status of this file.',
            group='File',
        ),
    ),
)
