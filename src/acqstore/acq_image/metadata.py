from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict, MISSING, replace
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import pandas as pd
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader, ReferenceImage

from acqstore.schema import (
    FieldSchema,
    SchemaDefinition,
    ValueType,
    validate_patch_for_schema,
    validate_values_for_schema,
)
from acqstore.utils.logging import get_logger

logger = get_logger(__name__)


EXPERIMENT_METADATA_SCHEMA = SchemaDefinition(
    schema_id='experiment_metadata',
    version=1,
    fields=(
        FieldSchema(
            name='species',
            display_name='Species',
            value_type=ValueType.STR,
            default_value='',
            description='Animal species (e.g., mouse, rat).',
            editable=True,
            group='Animal',
        ),
        FieldSchema(
            name='sex',
            display_name='Sex',
            value_type=ValueType.STR,
            default_value='',
            description='Biological sex or experimental sex label.',
            editable=True,
            group='Animal',
        ),
        FieldSchema(
            name='genotype',
            display_name='Genotype',
            value_type=ValueType.STR,
            default_value='',
            description='Genotype or strain label.',
            editable=True,
            group='Animal',
        ),
        FieldSchema(
            name='age',
            display_name='Age',
            value_type=ValueType.STR,
            default_value='',
            description='Animal age label (e.g., P30, 8 weeks).',
            editable=True,
            group='Animal',
        ),
        FieldSchema(
            name='region',
            display_name='Region',
            value_type=ValueType.STR,
            default_value='',
            description='Brain region or anatomical location.',
            editable=True,
            group='Sample',
        ),
        FieldSchema(
            name='cell_type',
            display_name='Cell type',
            value_type=ValueType.STR,
            default_value='',
            description='Type of cell or vessel being imaged.',
            editable=True,
            group='Sample',
        ),
        FieldSchema(
            name='depth',
            display_name='Depth',
            value_type=ValueType.FLOAT,
            default_value=None,
            description='Imaging depth in micrometers.',
            editable=True,
            group='Sample',
        ),
        FieldSchema(
            name='branch_order',
            display_name='Branch order',
            value_type=ValueType.INT,
            default_value=None,
            description='Branch order for vascular structures.',
            editable=True,
            group='Sample',
        ),
        FieldSchema(
            name='direction',
            display_name='Direction',
            value_type=ValueType.STR,
            default_value='',
            description='Flow direction or vessel orientation.',
            editable=True,
            group='Sample',
        ),
        FieldSchema(
            name='condition',
            display_name='Condition',
            value_type=ValueType.STR,
            default_value='',
            description='Experimental condition or treatment.',
            editable=True,
            group='Experiment',
        ),
        FieldSchema(
            name='condition2',
            display_name='Condition 2',
            value_type=ValueType.STR,
            default_value='',
            description='Second condition field.',
            editable=True,
            group='Experiment',
        ),
        FieldSchema(
            name='treatment',
            display_name='Treatment',
            value_type=ValueType.STR,
            default_value='',
            description='Treatment applied.',
            editable=True,
            group='Experiment',
        ),
        FieldSchema(
            name='treatment2',
            display_name='Treatment 2',
            value_type=ValueType.STR,
            default_value='',
            description='Second treatment field.',
            editable=True,
            group='Experiment',
        ),
        FieldSchema(
            name='date',
            display_name='Date',
            value_type=ValueType.STR,
            default_value='',
            description='User-editable date (e.g., experiment date).',
            editable=True,
            group='Experiment',
        ),
        FieldSchema(
            name='note',
            display_name='Note',
            value_type=ValueType.STR,
            default_value='',
            description='Free-form notes or comments.',
            editable=True,
            group='Notes',
        ),
    ),
)

@dataclass
class FieldMetadata:
    """Structured metadata for form field definitions.

    Provides type-safe field metadata to avoid typos in metadata dictionaries.
    Used by GUI forms to configure field visibility, editability, and layout.

    Attributes:
        editable: Whether the field can be edited by the user.
        label: Display label for the field.
        widget_type: Type of widget to use (e.g., "text", "number").
        grid_span: Number of grid columns this field spans.
        visible: Whether the field should be visible in forms.
        description: Human-readable description of the field for documentation.
    """

    editable: bool = True
    label: str = ""
    widget_type: str = "text"
    grid_span: int = 1
    visible: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for use in field(metadata=...).

        Returns:
            Dictionary containing all field metadata attributes
        """
        result = {
            "editable": self.editable,
            "label": self.label,
            "widget_type": self.widget_type,
            "grid_span": self.grid_span,
            "visible": self.visible,
            "description": self.description,
        }
        return result


def field_metadata(
    editable: bool = True,
    label: str = "",
    widget_type: str = "text",
    grid_span: int = 1,
    visible: bool = True,
    description: str = "",
) -> dict[str, Any]:
    """Create field metadata dictionary.

    Convenience function that creates a FieldMetadata instance and converts
    it to a dictionary suitable for use in dataclass field metadata.

    Args:
        editable: Whether the field can be edited by the user.
        label: Display label for the field.
        widget_type: Type of widget to use (e.g., "text", "number").
        grid_span: Number of grid columns this field spans.
        visible: Whether the field should be visible in forms.
        description: Human-readable description of the field for documentation.

    Returns:
        Dictionary containing field metadata attributes.
    """
    return FieldMetadata(
        editable=editable,
        label=label,
        widget_type=widget_type,
        grid_span=grid_span,
        visible=visible,
        description=description,
    ).to_dict()


def generate_schema_docs(schema: SchemaDefinition, print_markdown: bool = True) -> pd.DataFrame:
    """Generate documentation DataFrame from a SchemaDefinition.
    Args:
        schema: Semantic schema to document.
        print_markdown: If True, print markdown table to console.
    Returns:
        DataFrame with schema field documentation.
    """
    rows: list[dict[str, object]] = []
    for fs in schema.fields:
        rows.append(
            {
                'name': fs.name,
                'display_name': fs.display_name or fs.name,
                'value_type': fs.value_type.value,
                'default_value': repr(fs.default_value),
                'editable': fs.editable,
                'group': fs.group or '',
                'description': fs.description or '',
            }
        )
    df = pd.DataFrame(
        rows,
        columns=['name', 'display_name', 'value_type', 'default_value', 'editable', 'group', 'description'],
    )
    if print_markdown:
        try:
            print(f'\n## {schema.schema_id} (v{schema.version})\n')
            print(df.to_markdown(index=False))
            print()
        except ImportError as exc:
            print(df.to_string(index=False))
            print("\nNote: Install 'tabulate' for markdown table format")
            print(f'  -->> e:{exc}')
            print()
    return df
    
def _generateDocs(dc: type, print_markdown: bool = True) -> pd.DataFrame:
    """Generate documentation DataFrame from a dataclass.

    Extracts field information from a dataclass including name, display name,
    default value, and description. Optionally prints a markdown table to console.

    Args:
        dc: The dataclass type to document.
        print_markdown: If True, print markdown table to console. Defaults to True.

    Returns:
        pandas DataFrame with columns: name, display_name, default_value, description.
    """
    rows = []
    for field_obj in fields(dc):
        # Get metadata
        meta = field_obj.metadata
        
        # Extract field information
        name = field_obj.name
        display_name = meta.get("label", "") or name
        description = meta.get("description", "")
        
        # Handle default value
        if field_obj.default is not MISSING:
            # Regular default value
            default_value = field_obj.default
        elif field_obj.default_factory is not MISSING:
            # default_factory case
            factory_name = getattr(field_obj.default_factory, '__name__', 'callable')
            default_value = f"<factory: {factory_name}>"
        else:
            # No default (required field)
            default_value = "<required>"
        
        # Convert default to string representation
        if default_value is None:
            default_str = "None"
        elif isinstance(default_value, str):
            default_str = f'"{default_value}"'
        else:
            default_str = str(default_value)
        
        rows.append({
            "name": name,
            "display_name": display_name,
            "default_value": default_str,
            "description": description,
        })
    
    df = pd.DataFrame(rows)
    
    if print_markdown:
        try:
            # Try using pandas to_markdown (requires tabulate)
            print(f"\n## {dc.__name__}\n")
            print(df.to_markdown(index=False))
            print()
        except (ImportError) as e:
            # Fallback if tabulate is not available
            # print(f"\n## {dc.__name__}\n")
            print(df.to_string(index=False))
            print("\nNote: Install 'tabulate' for markdown table format")
            print(f"  -->> e:{e}")
            print()
    
    return df


@dataclass
class ExperimentMetadata:
    """User-provided experimental metadata for acquisition files.

    Field definitions and validation are driven by ``EXPERIMENT_METADATA_SCHEMA``.
    String fields are stored as ``str`` and never ``None`` (empty string is used
    for "no value"). ``depth`` and ``branch_order`` may be ``None``.

    Attributes:
        species: Animal species (e.g., "mouse", "rat").
        region: Brain region or anatomical location.
        cell_type: Type of cell or vessel being imaged.
        depth: Imaging depth in micrometers.
        branch_order: Branch order for vascular structures.
        direction: Flow direction or vessel orientation.
        sex: Animal sex.
        genotype: Genetic background or modification.
        age: Animal age label (e.g., P30, 8 weeks).
        condition: Experimental condition or treatment.
        condition2: Second condition field.
        treatment: Treatment applied.
        treatment2: Second treatment field.
        date: User-editable date (e.g., experiment date).
        note: Free-form notes or comments.
    """

    metadata_section_id: ClassVar[str] = 'experiment_metadata'
    display_section_title: ClassVar[str] = 'Experiment Metadata'

    species: str = ''
    region: str = ''
    cell_type: str = ''
    depth: float | None = None
    branch_order: int | None = None
    direction: str = ''
    sex: str = ''
    genotype: str = ''
    age: str = ''
    condition: str = ''
    condition2: str = ''
    treatment: str = ''
    treatment2: str = ''
    date: str = ''
    note: str = ''
    _is_dirty: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ExperimentMetadata:
        """Create instance from dictionary, ignoring unknown keys.

        All fields are coerced according to ``EXPERIMENT_METADATA_SCHEMA``.
        String fields accept ``None`` in the payload and coerce to ``""``.
        Numeric fields accept JSON strings such as ``"75"`` and coerce to
        ``float``/``int``. Empty strings coerce to ``None`` for nullable
        numeric fields.

        Args:
            payload: Dictionary containing metadata fields. Can be None or empty.

        Returns:
            ``ExperimentMetadata`` instance with values from payload, or defaults
            if payload is None or empty.

        Raises:
            ValueError: If a field value cannot be coerced to its schema type.
        """
        payload = payload or {}
        valid = {f.name for f in fields(cls) if f.init}
        fields_by_name = {fs.name: fs for fs in EXPERIMENT_METADATA_SCHEMA.fields}
        known: dict[str, Any] = {}
        for k in payload.keys() & valid:
            field_schema = fields_by_name[k]
            try:
                known[k] = cls._coerce_schema_value(field_schema, payload[k])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid experiment_metadata field {k!r}: {exc}"
                ) from exc
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        """Return init field values as a plain dictionary (not schema-validated).

        Internal state such as ``_is_dirty`` is excluded so sidecar JSON stays
        limited to experiment-metadata fields.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def is_dirty(self) -> bool:
        """Return whether metadata was edited since last ``set_clean``."""
        return self._is_dirty

    def set_clean(self) -> None:
        """Reset dirty flag after persisting metadata changes."""
        self._is_dirty = False

    def get_schema(self) -> SchemaDefinition:
        """Return the semantic schema for this metadata section."""
        return EXPERIMENT_METADATA_SCHEMA

    def get_values(self) -> dict[str, object]:
        """Return values keyed by ``FieldSchema.name`` for ``get_schema()``."""
        values: dict[str, object] = {}
        for fs in self.get_schema().fields:
            raw = getattr(self, fs.name)
            if fs.value_type is ValueType.STR:
                values[fs.name] = '' if raw is None else str(raw)
            else:
                values[fs.name] = raw
        validate_values_for_schema(self.get_schema(), values)
        return values

    @staticmethod
    def _coerce_schema_value(fs: FieldSchema, value: object) -> object:
        """Coerce one value to the type declared by ``fs``.

        Args:
            fs: Schema field describing the target type.
            value: Raw input value from a dict payload or edit patch.

        Returns:
            Coerced value suitable for ``setattr`` on ``ExperimentMetadata``.

        Raises:
            ValueError: If ``value`` cannot be coerced to ``fs.value_type``.
        """
        if fs.value_type is ValueType.STR:
            return '' if value is None else str(value).strip()
        if fs.value_type is ValueType.INT:
            if value is None:
                return None
            if isinstance(value, str) and value.strip() == '':
                return None
            try:
                float_value = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"expected integer-compatible value, got {value!r}") from exc
            int_value = int(float_value)
            if float_value != int_value:
                raise ValueError(f"expected integer-compatible value, got {value!r}")
            return int_value
        if fs.value_type is ValueType.FLOAT:
            if value is None:
                return None
            if isinstance(value, str) and value.strip() == '':
                return None
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"expected float-compatible value, got {value!r}") from exc
        return value

    @staticmethod
    def _coerce_patch_value(fs: FieldSchema, value: object) -> object:
        """Coerce a single patch value to a type suitable for ``setattr``."""
        return ExperimentMetadata._coerce_schema_value(fs, value)

    def update_values(self, patch: dict[str, object]) -> None:
        """Apply an edit patch to editable fields only.

        Args:
            patch: Mapping from schema field names to new values.

        Raises:
            KeyError: If a patch key is not declared by the schema.
            ValueError: If a patch targets a non-editable field.
        """
        schema = self.get_schema()
        validate_patch_for_schema(schema, patch)
        fields_by_name = {f.name: f for f in schema.fields}
        changed = False
        for name, value in patch.items():
            coerced = self._coerce_patch_value(fields_by_name[name], value)
            if getattr(self, name) != coerced:
                changed = True
            setattr(self, name, coerced)
        if changed:
            self._is_dirty = True


IMAGE_HEADER_METADATA_SCHEMA = SchemaDefinition(
    schema_id='acq_image_header',
    version=1,
    fields=(
        FieldSchema(
            name='shape', display_name='Shape', value_type=ValueType.STR, default_value='', editable=False, group='Header'
        ),
        FieldSchema(
            name='dims', display_name='Dims', value_type=ValueType.STR, default_value='', editable=False, group='Header'
        ),
        FieldSchema(
            name='sizes', display_name='Sizes', value_type=ValueType.STR, default_value='', editable=False, group='Header'
        ),
        FieldSchema(
            name='dtype', display_name='DType', value_type=ValueType.STR, default_value='', editable=False, group='Header'
        ),
        FieldSchema(
            name='num_channels',
            display_name='Channels',
            value_type=ValueType.INT,
            default_value=0,
            editable=False,
            group='Header',
        ),
        FieldSchema(
            name='num_scenes',
            display_name='Scenes',
            value_type=ValueType.INT,
            default_value=0,
            editable=False,
            visible=False,
            group='Header',
        ),
        FieldSchema(
            name='date', display_name='Date', value_type=ValueType.STR, default_value='', editable=False, group='Header'
        ),
        FieldSchema(
            name='time', display_name='Time', value_type=ValueType.STR, default_value='', editable=False, group='Header'
        ),
        FieldSchema(
            name='physical_unit_y',
            display_name='Physical Unit Y',
            value_type=ValueType.FLOAT,
            default_value=1.0,
            editable=True,
            group='Calibration',
        ),
        FieldSchema(
            name='physical_unit_x',
            display_name='Physical Unit X',
            value_type=ValueType.FLOAT,
            default_value=1.0,
            editable=True,
            group='Calibration',
        ),
        FieldSchema(
            name='physical_label_y',
            display_name='Physical Label Y',
            value_type=ValueType.STR,
            default_value='Pixels',
            editable=True,
            group='Calibration',
        ),
        FieldSchema(
            name='physical_label_x',
            display_name='Physical Label X',
            value_type=ValueType.STR,
            default_value='Pixels',
            editable=True,
            group='Calibration',
        ),
    ),
)


class ImageHeaderMetadata:
    """Schema-driven metadata adapter for ``ImageHeader``.

    Exposes loader header details as read-only schema fields, while allowing
    editing of Y/X physical unit and label calibration.
    """

    metadata_section_id: ClassVar[str] = 'acq_image_header'
    display_section_title: ClassVar[str] = 'Image Header'

    def __init__(self, image_header: ImageHeader, apply_header: Callable[[ImageHeader], None]) -> None:
        """Construct adapter around a snapshot header.

        The header is normalized with
        :meth:`ImageHeader.with_coerced_physical_calibration` so displayed values
        match loader/header defaults rather than ad hoc coercion in this layer.

        Args:
            image_header: Current ``ImageHeader`` (typically from a file loader).
            apply_header: Callback invoked after calibration edits; must persist
                the header on the backing loader (for example
                :meth:`BaseFileLoader.replace_header`).
        """
        self._header = image_header.with_coerced_physical_calibration()
        self._apply_header = apply_header
        self._is_dirty = False

    def is_dirty(self) -> bool:
        """Return whether metadata was edited since last ``set_clean``."""
        return self._is_dirty

    def set_clean(self) -> None:
        """Reset dirty flag after persisting metadata changes."""
        self._is_dirty = False

    def get_schema(self) -> SchemaDefinition:
        """Return schema for the header metadata section."""
        return IMAGE_HEADER_METADATA_SCHEMA

    def _dim_index(self, dim: str) -> int:
        if dim not in self._header.dims:
            raise ValueError(f'Image header missing required {dim!r} dim: {self._header.dims!r}')
        return self._header.dims.index(dim)

    def _header_unit(self, dim: str) -> float:
        idx = self._dim_index(dim)
        return float(self._header.physical_units[idx])

    def _header_label(self, dim: str) -> str:
        idx = self._dim_index(dim)
        return str(self._header.physical_units_labels[idx])

    def get_values(self) -> dict[str, object]:
        """Return header values keyed by schema field names."""
        values: dict[str, object] = {
            'shape': str(self._header.shape),
            'dims': str(self._header.dims),
            'sizes': str(self._header.sizes),
            'dtype': str(self._header.dtype),
            'num_channels': int(self._header.num_channels),
            'num_scenes': int(self._header.num_scenes),
            'date': str(self._header.date),
            'time': str(self._header.time),
            'physical_unit_y': self._header_unit('Y'),
            'physical_unit_x': self._header_unit('X'),
            'physical_label_y': self._header_label('Y'),
            'physical_label_x': self._header_label('X'),
        }
        validate_values_for_schema(self.get_schema(), values)
        return values

    @classmethod
    def editable_field_names(cls) -> frozenset[str]:
        """Return schema field names that may be hydrated from a sidecar patch."""
        return frozenset(
            field.name for field in IMAGE_HEADER_METADATA_SCHEMA.fields if field.editable
        )

    @classmethod
    def editable_patch_from_sidecar(cls, raw: Mapping[str, object]) -> dict[str, object]:
        """Return editable calibration keys present in one sidecar header dict.

        Non-editable and unknown keys are ignored so file-derived structural
        header fields remain authoritative on load.

        Args:
            raw: ``image_header_metadata`` object from an AcqImage sidecar.

        Returns:
            Patch suitable for :meth:`update_values` (may be empty).
        """
        editable = cls.editable_field_names()
        return {name: raw[name] for name in raw if name in editable}

    def apply_sidecar_calibration(self, raw: Mapping[str, object]) -> None:
        """Apply editable calibration from sidecar without marking the section dirty.

        Args:
            raw: ``image_header_metadata`` object from an AcqImage sidecar.

        Raises:
            ValueError: If ``raw`` contains invalid editable calibration values.
        """
        patch = self.editable_patch_from_sidecar(raw)
        if not patch:
            return
        self.update_values(patch)
        self.set_clean()

    def update_values(self, patch: dict[str, object]) -> None:
        """Apply editable calibration patch and write back to loader header."""
        schema = self.get_schema()
        validate_patch_for_schema(schema, patch)
        if not patch:
            return
        current = self.get_values()
        merged = dict(current)
        merged.update(patch)
        unit_y = float(merged['physical_unit_y'])  # raises on invalid
        unit_x = float(merged['physical_unit_x'])
        if unit_y <= 0 or unit_x <= 0:
            raise ValueError('physical_unit_y and physical_unit_x must be > 0')
        label_y = '' if merged['physical_label_y'] is None else str(merged['physical_label_y'])
        label_x = '' if merged['physical_label_x'] is None else str(merged['physical_label_x'])

        units = list(self._header.physical_units)
        labels = list(self._header.physical_units_labels)
        y_idx = self._dim_index('Y')
        x_idx = self._dim_index('X')
        changed = False
        if float(units[y_idx]) != unit_y:
            units[y_idx] = unit_y
            changed = True
        if float(units[x_idx]) != unit_x:
            units[x_idx] = unit_x
            changed = True
        if str(labels[y_idx]) != label_y:
            labels[y_idx] = label_y
            changed = True
        if str(labels[x_idx]) != label_x:
            labels[x_idx] = label_x
            changed = True
        if not changed:
            return

        self._header = replace(
            self._header,
            physical_units=tuple(units),
            physical_units_labels=tuple(labels),
        )
        self._apply_header(self._header)
        self._is_dirty = True


def _normalize_reference_physical_label(label: str) -> str:
    """Return a stable physical-unit label for reference-image metadata."""
    normalized = label.strip().lower()
    if normalized in ('micrometer', 'µm', 'um'):
        return 'um'
    return label


REFERENCE_IMAGE_METADATA_SCHEMA = SchemaDefinition(
    schema_id='reference_image_metadata',
    version=2,
    fields=(
        FieldSchema(
            name='shape',
            display_name='Shape',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            group='Reference',
        ),
        FieldSchema(
            name='dims',
            display_name='Dims',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            group='Reference',
        ),
        FieldSchema(
            name='sizes',
            display_name='Sizes',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            group='Reference',
        ),
        FieldSchema(
            name='dtype',
            display_name='DType',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            group='Reference',
        ),
        FieldSchema(
            name='num_channels',
            display_name='Channels',
            value_type=ValueType.INT,
            default_value=0,
            editable=False,
            group='Reference',
        ),
        FieldSchema(
            name='physical_unit_y',
            display_name='Physical Unit Y',
            value_type=ValueType.FLOAT,
            default_value=1.0,
            editable=False,
            group='Calibration',
        ),
        FieldSchema(
            name='physical_unit_x',
            display_name='Physical Unit X',
            value_type=ValueType.FLOAT,
            default_value=1.0,
            editable=False,
            group='Calibration',
        ),
        FieldSchema(
            name='physical_label_y',
            display_name='Physical Label Y',
            value_type=ValueType.STR,
            default_value='um',
            editable=False,
            group='Calibration',
        ),
        FieldSchema(
            name='physical_label_x',
            display_name='Physical Label X',
            value_type=ValueType.STR,
            default_value='um',
            editable=False,
            group='Calibration',
        ),
        FieldSchema(
            name='has_scan_path',
            display_name='Has Scan Path',
            value_type=ValueType.BOOL,
            default_value=False,
            editable=False,
            group='Scan path',
        ),
        FieldSchema(
            name='scan_path_num_points',
            display_name='Scan Path Points',
            value_type=ValueType.INT,
            default_value=0,
            editable=False,
            group='Scan path',
        ),
        FieldSchema(
            name='line_roi',
            display_name='Line ROI',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            group='Scan path',
        ),
        FieldSchema(
            name='scan_path_x_pixels',
            display_name='Scan Path X (px)',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            visible=False,
            group='Scan path',
        ),
        FieldSchema(
            name='scan_path_y_pixels',
            display_name='Scan Path Y (px)',
            value_type=ValueType.STR,
            default_value='',
            editable=False,
            visible=False,
            group='Scan path',
        ),
    ),
)


class ReferenceImageMetadata:
    """Schema-driven metadata adapter for a reference/overview image snapshot."""

    metadata_section_id: ClassVar[str] = 'reference_image_metadata'
    display_section_title: ClassVar[str] = 'Reference Image'

    def __init__(self, reference_image: ReferenceImage) -> None:
        """Construct adapter around one frozen :class:`ReferenceImage`.

        Args:
            reference_image: Reference snapshot from the backing file loader.
        """
        self._reference_image = reference_image
        self._is_dirty = False

    @classmethod
    def from_reference_image(cls, reference_image: ReferenceImage) -> ReferenceImageMetadata:
        """Build metadata from one loader reference snapshot.

        Args:
            reference_image: Frozen reference image from a file loader.

        Returns:
            Metadata adapter for the reference image.
        """
        return cls(reference_image)

    def is_dirty(self) -> bool:
        """Return whether metadata was edited since last ``set_clean``."""
        return self._is_dirty

    def set_clean(self) -> None:
        """Reset dirty flag after persisting metadata changes."""
        self._is_dirty = False

    def get_schema(self) -> SchemaDefinition:
        """Return schema for the reference-image metadata section."""
        return REFERENCE_IMAGE_METADATA_SCHEMA

    def _scale_for_dim(self, dim: str) -> float:
        mapping = dict(self._reference_image.coord_scales)
        value = mapping.get(dim)
        if value is None:
            return 1.0
        scale = float(value)
        if scale <= 0.0:
            return 1.0
        return scale

    def _label_for_dim(self, dim: str) -> str:
        unit = dict(self._reference_image.coord_units).get(dim)
        if unit:
            return _normalize_reference_physical_label(str(unit))
        return 'um'

    def _scan_path_values(self) -> dict[str, object]:
        """Return scan-path summary and coordinate lists for metadata export."""
        line_roi = self._reference_image.get_line_roi()
        line_roi_str = '' if line_roi is None else str(tuple(float(v) for v in line_roi))
        if not self._reference_image.has_scan_path():
            return {
                'has_scan_path': False,
                'scan_path_num_points': 0,
                'line_roi': line_roi_str,
                'scan_path_x_pixels': [],
                'scan_path_y_pixels': [],
            }
        scan_path_plot = self._reference_image.get_scan_path_plot()
        if scan_path_plot is None:
            return {
                'has_scan_path': False,
                'scan_path_num_points': 0,
                'line_roi': line_roi_str,
                'scan_path_x_pixels': [],
                'scan_path_y_pixels': [],
            }
        x_pixels, y_pixels = scan_path_plot
        return {
            'has_scan_path': True,
            'scan_path_num_points': int(len(x_pixels)),
            'line_roi': line_roi_str,
            'scan_path_x_pixels': [float(value) for value in x_pixels],
            'scan_path_y_pixels': [float(value) for value in y_pixels],
        }

    def get_values(self) -> dict[str, object]:
        """Return reference-image values keyed by schema field names."""
        plane = self._reference_image.get_plane(channel=0)
        y_size, x_size = (int(plane.array.shape[0]), int(plane.array.shape[1]))
        dims = ('Y', 'X')
        sizes = {'Y': y_size, 'X': x_size}
        values: dict[str, object] = {
            'shape': str((y_size, x_size)),
            'dims': str(dims),
            'sizes': str(sizes),
            'dtype': str(self._reference_image.array.dtype),
            'num_channels': int(self._reference_image.num_channels),
            'physical_unit_y': self._scale_for_dim('Y'),
            'physical_unit_x': self._scale_for_dim('X'),
            'physical_label_y': self._label_for_dim('Y'),
            'physical_label_x': self._label_for_dim('X'),
        }
        values.update(self._scan_path_values())
        validate_values_for_schema(self.get_schema(), values)
        return values

    def apply_sidecar_calibration(self, raw: Mapping[str, object]) -> None:
        """No-op for v1: reference metadata fields are file-derived and read-only.

        Args:
            raw: ``reference_image_metadata`` object from an AcqImage sidecar.
        """
        _ = raw

    def update_values(self, patch: dict[str, object]) -> None:
        """Reject edits because reference metadata is read-only in v1.

        Args:
            patch: Field patch for this section.

        Raises:
            ValueError: If ``patch`` contains any keys.
        """
        validate_patch_for_schema(self.get_schema(), patch)
        if patch:
            raise ValueError('reference_image_metadata fields are read-only')


