"""Tests for AcqImage sidecar JSON persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.image_contrast import ImageContrast
from acqstore.acq_image.roi import LineEndpoints, RectRoiBounds

_OIR_SAMPLES = Path(__file__).resolve().parent / 'data' / 'oir-samples'
_KYMOGRAPH = _OIR_SAMPLES / '20251030_A106_0002.oir'


def _write_tif(path: Path) -> None:
    tifffile.imwrite(path, np.zeros((10, 20), dtype=np.uint8))


def test_sidecar_path_uses_full_filename_plus_json(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    acq = AcqImage(str(path))
    assert acq.get_sidecar_json_path() == str(path.resolve()) + '.json'


def test_save_writes_expected_top_level_contract(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    acq = AcqImage(str(path))
    acq.rois.create_rect_roi(RectRoiBounds(1, 5, 2, 7), name='rect')
    acq.rois.create_line_roi(LineEndpoints(1, 2, 3, 4), name='line')
    acq.apply_metadata_patch('experiment_metadata', {'species': 'mouse'})

    acq.save()

    sidecar = Path(acq.get_sidecar_json_path())
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    assert set(payload.keys()) == {
        'accepted',
        'analysis',
        'experiment_metadata',
        'image_contrast',
        'image_header_metadata',
        'rois',
        'version',
    }
    assert payload['version'] == 2
    assert isinstance(payload['rois'], list)
    assert payload['experiment_metadata']['species'] == 'mouse'
    assert payload['image_contrast'] == {}


def test_load_round_trip_restores_rois_and_experiment_metadata(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)

    source = AcqImage(str(path))
    source.rois.create_rect_roi(RectRoiBounds(1, 5, 2, 7), name='rect')
    source.rois.create_line_roi(LineEndpoints(1, 2, 3, 4), name='line')
    source.apply_metadata_patch('experiment_metadata', {'species': 'mouse', 'genotype': 'wt'})
    source.save()

    loaded = AcqImage(str(path))
    assert loaded.rois.num_rois == 2
    assert loaded.rois.get_roi_ids() == [1, 2]
    exp = loaded.get_metadata_section('experiment_metadata')
    assert exp.species == 'mouse'
    assert exp.genotype == 'wt'


def test_load_round_trip_restores_image_header_calibration(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)

    source = AcqImage(str(path))
    source.apply_metadata_patch(
        'acq_image_header',
        {
            'physical_unit_x': 0.02,
            'physical_unit_y': 0.001,
            'physical_label_x': 'um',
            'physical_label_y': 'seconds',
        },
    )
    source.save()

    loaded = AcqImage(str(path))
    header = loaded.get_metadata_section('acq_image_header').get_values()
    assert header['physical_unit_x'] == 0.02
    assert header['physical_unit_y'] == 0.001
    assert header['physical_label_x'] == 'um'
    assert header['physical_label_y'] == 'seconds'
    assert loaded.images.header.physical_units[loaded.images.header.dims.index('X')] == 0.02
    assert loaded.images.header.physical_units[loaded.images.header.dims.index('Y')] == 0.001
    assert loaded.is_dirty is False


def test_load_applies_partial_image_header_calibration_patch(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    acq = AcqImage(str(path))
    acq.save()

    sidecar = Path(acq.get_sidecar_json_path())
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    payload['image_header_metadata'] = {'physical_unit_x': 0.05}
    sidecar.write_text(json.dumps(payload), encoding='utf-8')

    loaded = AcqImage(str(path))
    header = loaded.get_metadata_section('acq_image_header').get_values()
    assert header['physical_unit_x'] == 0.05
    assert header['physical_unit_y'] == 1.0
    assert header['physical_label_x'] == 'Pixels'
    assert header['physical_label_y'] == 'Pixels'


def test_load_ignores_non_editable_image_header_sidecar_keys(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    acq = AcqImage(str(path))
    acq.save()

    sidecar = Path(acq.get_sidecar_json_path())
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    payload['image_header_metadata']['shape'] = '(999, 999)'
    payload['image_header_metadata']['num_channels'] = 99
    payload['image_header_metadata']['physical_unit_x'] = 0.02
    sidecar.write_text(json.dumps(payload), encoding='utf-8')

    loaded = AcqImage(str(path))
    header = loaded.get_metadata_section('acq_image_header').get_values()
    assert header['shape'] == str(loaded.images.header.shape)
    assert header['shape'] != '(999, 999)'
    assert header['num_channels'] == int(loaded.images.header.num_channels)
    assert header['num_channels'] != 99
    assert header['physical_unit_x'] == 0.02


def test_load_tolerates_invalid_image_header_calibration(tmp_path: Path, caplog) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    sidecar_path = Path(str(path.resolve()) + '.json')
    sidecar_path.write_text(
        json.dumps(
            {
                'version': 2,
                'accepted': True,
                'rois': [],
                'experiment_metadata': {'species': 'mouse'},
                'image_header_metadata': {
                    'physical_unit_x': -1.0,
                    'physical_unit_y': 0.001,
                },
                'analysis': [],
            }
        ),
        encoding='utf-8',
    )

    with caplog.at_level('WARNING'):
        loaded = AcqImage(str(path))

    header = loaded.get_metadata_section('acq_image_header').get_values()
    assert header['physical_unit_x'] == 1.0
    assert header['physical_unit_y'] == 1.0
    assert loaded.get_metadata_section('experiment_metadata').species == 'mouse'
    assert any('Skipping image_header_metadata calibration' in r.message for r in caplog.records)


def test_malformed_or_invalid_sidecar_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    sidecar = Path(str(path.resolve()) + '.json')
    sidecar.write_text('{"version": 999, "rois": []}', encoding='utf-8')

    loaded = AcqImage(str(path))
    assert loaded.rois.num_rois == 0
    exp = loaded.get_metadata_section('experiment_metadata')
    assert exp.species == ''


def test_image_contrast_round_trip_single_and_many_channels(tmp_path: Path) -> None:
    """``image_contrast`` round-trips through the sidecar for multiple channels."""
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    source = AcqImage(str(path))
    source.set_image_contrast(
        0, ImageContrast(color_lut='Green', value_min=5, value_max=240, img_min=0, img_max=255)
    )
    source.set_image_contrast(
        2, ImageContrast(color_lut='Plasma', value_min=10, value_max=200, img_min=0, img_max=4095)
    )
    source.save()

    payload = json.loads(Path(source.get_sidecar_json_path()).read_text(encoding='utf-8'))
    assert payload['image_contrast'] == {
        '0': {
            'color_lut': 'Green',
            'value_min': 5,
            'value_max': 240,
            'img_min': 0,
            'img_max': 255,
        },
        '2': {
            'color_lut': 'Plasma',
            'value_min': 10,
            'value_max': 200,
            'img_min': 0,
            'img_max': 4095,
        },
    }

    loaded = AcqImage(str(path))
    a = loaded.get_image_contrast(0)
    b = loaded.get_image_contrast(2)
    assert a is not None and a.color_lut == 'Green' and (a.value_min, a.value_max) == (5, 240)
    assert b is not None and b.color_lut == 'Plasma' and (b.value_min, b.value_max) == (10, 200)
    # Loaded entries should not mark the file dirty.
    assert loaded.is_dirty is False


def test_load_succeeds_when_image_contrast_key_is_absent(tmp_path: Path) -> None:
    """Old v2 sidecars (without ``image_contrast``) load cleanly and produce no entries."""
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    sidecar_path = Path(str(path.resolve()) + '.json')
    sidecar_path.write_text(
        json.dumps(
            {
                'version': 2,
                'accepted': True,
                'rois': [],
                'experiment_metadata': {},
                'image_header_metadata': {},
                'analysis': [],
            }
        ),
        encoding='utf-8',
    )

    loaded = AcqImage(str(path))
    assert loaded.get_image_contrast(0) is None
    assert loaded.is_dirty is False


def test_load_tolerates_malformed_image_contrast_entries(tmp_path: Path) -> None:
    """Malformed entries are skipped with a warning; valid entries still load."""
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    sidecar_path = Path(str(path.resolve()) + '.json')
    sidecar_path.write_text(
        json.dumps(
            {
                'version': 2,
                'accepted': True,
                'rois': [],
                'experiment_metadata': {},
                'image_header_metadata': {},
                'analysis': [],
                'image_contrast': {
                    '0': {
                        'color_lut': 'Gray',
                        'value_min': 0,
                        'value_max': 255,
                        'img_min': 0,
                        'img_max': 255,
                    },
                    'not-a-channel': {
                        'color_lut': 'Gray',
                        'value_min': 0,
                        'value_max': 255,
                        'img_min': 0,
                        'img_max': 255,
                    },
                    '7': {'color_lut': 'Gray'},  # missing required keys
                    '9': 'not-an-object',
                },
            }
        ),
        encoding='utf-8',
    )

    loaded = AcqImage(str(path))
    a = loaded.get_image_contrast(0)
    assert a is not None and a.color_lut == 'Gray'
    assert loaded.get_image_contrast(7) is None
    assert loaded.get_image_contrast(9) is None


def test_image_contrast_key_does_not_trigger_unknown_key_warning(
    tmp_path: Path, caplog
) -> None:
    """Writing the optional ``image_contrast`` key must not warn on the next load."""
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    src = AcqImage(str(path))
    src.save()
    with caplog.at_level('WARNING'):
        AcqImage(str(path))
    assert not any('Ignoring unknown AcqImage sidecar keys' in r.message for r in caplog.records)


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason='kymograph OIR fixture missing')
def test_save_includes_reference_image_metadata_for_oir(tmp_path: Path) -> None:
    """OIR files with a reference image persist reference metadata in the sidecar."""
    oir_path = tmp_path / _KYMOGRAPH.name
    shutil.copy(_KYMOGRAPH, oir_path)
    acq = AcqImage(str(oir_path), load_images=False, load_analysis_csv=False)
    assert acq.images.has_reference_image
    acq.save()

    payload = json.loads(Path(acq.get_sidecar_json_path()).read_text(encoding='utf-8'))
    assert 'reference_image_metadata' in payload
    ref_meta = payload['reference_image_metadata']
    assert ref_meta['dims'] == "('Y', 'X')"
    assert ref_meta['physical_label_x'] == 'um'
    assert ref_meta['physical_label_y'] == 'um'
    assert ref_meta['physical_unit_x'] == pytest.approx(0.274, rel=1e-3)
    assert ref_meta['physical_unit_y'] == pytest.approx(0.274, rel=1e-3)
    assert 'has_scan_path' in ref_meta
    assert 'scan_path_x_pixels' in ref_meta
    assert isinstance(ref_meta['scan_path_x_pixels'], list)


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason='kymograph OIR fixture missing')
def test_load_round_trip_restores_reference_image_metadata(tmp_path: Path) -> None:
    """Reference metadata round-trips through save/load from file-derived values."""
    oir_path = tmp_path / _KYMOGRAPH.name
    shutil.copy(_KYMOGRAPH, oir_path)
    source = AcqImage(str(oir_path), load_images=False, load_analysis_csv=False)
    source.save()

    loaded = AcqImage(str(oir_path), load_images=False, load_analysis_csv=False)
    ref = loaded.get_metadata_section('reference_image_metadata').get_values()
    assert ref['shape'] == '(512, 512)'
    assert ref['physical_unit_x'] == pytest.approx(0.274, rel=1e-3)
    assert ref['physical_unit_y'] == pytest.approx(0.274, rel=1e-3)


def test_reference_image_metadata_key_does_not_trigger_unknown_key_warning(
    tmp_path: Path, caplog
) -> None:
    """Optional ``reference_image_metadata`` must not warn on load."""
    path = tmp_path / 'sample.tif'
    _write_tif(path)
    sidecar = Path(str(path.resolve()) + '.json')
    sidecar.write_text(
        json.dumps(
            {
                'version': 2,
                'accepted': True,
                'rois': [],
                'experiment_metadata': {},
                'image_header_metadata': {},
                'analysis': [],
                'reference_image_metadata': {'dtype': 'uint8'},
            }
        ),
        encoding='utf-8',
    )
    with caplog.at_level('WARNING'):
        AcqImage(str(path))
    assert not any('Ignoring unknown AcqImage sidecar keys' in r.message for r in caplog.records)
