"""Tests for the AcqStore Web Dataset v1 exporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.analysis.model import AnalysisPlotData, AnalysisResult, BaseAnalysis
from acqstore.acq_image.roi import RectRoiBounds
from acqstore.acq_image.web_export import (
    build_acq_image_document,
    build_analysis_document,
    build_dataset_document,
    build_dataset_image_index,
    export_acq_image,
    export_acq_image_list,
)


class _PlotAnalysis(BaseAnalysis):
    analysis_name = "diameter"

    def run(self, data_provider, *, context=None, dependencies=None) -> AnalysisResult:
        raise NotImplementedError

    def get_plot_data(self) -> AnalysisPlotData:
        return AnalysisPlotData(
            x=(0.0, 1.0),
            y=(2.0, 3.0),
            x_label="Time (s)",
            y_label="Diameter (um)",
            series_name="Diameter",
        )


class _PeakAnalysis(BaseAnalysis):
    analysis_name = "sum_intensity"

    def run(self, data_provider, *, context=None, dependencies=None) -> AnalysisResult:
        raise NotImplementedError

    def get_plot_data(self) -> AnalysisPlotData:
        return AnalysisPlotData(
            x=(0.0, 1.0),
            y=(1.0, 4.0),
            x_label="Time (s)",
            y_label="df/f0",
            series_name="df/f0 signal",
        )

    @classmethod
    def get_pool_peak_columns(cls) -> tuple[str, ...]:
        return ("peak_id", "peak_time_sec", "peak_value")

    def get_pool_peak_rows(self) -> tuple[dict[str, object], ...]:
        return ({"peak_id": 1, "peak_time_sec": 1.0, "peak_value": 4.0},)


def _acq() -> AcqImage:
    acq = AcqImage.from_array(
        np.arange(2 * 8 * 10, dtype=np.uint16).reshape(2, 8, 10),
        axes=("C", "Y", "X"),
        source_id="cell01.oir",
        axis_spacing={"C": 1.0, "Y": 0.5, "X": 0.25},
        axis_units={"C": "channel", "Y": "um", "X": "um"},
    )
    roi = acq.rois.create_rect_roi(
        RectRoiBounds(dim0_start=1, dim0_stop=6, dim1_start=2, dim1_stop=9),
        name="ROI 1",
    )

    diameter = _PlotAnalysis(channel=0, roi_id=roi.roi_id)
    diameter.result = AnalysisResult(
        summary={"mean": 2.5},
        table=pd.DataFrame({"time_s": [0.0, 1.0], "diameter_um": [2.0, 3.0]}),
    )
    acq.analysis_set.add(diameter)

    sum_intensity = _PeakAnalysis(channel=0, roi_id=roi.roi_id)
    sum_intensity.result = AnalysisResult(
        summary={"num_peaks": 1},
        table=pd.DataFrame({"time_sec": [0.0, 1.0], "df_f_signal": [1.0, 4.0]}),
    )
    acq.analysis_set.add(sum_intensity)
    return acq


def _fake_ome_zarr(self: AcqImage, path: str | Path, **kwargs) -> None:
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=False)
    (dest / ".test-placeholder").write_text("OME-Zarr writer invoked\n", encoding="utf-8")


def test_export_acq_image_writes_thin_web_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acq = _acq()
    monkeypatch.setattr(AcqImage, "save_as_ome_zarr", _fake_ome_zarr)

    destination = tmp_path / "image-package"
    export_acq_image(acq, destination)

    payload = json.loads((destination / "acqimage.json").read_text(encoding="utf-8"))
    assert payload["format"] == "acqstore-web-acqimage"
    assert payload["format_version"] == 1
    assert payload["image"]["shape"] == [2, 8, 10]
    assert payload["image"]["dims"] == ["c", "y", "x"]
    assert payload["image"]["sizes"] == {"c": 2, "y": 8, "x": 10}
    assert payload["image"]["axes"][1]["spacing"] == 0.5
    assert payload["image"]["axes"][2]["unit"] == "um"
    assert payload["rois"][0] == {
        "id": 1,
        "type": "rect",
        "name": "ROI 1",
        "note": "",
        "x_start": 2,
        "x_stop": 9,
        "y_start": 1,
        "y_stop": 6,
    }

    by_type = {item["analysis_type"]: item for item in payload["analyses"]}
    assert set(by_type) == {"diameter", "sum_intensity"}
    assert by_type["diameter"]["plot"]["series_name"] == "Diameter"
    assert by_type["sum_intensity"]["peaks"]["count"] == 1
    assert (destination / by_type["diameter"]["table"]["href"]).is_file()
    assert (destination / by_type["diameter"]["plot"]["href"]).is_file()
    assert (destination / by_type["sum_intensity"]["peaks"]["href"]).is_file()
    assert (destination / "image.ome.zarr").is_dir()
    assert str(tmp_path) not in (destination / "acqimage.json").read_text(encoding="utf-8")


def test_web_document_builders_are_transport_neutral() -> None:
    acq = _acq()
    analysis = acq.analysis_set.as_list()[0]
    analysis_document = build_analysis_document(
        analysis,
        table_href="/api/table.csv",
        plot_href="/api/plot.csv",
    )
    detail = build_acq_image_document(
        acq,
        image_id="image-1",
        image_href="/api/planes",
        analyses=[analysis_document],
    )
    row = build_dataset_image_index(detail, image_id="image-1", href="/api/images/image-1")
    dataset = build_dataset_document(
        dataset_id="dataset-1",
        name="Live dataset",
        images=[row],
        created_utc="2026-08-14T00:00:00Z",
    )

    assert detail["image"]["href"] == "/api/planes"
    assert detail["analyses"][0]["plot"]["href"] == "/api/plot.csv"
    assert dataset["images"][0]["href"] == "/api/images/image-1"
    assert dataset["format_version"] == 1


def test_export_skips_analysis_without_result_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acq = _acq()
    pending = _PlotAnalysis(channel=1, roi_id=1)
    acq.analysis_set.add(pending)
    monkeypatch.setattr(AcqImage, "save_as_ome_zarr", _fake_ome_zarr)

    destination = tmp_path / "image-package"
    export_acq_image(acq, destination)
    payload = json.loads((destination / "acqimage.json").read_text(encoding="utf-8"))

    identities = {(a["analysis_type"], a["channel"], a["roi_id"]) for a in payload["analyses"]}
    assert ("diameter", 1, 1) not in identities


def test_web_export_keeps_kymograph_semantics_outside_safe_raster_ngff(tmp_path: Path) -> None:
    """The manifest retains time calibration while NGFF uses raster Y pixels."""
    import zarr

    acq = AcqImage.from_array(
        np.arange(32 * 20, dtype=np.uint16).reshape(32, 20),
        axes=("Y", "X"),
        source_id="kymograph.tif",
        axis_spacing={"Y": 0.002, "X": 0.4},
        axis_units={"Y": "seconds", "X": "micrometer"},
    )
    destination = tmp_path / "image-package"

    export_acq_image(acq, destination)

    payload = json.loads((destination / "acqimage.json").read_text(encoding="utf-8"))
    multiscale = dict(zarr.open_group(str(destination / "image.ome.zarr"), mode="r").attrs)["ome"][
        "multiscales"
    ][0]
    assert payload["image"]["axes"] == [
        {"name": "y", "size": 32, "spacing": 0.002, "unit": "seconds"},
        {"name": "x", "size": 20, "spacing": 0.4, "unit": "micrometer"},
    ]
    assert multiscale["axes"] == [
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 0.4]


def test_dataset_index_is_single_fetch_table_summary_and_preserves_id_on_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acq = _acq()
    monkeypatch.setattr(AcqImage, "save_as_ome_zarr", _fake_ome_zarr)

    images = AcqImageList.__new__(AcqImageList)
    images.path = "/data/experiment-001"
    images.source_root_path = "/data/experiment-001"
    images.file_list = [acq.path]
    images._files = [acq]
    images._files_by_id = {acq.file_id: acq}

    destination = tmp_path / "dataset"
    export_acq_image_list(images, destination, name="Experiment 001")
    first = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))

    assert first["name"] == "Experiment 001"
    assert len(first["images"]) == 1
    row = first["images"][0]
    assert row["shape"] == [2, 8, 10]
    assert row["dims"] == ["c", "y", "x"]
    assert row["num_channels"] == 2
    assert row["num_rois"] == 1
    assert row["analysis_types"] == ["diameter", "sum_intensity"]
    assert row["href"].endswith("/acqimage.json")

    first_id = first["id"]
    export_acq_image_list(images, destination, name="Experiment 001", overwrite=True)
    second = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
    assert second["id"] == first_id
    assert second["images"][0]["id"] == row["id"]


def test_export_rejects_existing_destination_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acq = _acq()
    monkeypatch.setattr(AcqImage, "save_as_ome_zarr", _fake_ome_zarr)
    destination = tmp_path / "image-package"
    export_acq_image(acq, destination)

    with pytest.raises(FileExistsError):
        export_acq_image(acq, destination)
