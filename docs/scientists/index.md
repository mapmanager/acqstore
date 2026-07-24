# Data Scientist Guide

CloudScope is organized around a small scientific data model that supports both GUI workflows and Python scripting.

The main scientific objects are:

- `AcqImage`: one acquisition-backed image dataset
- `AcqImageList`: a collection of `AcqImage` objects
- ROI: the region where analysis is performed
- Analysis: a quantitative workflow applied to an image, channel, and ROI

Current quantitative analysis workflows are designed for **line scan kymographs** and include:

- [blood flow velocity analysis](velocity-analysis.md) from line scan kymographs using a Radon-transform-based method
- [vessel diameter analysis](diameter-analysis.md) from line scan kymographs
- [sum intensity analysis](sum-intensity-analysis.md) from line scan kymographs for functional reporter fluorescence (like GCaMP)

## Why CloudScope?

CloudScope exists to keep image loading, visualization, metadata, ROIs, and analysis connected through one shared backend.

The same `acqstore` code is used by:

- the desktop GUI
- the browser GUI
- Python scripts
- Jupyter notebooks
- unit tests

This makes it possible to reproduce GUI analysis results programmatically using the same underlying APIs.

## Core concepts

### AcqImage

An `AcqImage` represents one acquisition-backed image dataset.

An `AcqImage` combines:

- image pixels
- image header metadata
- experimental metadata
- ROIs
- analysis summaries and results

See the [AcqImage API](../api/acq-image.md).

### AcqImageList

An `AcqImageList` manages a collection of `AcqImage` objects.

Typical uses include:

- loading a folder of acquisition files
- managing multiple images in one workflow
- running batch analysis
- building tables from file-level metadata

See the [AcqImageList API](../api/acq-image-list.md).

### ROIs

ROIs define the region of image data used for analysis. Analysis results are associated with an image, channel, and ROI.

### Analysis

Analysis modules take detection parameters as input and produce summary values and optional tabular outputs. Detection parameters define scientific behavior; execution options control runtime behavior such as progress, cancellation, and parallel processing.

## Line scan kymographs

The current quantitative analysis workflows operate on line scan kymographs rather than general two-dimensional image fields.

A line scan kymograph represents repeated sampling along a spatial line over time. CloudScope uses this structure for blood flow velocity, vessel diameter, and functional reporter fluorescence measurements.

## Current analysis workflows

### Blood flow velocity

Velocity analysis estimates blood flow velocity from line scan kymographs using a Radon-transform-based method.

Read more in [Velocity Analysis](velocity-analysis.md).

### Vessel diameter

Diameter analysis estimates vessel diameter from line scan kymographs using intensity-profile-based measurements.

Read more in [Diameter Analysis](diameter-analysis.md).

### Functional reporter fluorescence

Sum intensity analysis measures normalized line intensity along a kymograph ROI and detects
transient peaks from a functional reporter (like GCaMP).

Read more in [Sum Intensity Analysis](sum-intensity-analysis.md).

## Saved analysis files

For each acquisition file, CloudScope stores analysis state and results as sidecar files next to the source file.

For a source file named `my_file.tif`, CloudScope may save:

```text
my_file.tif
my_file.tif.json
my_file.tif.radon_velocity.csv
my_file.tif.diameter.csv
my_file.tif.sum_intensity.csv
```

The JSON sidecar stores:

- accepted/rejected state
- experimental metadata
- image header metadata
- ROIs
- detection parameters used for each analysis
- analysis summaries

The CSV files store tabular analysis outputs for analysis types that provide CSV export.

When CloudScope reloads an `AcqImage`, the JSON sidecar is used to restore relevant metadata, ROIs, and saved analysis state.

## Metadata

CloudScope separates image header metadata from user-editable experimental metadata.

Image header metadata is read from the source file when available. Experimental metadata is edited by the user and saved with the `AcqImage` sidecar.

See [AcqImage Metadata](acqimage-metadata.md).

## Current limitations

CloudScope can load and visualize supported image formats, but the currently implemented quantitative analysis workflows are designed for line scan kymographs.

Traditional two-dimensional segmentation, tracking, and image-analysis workflows are not yet implemented as dedicated CloudScope analysis modules.

## Where to go next

- [Velocity Analysis](velocity-analysis.md)
- [Diameter Analysis](diameter-analysis.md)
- [Sum Intensity Analysis](sum-intensity-analysis.md)
- [AcqImage Metadata](acqimage-metadata.md)
- [Notebook Workflows](notebooks.md)
- [API Reference](../api/index.md)
