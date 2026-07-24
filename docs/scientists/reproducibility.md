# Reproducibility

AcqStore is designed so GUI applications, scripts, and notebooks can share the same scientific backend for loading files, managing ROIs, running analysis, and saving results.

This is important for scientific reproducibility because analysis behavior should not depend on whether a user clicks through a GUI or calls the API from Python.

## Versioned releases

Official AcqStore releases are tied to git tags and published on the [AcqStore Releases](https://github.com/mapmanager/acqstore/releases){target="_blank" rel="noopener"} page.

A published release provides a reproducible reference point for scientific analysis while allowing active development to continue.

Release artifacts can include:

- source code archives
- documentation builds
- version metadata

## Sidecar files

AcqStore saves analysis state and results next to the source image file.

For a source file named `my_file.tif`, saved files may include:

```text
my_file.tif.json
my_file.tif.radon_velocity.csv
my_file.tif.diameter.csv
my_file.tif.sum_intensity.csv
```

The JSON sidecar records the ROIs, detection parameters, and analysis summaries used to generate results. These files should be retained with the source data when preserving an analysis.

## Same backend across interfaces

The same `acqstore` APIs are used from:

- GUI applications (for example CloudScope)
- Python scripts
- Jupyter notebooks

This reduces divergence between interactive and scripted workflows.
