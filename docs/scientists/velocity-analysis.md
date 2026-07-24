# Velocity Analysis

Velocity analysis estimates **blood flow velocity from line scan kymographs** using a Radon-transform-based method.

Later documentation may refer to this workflow simply as velocity analysis, but the implemented analysis is specifically designed for line scan kymograph data.

## Input data

Velocity analysis expects image data organized as a line scan kymograph. The ROI should cover the region of the kymograph used to estimate flow velocity.

## Detection parameters

Detection parameters define the scientific behavior of the analysis. For velocity analysis, the primary parameter controls the width of each Radon analysis window.

--8<-- "schemas/velocity_detection_parameters.md"

## Results

Velocity analysis stores summary values in the `AcqImage` JSON sidecar and writes tabular output to a CSV file.

For a source file named `my_file.tif`, velocity analysis saves:

```text
my_file.tif.json
my_file.tif.radon_velocity.csv
```

The JSON sidecar includes the detection parameters and summary values for each analyzed ROI. Typical summary values include:

- `velocity_mean`
- `velocity_median`
- `velocity_cv`
- `num_windows`

The CSV file stores per-window tabular velocity results.

## Programmatic use

Velocity analysis can be run from the GUI or from Python code using the same `acqstore` backend.

See the [Velocity Analysis API](../api/velocity-analysis.md).
