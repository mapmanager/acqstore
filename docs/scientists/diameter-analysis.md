# Diameter Analysis

Diameter analysis estimates **vessel diameter from line scan kymographs**.

The analysis operates on intensity profiles derived from a selected ROI and produces diameter summaries, per-row or per-window results, and quality-control values.

## Input data

Diameter analysis expects image data organized as a line scan kymograph. The ROI should cover the spatial region used to estimate vessel diameter.

## Detection parameters

Detection parameters define the scientific behavior of the analysis. Diameter analysis parameters control profile aggregation, polarity, thresholding, gradient-based edge detection, motion gating, and post-filtering.

--8<-- "schemas/diameter_detection_parameters.md"

## Results

Diameter analysis stores summary values in the `AcqImage` JSON sidecar and writes tabular output to a CSV file.

For a source file named `my_file.tif`, diameter analysis saves:

```text
my_file.tif.json
my_file.tif.diameter.csv
```

The JSON sidecar includes the detection parameters and summary values for each analyzed ROI. Typical summary values include:

- `diameter_um_mean`
- `diameter_um_median`
- `diameter_um_cv`
- `num_rows`
- `qc_score_mean`
- quality-control violation counts

The CSV file stores tabular diameter results.

## Programmatic use

Diameter analysis can be run from the GUI or from Python code using the same `acqstore` backend.

See the [Diameter Analysis API](../api/diameter-analysis.md).
