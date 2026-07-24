# Heart Rate Analysis

Heart rate analysis estimates a periodic **heartbeat frequency** from a velocity
time-series and reports it in **beats-per-minute (bpm)** and **Hz**
(`bpm = 60 × Hz`).

## Input data

Heart rate analysis is a **dependent analysis**. It is seeded by a
`radon_velocity` analysis on the *same* `(channel, roi_id)` and reads the parent
velocity series through the generic `get_plot_data()` API, so it never depends on
Radon-specific table columns. Run velocity analysis first (see
[Velocity Analysis](velocity-analysis.md)).

## Two estimators and quality control

Every run computes the dominant periodicity with **two** independent estimators:

- **Lomb-Scargle** periodogram
- **Welch** power spectral density

Running both in a single pass enables an automatic agreement check. When the two
estimates fall within `agree_tol_bpm` of each other the summary `status` is
`ok` (**accept**); when they diverge the status becomes `method_disagree`
(**reject / review**). A recording with too few valid velocity samples reports
`insufficient_valid`.

## Detection parameters

--8<-- "schemas/heart_rate_detection_parameters.md"

## Results

Heart rate stores a compact summary in the `AcqImage` JSON sidecar. There is
**no CSV table** for heart rate.

For a source file named `my_file.tif`, heart rate results are written to:

```text
my_file.tif.json
```

The summary includes the per-estimator results (`lomb`, `welch`), the rollup
`status`, and an `agreement` block (`abs_delta_bpm`, `agree_ok`).

## Programmatic use

Heart rate analysis is available from Python through the `acqstore` backend;
there is no dedicated GUI panel yet. See the worked examples in the
[Heart Rate Analysis notebook](../notebooks/heart-rate-analysis.ipynb) and the
[Heart Rate Batch Analysis notebook](../notebooks/heart-rate-batch-analysis.ipynb),
and the [Heart Rate Analysis API](../api/heart-rate-analysis.md).
