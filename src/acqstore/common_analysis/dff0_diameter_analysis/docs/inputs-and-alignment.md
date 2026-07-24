# Inputs and Alignment

Sidecar development uses:

- `*.diameter.csv`
- `*.sum_intensity.csv`
- `*.json`

The JSON supplies reporter events, including authoritative `onset.index` values. Stored onset times are checked against the reporter table. Diameter `center_row`, reporter `time_index`, and both time columns must align exactly within tolerance.

`from_acq_image()` uses public `analysis_set.get_analysis(...)`, `result.table`, and `get_peak_events()` APIs. Both analyses are mandatory for the selected channel and ROI.
