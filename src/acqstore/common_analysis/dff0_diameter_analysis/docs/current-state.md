# Current State

Implemented triggered-event analysis:

- generic seeded-event engine;
- pre-seed baseline and trend measurements;
- next-seed truncation;
- independent post-seed extremum search limit;
- extremum, amplitude, timing, slope, recovery, and AUC;
- enums and serialization;
- sidecar and AcqImage adapters;
- Plotly overview, event, and metric-versus-time plots.

Implemented continuous coupling analysis:

- independent full-trace analysis that does not require triggered events;
- normalized Pearson correlation at integer lags;
- positive-lag convention: reporter leads, diameter follows;
- strongest positive, negative, and absolute lag summaries;
- overlap count and minimum-overlap validation;
- median reporter filtering and Savitzky-Golay diameter filtering;
- optional linear detrending;
- schema serialization and DataFrame output;
- continuous Plotly figures and a separate NiceGUI page at `/continuous`;
- shared AcqImage/sidecar dataset loading for both branches.

Development examples:

- `220110n_0005.tif`, channel 0, ROI 1;
- `220110n_0020.tif`, channel 0, ROI 1;
- `220110n_0022.tif`, channel 0, ROI 1;
- `220110n_0023.tif`, channel 0, ROI 1.

Next scientific work should inspect global lag curves across these recordings
before adding fixed-window stationarity analysis, derivative coupling, or more
advanced preprocessing.
