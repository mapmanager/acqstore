# Continuous Coupling

The continuous branch compares the full `df_f_signal` and diameter traces without
using reporter peak events or triggered-event measurements.

## Lag convention

Positive lag means the reporter changes first and diameter follows later.
Negative lag means diameter changes first and the reporter follows later.

Because increased reporter activity can accompany decreased diameter, the most
biologically relevant relationship may be a negative correlation at a positive
lag. Results therefore retain the strongest positive, strongest negative, and
strongest absolute correlations and their lags.

## Preprocessing

The first implementation defaults to:

- reporter: median filter, kernel 3;
- diameter: Savitzky-Golay filter, window 15, polynomial order 4;
- optional linear detrending of both signals.

These settings are local to continuous coupling and do not modify upstream
Sum Intensity or diameter results. The standalone app exposes the filter settings
for runtime exploration.

## Calculation

For each integer lag, Pearson correlation is calculated over the overlapping
finite samples. Each lag stores its overlap count. Lags with fewer than the
configured minimum number of paired samples are unresolved.

The global result includes:

- correlation versus lag;
- zero-lag correlation;
- strongest positive correlation and lag;
- strongest negative correlation and lag;
- strongest absolute correlation and lag.

## Plots

The continuous page contains:

1. filtered reporter and diameter traces overlaid on separate y-axes;
2. Pearson correlation versus lag;
3. a standardized overlay after shifting diameter by the strongest absolute lag.

A single global lag is an exploratory summary, not proof of stationarity or
causality. Fixed-window stationarity analysis is intentionally deferred until
results from the initial recordings have been inspected.
