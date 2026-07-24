# Standalone App

Run from the AcqStore repository root (requires NiceGUI and NiceWidgets):

```bash
uv run python -m examples.app.dff0_diameter_analysis.main
```

The app uses the hard-coded `SOURCE_PATH` in
`examples/app/dff0_diameter_analysis/main.py` and discovers every
file/channel/ROI selection containing both Sum Intensity and diameter analyses.

Pages:

- `/` — triggered-event analysis;
- `/continuous` — continuous lagged-correlation analysis.

Both pages reuse the same analysis-hit discovery. Selecting a grid row updates
page state; analysis runs only when its Plot button is clicked.

The continuous page exposes reporter and diameter filters, lag range, minimum
overlap, and optional linear detrending. It rebuilds Plotly figures rather than
performing incremental figure updates.
