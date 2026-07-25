# Notebook Workflows

AcqStore includes notebook examples for users who want to load data, inspect
arrays, run analysis, and view results from Python.

The notebooks are rendered in the documentation site and are not currently
executed in CI.

## Available notebooks

- [Load and Plot Image](notebooks/load-and-plot-image.ipynb)
- [Velocity Analysis](notebooks/velocity-analysis.ipynb)
- [Diameter Analysis](notebooks/diameter-analysis.ipynb)
- [Sum Intensity Analysis](notebooks/sum-intensity-analysis.ipynb): functional reporter fluorescence (like GCaMP) from normalized line intensity
- [Kymograph Reference Images](notebooks/kymograph-reference-image.ipynb)
- [Heart Rate Analysis](notebooks/heart-rate-analysis.ipynb)
- [Heart Rate Batch Analysis](notebooks/heart-rate-batch-analysis.ipynb)
- [Generating a Randomized File Subset](notebooks/generating-randomized-file-for-analysis.ipynb): sample a large dataset into an unbiased per-condition subset

## Plotting

AcqStore runtime code does not depend on Matplotlib. Matplotlib is included only
in the documentation dependency group for examples and notebooks.

Notebook examples may use Matplotlib for simple inline plots and Plotly where
interactive visualization is useful.
