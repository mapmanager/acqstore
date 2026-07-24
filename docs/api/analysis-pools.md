---
search:
  exclude: true
---

# Analysis pools

Analysis pools are backend data-model objects that collect per-file analysis
summaries into a single flat pandas DataFrame, one row per acquisition
image/channel/ROI selection. CloudScope consumes the same DataFrame at runtime
for its pool plots.

::: acqstore.analysis_pool.base_analysis_pool.AnalysisPool

::: acqstore.analysis_pool.velocity_analysis_pool.VelocityAnalysisPool

::: acqstore.analysis_pool.sum_intensity_analysis_pool.SumIntensityAnalysisPool
