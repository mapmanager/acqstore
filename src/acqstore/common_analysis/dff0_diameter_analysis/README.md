# DFF0–Diameter Analysis

Pure-Python cross-analysis between a continuous functional reporter trace and a simultaneously measured diameter trace. Reporter peak onsets are upstream seeds; this package does not redetect those seeds.

The core algorithm is domain-independent:

```python
analyze_triggered_events(time, signal, seed_indices, params)
```

The diameter use case measures what the diameter was doing before each reporter seed and what response occurred after it, bounded by the next seed or the signal end.

## Run tests

```bash
uv run pytest src/acqstore/common_analysis/dff0_diameter_analysis/tests
```

## Run the exploratory app

The NiceGUI demo lives outside the installable package:

```bash
uv run python -m examples.app.dff0_diameter_analysis.main
```

Requires NiceGUI and NiceWidgets in the environment. See
`docs/current-state.md` and `docs/decisions.md` for the durable project handoff.

## Continuous coupling

The package also provides an independent continuous lagged-correlation branch:

```python
from acqstore.common_analysis.dff0_diameter_analysis.continuous_analysis import (
    Dff0DiameterContinuousAnalysis,
)

analysis = Dff0DiameterContinuousAnalysis.from_acq_image(
    acq_image=acq_image,
    channel=0,
    roi_id=1,
)
```

Positive lag means the reporter leads and diameter follows. See
`docs/continuous-coupling.md` for definitions and limitations. The standalone
app exposes this analysis at `/continuous`.
