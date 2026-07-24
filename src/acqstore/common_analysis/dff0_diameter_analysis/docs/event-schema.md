# TriggeredEvent Schema

`TriggeredEvent` is a frozen dataclass with schema version 1 and `to_dict()` / `from_dict()` serialization.

Key groups:

- identity: seed id/index/time;
- boundaries: pre/post window, next seed, truncation flags;
- baseline: median, standard deviation, slope, pre-seed value/change;
- extremum: index/time/value and seed-to-extremum delay;
- amplitude: signed, absolute, fractional, percent;
- kinetics: average seed-to-extremum slope and maximum oriented slope;
- recovery: detected flag and recovery timing;
- integral: baseline-adjusted AUC from seed to event stop;
- quality: status and warnings.

All absent or unresolved measurements are `None`; values are never fabricated.
