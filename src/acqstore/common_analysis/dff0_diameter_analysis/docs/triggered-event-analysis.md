# Triggered Event Analysis

For each supplied seed index, the algorithm collects a pre-seed window and a post-seed event window.

The pre-window supports baseline, baseline variability, baseline slope, and immediate pre-seed state. The post-window supports extremum, seed-to-extremum delay, amplitude, slope, recovery, and baseline-adjusted area.

The event stop is the earliest of:

1. configured post window,
2. next seed index,
3. end of signal.

The extremum search has an additional `post_search_window_points` limit. This prevents a late unrelated extremum from being assigned to the seed while allowing a longer window for recovery and area.

No response-onset detector is required in version one. A derivative-based onset detector may be proposed later, but requires explicit approval before implementation.
