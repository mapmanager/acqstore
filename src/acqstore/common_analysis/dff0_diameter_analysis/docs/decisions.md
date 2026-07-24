# Decisions

- Use generic names: `TriggeredEventParams`, `TriggeredEvent`, and `analyze_triggered_events`.
- Use enums rather than string literals for direction, filtering, and status.
- Use reporter `onset.index` values as authoritative seeds.
- Include a pre-seed window for baseline and pre-event state.
- Do not require diameter derivative onset detection in version one.
- Limit extremum search with `post_search_window_points` and hard-limit all analysis at the next seed.
- Support `none`, median, and Savitzky–Golay filtering.
- Keep package-local tests under this package.
- Keep continuous lagged correlation independent of triggered-event analysis.
- Define positive lag as reporter leading and diameter following.
- Report strongest positive, negative, and absolute correlations rather than one ambiguous maximum.
- Default continuous reporter filtering to median kernel 3.
- Default continuous diameter filtering to Savitzky-Golay window 15, polynomial order 4.
- Support optional linear detrending but defer deeper stationarity and frequency analyses.
- Reuse one shared validated dataset loader for triggered and continuous branches.
- Provide continuous coupling on a separate NiceGUI page at `/continuous`.
