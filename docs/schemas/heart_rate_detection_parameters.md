# Heart Rate Detection Parameters

| name               | display_name          | type   |   default | choices   | unit   | editable   | visible   | methods   | description                                                 |
|:-------------------|:----------------------|:-------|----------:|:----------|:-------|:-----------|:----------|:----------|:------------------------------------------------------------|
| bpm_min            | Heart rate min        | float  |    240    |           | bpm    | True       | True      |           | Lower heart-rate bound of the analysis band.                |
| bpm_max            | Heart rate max        | float  |    600    |           | bpm    | True       | True      |           | Upper heart-rate bound of the analysis band.                |
| use_abs            | Use absolute velocity | bool   |      1    |           | nan    | True       | True      |           | Analyze absolute velocity instead of signed velocity.       |
| outlier_k_mad      | Outlier clip (MAD)    | float  |      4    |           | nan    | True       | True      |           | MAD winsorization factor applied during preprocessing.      |
| lomb_n_freq        | Lomb frequencies      | int    |    512    |           | nan    | True       | True      |           | Number of frequencies in the Lomb-Scargle grid.             |
| interp_max_gap_sec | Max interp gap        | float  |      0.05 |           | s      | True       | True      |           | Maximum NaN gap interpolated for the Welch path.            |
| bandpass_order     | Bandpass order        | int    |      3    |           | nan    | True       | True      |           | Butterworth band-pass order for the Welch path.             |
| nperseg_sec        | Welch segment         | float  |      2    |           | s      | True       | True      |           | Welch PSD segment duration.                                 |
| edge_margin_hz     | Edge margin           | float  |     -1    |           | Hz     | True       | True      |           | Edge margin in Hz for edge flagging. Use -1.0 for auto.     |
| peak_half_width_hz | Peak half width       | float  |      0.5  |           | Hz     | True       | True      |           | Half-width around the peak used for band concentration.     |
| agree_tol_bpm      | Agreement tolerance   | float  |     30    |           | bpm    | True       | True      |           | Maximum Lomb-vs-Welch bpm delta considered agreement.       |
| do_segments        | Compute segments      | bool   |      0    |           | nan    | True       | True      |           | Compute a compact windowed segment summary.                 |
| seg_win_sec        | Segment window        | float  |      6    |           | s      | True       | True      |           | Segment window length when segments are computed.           |
| seg_step_sec       | Segment step          | float  |      1    |           | s      | True       | True      |           | Segment window step when segments are computed.             |
| seg_min_valid_frac | Segment min valid     | float  |      0.5  |           | nan    | True       | True      |           | Minimum finite-sample fraction required per segment window. |
