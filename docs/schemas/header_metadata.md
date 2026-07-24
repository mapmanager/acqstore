# Image Header Metadata

| name             | display_name     | type   | default   | unit   | choices   | editable   | group       | description   |
|:-----------------|:-----------------|:-------|:----------|:-------|:----------|:-----------|:------------|:--------------|
| shape            | Shape            | str    |           |        |           | False      | Header      |               |
| dims             | Dims             | str    |           |        |           | False      | Header      |               |
| sizes            | Sizes            | str    |           |        |           | False      | Header      |               |
| dtype            | DType            | str    |           |        |           | False      | Header      |               |
| num_channels     | Channels         | int    | 0         |        |           | False      | Header      |               |
| num_scenes       | Scenes           | int    | 0         |        |           | False      | Header      |               |
| date             | Date             | str    |           |        |           | False      | Header      |               |
| time             | Time             | str    |           |        |           | False      | Header      |               |
| physical_unit_y  | Physical Unit Y  | float  | 1.0       |        |           | True       | Calibration |               |
| physical_unit_x  | Physical Unit X  | float  | 1.0       |        |           | True       | Calibration |               |
| physical_label_y | Physical Label Y | str    | Pixels    |        |           | True       | Calibration |               |
| physical_label_x | Physical Label X | str    | Pixels    |        |           | True       | Calibration |               |
