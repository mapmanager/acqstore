# Experiment Metadata

| name         | display_name   | type   |   default | unit   | choices   | editable   | group      | description                                 |
|:-------------|:---------------|:-------|----------:|:-------|:----------|:-----------|:-----------|:--------------------------------------------|
| species      | Species        | str    |           |        |           | True       | Animal     | Animal species (e.g., mouse, rat).          |
| sex          | Sex            | str    |           |        |           | True       | Animal     | Biological sex or experimental sex label.   |
| genotype     | Genotype       | str    |           |        |           | True       | Animal     | Genotype or strain label.                   |
| age          | Age            | str    |           |        |           | True       | Animal     | Animal age label (e.g., P30, 8 weeks).      |
| region       | Region         | str    |           |        |           | True       | Sample     | Brain region or anatomical location.        |
| cell_type    | Cell type      | str    |           |        |           | True       | Sample     | Type of cell or vessel being imaged.        |
| depth        | Depth          | float  |       nan |        |           | True       | Sample     | Imaging depth in micrometers.               |
| branch_order | Branch order   | int    |       nan |        |           | True       | Sample     | Branch order for vascular structures.       |
| direction    | Direction      | str    |           |        |           | True       | Sample     | Flow direction or vessel orientation.       |
| condition    | Condition      | str    |           |        |           | True       | Experiment | Experimental condition or treatment.        |
| condition2   | Condition 2    | str    |           |        |           | True       | Experiment | Second condition field.                     |
| treatment    | Treatment      | str    |           |        |           | True       | Experiment | Treatment applied.                          |
| treatment2   | Treatment 2    | str    |           |        |           | True       | Experiment | Second treatment field.                     |
| date         | Date           | str    |           |        |           | True       | Experiment | User-editable date (e.g., experiment date). |
| note         | Note           | str    |           |        |           | True       | Notes      | Free-form notes or comments.                |
