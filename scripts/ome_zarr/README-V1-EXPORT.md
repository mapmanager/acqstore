# 20260921

this is always confusing.

Using this template i will save 2-3 sample ome zarr v1 folders:

```
uv run python scripts/ome_zarr/try_ome_zarr_collection_v1_export.py \
  /absolute/path/to/source \
  /Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/collection-name-v1.ome.zarr \
  --name "Collection display name"
```

then we verify the output using this template:

```
uv run python scripts/ome_zarr/validate_ome_zarr_collection_v1.py \
  /Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/collection-name-v1.ome.zarr
```

## diameter sample data

```
uv run python scripts/ome_zarr/try_ome_zarr_collection_v1_export.py \
  /Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/diameter-sample-data \
  /Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/diameter-sample-data.ome.zarr \
  --name "Dimaeter Sample Data"
```

that outputs this "/Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/diameter-sample-data.ome.zarr"

then verify with


```
uv run python scripts/ome_zarr/validate_ome_zarr_collection_v1.py \
  /Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/diameter-sample-data.ome.zarr
```

that outputs: "VALID: AcqStore OME-Zarr Collection v1 additive resources
NOTE: validate each primary/reference image independently as OME-NGFF."

## velocity sample data

```
uv run python scripts/ome_zarr/try_ome_zarr_collection_v1_export.py \
  /Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/velocity-sample-data \
  /Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/velocity-sample-data.ome.zarr \
  --name "Velocity Sample Data"
```

then verify with


```
uv run python scripts/ome_zarr/validate_ome_zarr_collection_v1.py \
  /Users/cudmore/Sites/cs_project/cloudscope-data/ome-zarr-output/v2/velocity-sample-data.ome.zarr
```
