# AcqImage Metadata

CloudScope stores metadata with each `AcqImage` so GUI workflows and scripted workflows can use the same acquisition context.

Metadata is saved in the JSON sidecar next to the source image file.

For a source file named `my_file.tif`, CloudScope saves:

```text
my_file.tif.json
```

The JSON sidecar stores image header metadata, user-editable experimental metadata, ROIs, and analysis summaries.

## Experimental metadata

Experimental metadata is user-editable. These fields describe the biological sample, experimental condition, and notes associated with the image.

--8<-- "schemas/experimental_metadata.md"

## Image header metadata

Image header metadata is read from the source image file when available. Some calibration fields may be editable when users need to correct or provide missing physical units.

--8<-- "schemas/header_metadata.md"

## API links

- [AcqImage API](../api/acq-image.md)
- [AcqImageList API](../api/acq-image-list.md)
