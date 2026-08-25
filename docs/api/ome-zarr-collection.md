---
search:
  exclude: true
---

# OME-Zarr Collection v1 export

::: acqstore.acq_image.io.ome_zarr_collection_v1.exporter.AcqStoreOmeZarrCollectionExporter

The collection exporter delegates each member to the focused image exporter:

::: acqstore.acq_image.io.ome_zarr_collection_v1.exporter.AcqStoreOmeZarrImageExporter

Validation uses the published Draft 2020-12 schema plus cross-document checks:

::: acqstore.acq_image.io.ome_zarr_collection_v1.validator.validate_collection
