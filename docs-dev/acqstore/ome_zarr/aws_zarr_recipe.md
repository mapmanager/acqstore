# AWS S3 recipe for acqstore OME-Zarr stores

This developer note describes the expected AWS setup for testing acqstore local
and remote Zarr workflows. The CloudScope GUI should call acqstore public APIs;
AWS-specific read/write behavior belongs in `src/acqstore/`.

## Install dependencies

From the repository root:

```bash
uv add bioio bioio-ome-zarr s3fs
```

`bioio-ome-zarr` provides the OME-Zarr writer used by acqstore. `s3fs` provides
fsspec-backed `s3://` file access. `boto3` is useful for AWS credential/session
workflows and AWS scripting even when the primary image IO path goes through
fsspec.

## Install AWS CLI v2 on macOS

Install the official AWS CLI v2 macOS package from AWS, then verify:

```bash
aws --version
```

## Configure credentials

For development, create an IAM user or role with access to the test bucket. Then
configure your local profile:

```bash
aws configure
```

You will be prompted for:

```text
AWS Access Key ID
AWS Secret Access Key
Default region name, e.g. us-east-1
Default output format, e.g. json
```

This creates files under `~/.aws/` that are used by `aws`, `boto3`, and `s3fs`.

## Create a test bucket

Use a globally unique bucket name:

```bash
aws s3 mb s3://cloudscope-dev-data-yourname --region us-east-1
```

Keep the bucket private for development.

## Upload a local directory store

Zarr directory stores contain many small files. Use `sync`, not a single-file
copy:

```bash
aws s3 sync \
  /path/to/sample.cs.ome.zarr \
  s3://cloudscope-dev-data-yourname/sample.cs.ome.zarr
```

## Download a directory store

```bash
aws s3 sync \
  s3://cloudscope-dev-data-yourname/sample.cs.ome.zarr \
  /tmp/sample.cs.ome.zarr
```

## Inspect uploaded contents

```bash
aws s3 ls \
  s3://cloudscope-dev-data-yourname/sample.cs.ome.zarr \
  --recursive \
  --summarize
```

## Direct acqstore API target

The desired backend API is:

```python
from acqstore.acq_image.acq_image import AcqImage

acq = AcqImage('/path/to/source.tif')
acq.save_as_ome_zarr('s3://cloudscope-dev-data-yourname/source.ome.zarr')
acq.save_native_zarr('s3://cloudscope-dev-data-yourname/source.cs.ome.zarr')

reloaded = AcqImage('s3://cloudscope-dev-data-yourname/source.cs.ome.zarr')
```

For early debugging, first test local save/load, then AWS CLI sync upload/download,
then direct `s3://` save/load.
