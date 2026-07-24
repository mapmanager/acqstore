#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: scripts/make_source_zip.sh <output.zip>" >&2
    exit 2
fi

ZIP_NAME="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Resolve a relative output path against the repository root.
if [[ "$ZIP_NAME" = /* ]]; then
    ZIP_PATH="$ZIP_NAME"
else
    ZIP_PATH="$REPO_ROOT/$ZIP_NAME"
fi

# The output directory must already exist.
OUTPUT_DIR="$(dirname "$ZIP_PATH")"
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Output directory does not exist: $OUTPUT_DIR" >&2
    exit 2
fi

rm -f "$ZIP_PATH"

zip -r "$ZIP_PATH" \
    src \
    tests \
    docs \
    docs-dev \
    examples \
    scripts \
    .github/workflows \
    pyproject.toml \
    uv.lock \
    mkdocs.yml \
    README.md \
    LICENSE \
    .gitignore \
    .python-version \
    -x \
    "tests/acqstore/data/*" \
    "*/__pycache__/*" \
    "*.pyc" \
    "*.pyo" \
    "*/.pytest_cache/*" \
    "*/.mypy_cache/*" \
    "*/.ruff_cache/*" \
    "*/.coverage" \
    "*/htmlcov/*" \
    "*/.DS_Store" \
    "*/.ipynb_checkpoints/*" \
    "*/.idea/*" \
    "*/.vscode/*" \
    "site/*" \
    "build/*" \
    "dist/*" \
    ".venv/*" \
    ".git/*" \
    "tmp/*" \
    "*.egg-info/*" \
    "*.oir" \
    "*.czi" \
    "*.nd2" \
    "*.abf" \
    "*.zip" \
    "*.zarr/*"

echo "Created: $ZIP_PATH"