#!/usr/bin/env bash
# Smaller source zip: omit docs/, docs-dev/, scripts/, and mkdocs.yml.
# Always writes to zips/acqstore-YYYYMMDD-vN.zip (N increments for today's date).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ZIPS_DIR="$REPO_ROOT/zips"
mkdir -p "$ZIPS_DIR"

TODAY="$(date +%Y%m%d)"
PREFIX="acqstore-${TODAY}-v"

NEXT_N=1
shopt -s nullglob
for existing in "$ZIPS_DIR"/${PREFIX}*.zip; do
    base="$(basename "$existing" .zip)"
    suffix="${base#"$PREFIX"}"
    if [[ "$suffix" =~ ^[0-9]+$ ]]; then
        n=$((10#$suffix))
        if (( n >= NEXT_N )); then
            NEXT_N=$((n + 1))
        fi
    fi
done
shopt -u nullglob

ZIP_PATH="$ZIPS_DIR/${PREFIX}${NEXT_N}.zip"
rm -f "$ZIP_PATH"

zip -r "$ZIP_PATH" \
    src \
    tests \
    examples \
    .github/workflows \
    pyproject.toml \
    uv.lock \
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
ls -lh "$ZIP_PATH"
