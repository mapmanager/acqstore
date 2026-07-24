# AcqStore Phase 1 Migration Report

Final repository polish completed per
`tmp/handoff-acqstore-repository-final-polish-v2.md`.

Canonical report filename: **`migration-report-acqstore.md`** (there is no
`migration-report.md`).

## 1. Executive summary

`cs_project/acqstore/` is a publication-ready standalone Python package
(`acqstore` **0.1.0**) extracted from the CloudScope monolith.

- Public APIs and scientific behavior under `src/acqstore/` are preserved.
- Runtime package has **no** imports of `cloudscope`, `nicewidgets`,
  `acqstore_server`, or `nicegui`.
- Demo remains at `examples/app/dff0_diameter_analysis/` and is excluded from
  dependencies, locking, tests, CI, docs validation, and package builds.
- Developer scripts live flat under `scripts/` (no `scripts/acqstore/`).
- `tmp/` remains a local ignored workspace and is excluded from source ZIPs.
- Final validation: **482 passed**, **48 skipped** (optional fixtures),
  **0 failed**; MkDocs `--strict` OK; sdist + wheel OK.
- Verified source archive: **`acqstore_20260724_v3.zip`**.
- Git was **not** initialized; `cloudscope/` and `cloudscope-data/` were not
  modified.

## 2. Final repository tree

```text
acqstore/
├── .github/workflows/
│   ├── tests.yml
│   └── docs.yml
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── migration-report-acqstore.md
├── mkdocs.yml
├── pyproject.toml
├── uv.lock
├── docs/
├── docs-dev/
├── examples/app/dff0_diameter_analysis/
├── scripts/                 # flattened try_* diagnostics
│   ├── ome_zarr/
│   ├── dev/
│   └── make_source_zip.sh
├── src/acqstore/
├── tests/__init__.py
├── tests/acqstore/          # unchanged layout (mirrors package)
└── tmp/                     # local ignored workspace (not published)
```

Local ignored artifacts may remain on disk: `.venv/`, `site/`, `dist/`,
`.cache/`, `.pytest_cache/`, `*.zip`.

## 3. Final polish actions (this pass)

| Action | Result |
|---|---|
| Fix `scripts/make_source_zip.sh` | Includes `migration-report-acqstore.md`; removed obsolete `migration-report.md` |
| Flatten `scripts/acqstore/` | Contents moved to `scripts/` and `scripts/ome_zarr/`; empty dir removed |
| Update path references | Docs, package notes, and script docstrings updated |
| Finalize `.gitignore` | Confirmed required ignores, including `tmp/` |
| README review | Accurate; no stale monolith / `scripts/acqstore/` paths |
| Notebook sample-data fix | Replaced undefined constants with catalog IDs (see §8) |
| Source ZIP | Generated and inspected `acqstore_20260724_v3.zip` |

## 4. Source ZIP script and verified archive

### Script correction

`scripts/make_source_zip.sh` now archives:

```text
migration-report-acqstore.md
```

It does not archive `migration-report.md`. `tmp/` is not in the include list and
is absent from the archive.

### Verified archive

```bash
bash scripts/make_source_zip.sh acqstore_20260724_v3.zip
```

| Check | Result |
|---|---|
| `migration-report-acqstore.md` present | **Yes** |
| `migration-report.md` present | **No** |
| Flattened `scripts/*.py` present | **Yes** |
| `scripts/ome_zarr/try_ome_zarr.py` present | **Yes** |
| `scripts/acqstore/` present | **No** |
| `tmp/`, `.venv/`, `site/`, `dist/`, `.git/` present | **No** |
| Required root/config/docs/tests/examples/workflows present | **Yes** (spot-checked) |
| Entry count | 329 |

## 5. Flattened script layout

Moved from `scripts/acqstore/` to `scripts/` (including `ome_zarr/`).

Kept:

- `scripts/dev/`
- `scripts/make_source_zip.sh`

Updated references in:

- `docs/developers/index.md`
- `src/acqstore/acq_trace/readme-acq-trace.md`
- script module docstrings / run instructions

`tests/acqstore/` was **not** flattened.

## 6. `.gitignore` policy

Confirmed ignores include at least:

```text
.venv/
.cache/
.pytest_cache/
.ruff_cache/
site/
dist/
build/
*.egg-info/
__pycache__/
*.py[cod]
.DS_Store
tmp/
```

Also ignores local archives (`*.zip`), coverage artifacts, IDE dirs, and NiceGUI
storage. Does **not** ignore source, docs, tests, examples, workflows, `uv.lock`,
or `migration-report-acqstore.md`.

`tmp/` remains available locally for handoffs and notes.

## 7. Root audit

Intentional publication roots:

```text
.github/
docs/
docs-dev/
examples/
scripts/
src/
tests/
.gitignore
.python-version
LICENSE
README.md
mkdocs.yml
pyproject.toml
uv.lock
migration-report-acqstore.md
```

Local ignored content may remain: `tmp/`, `.venv/`, `site/`, `dist/`, caches,
and local ZIP snapshots.

## 8. README and documentation review

### README

Confirmed concise and accurate: package description, `uv` install, usage,
docs/test commands, demo disclaimer, GPL-3.0-only, future GitHub URL
`https://github.com/mapmanager/acqstore`.

### Stale-reference sweep

Corrected repository-internal stale paths. Retained accurate CloudScope-as-
consumer prose and `cloudscope-data` sample-data contract references.

### Notebooks

Seven notebooks imported undefined `VELOCITY_SAMPLE_DATA` /
`DIAMETER_SAMPLE_DATA`. Mapping is unambiguous from:

- `cloudscope-data/catalog.json` ids `velocity-sample-data` and
  `diameter-sample-data`
- notebook prose already naming those catalog ids
- current API `ensure_sample(name: str)`

**Fix applied (notebooks only; no `src/acqstore/` API change):**

| Constant (removed) | Replacement |
|---|---|
| `VELOCITY_SAMPLE_DATA` | `ensure_sample('velocity-sample-data')` |
| `DIAMETER_SAMPLE_DATA` | `ensure_sample('diameter-sample-data')` |

Notebooks updated:

- `docs/notebooks/heart-rate-batch-analysis.ipynb`
- `docs/notebooks/generating-randomized-file-for-analysis.ipynb`
- `docs/notebooks/velocity-analysis.ipynb`
- `docs/notebooks/heart-rate-analysis.ipynb`
- `docs/notebooks/load-and-plot-image.ipynb`
- `docs/notebooks/diameter-analysis.ipynb`
- `docs/notebooks/sum-intensity-analysis.ipynb`

Notebooks were not executed during MkDocs validation (`execute: false`).

## 9. Extraction history (summary)

| Source (cloudscope) | Destination | Notes |
|---|---|---|
| `src/acqstore/**` | `src/acqstore/**` | Demo `app/` removed from package |
| Demo app | `examples/app/dff0_diameter_analysis/` | Manual NiceGUI example |
| `tests/acqstore/**` | `tests/acqstore/**` | All 71 tracked files |
| `tests/__init__.py` | `tests/__init__.py` | Added in prior finalization pass |
| Docs API/scientists/notebooks/schemas | `docs/` | No CloudScope GUI `users/` docs |
| AcqStore `docs-dev` notes | `docs-dev/` | |
| AcqStore try_* + batch scripts | `scripts/` (flattened this pass) | |
| `LICENSE`, `.python-version` | root | |

### Modifications under `src/acqstore/` (docs-only)

| File | Reason |
|---|---|
| Removed `.../dff0_diameter_analysis/app/` | Demo moved to examples |
| `.../dff0_diameter_analysis/README.md`, `docs/app.md` | Demo path updates |
| `src/acqstore/README.md` | Broken monolith doc-link fixes |
| `src/acqstore/acq_trace/readme-acq-trace.md` | Script path update after flatten |

**No Python API/behavior changes under `src/acqstore/`.**

## 10. Cross-boundary and CI

### Runtime boundary (`src/acqstore/`)

| Import | Result |
|---|---|
| `cloudscope` | None |
| `nicewidgets` | None |
| `nicegui` | None |
| `acqstore_server` | None |

Soft couplings retained: `CLOUDSCOPE_SAMPLE_DATA_DIR`, `CLOUDSCOPE_UPLOAD_DIR`,
catalog URL `mapmanager/cloudscope-data`.

### CI

- `tests.yml`: `uv sync --frozen --group dev`; pytest with AcqStore coverage.
- `docs.yml`: `uv sync --frozen --group docs`; `mkdocs build --strict`; Pages
  deploy.
- No demo/NiceGUI/NiceWidgets install; no release-publishing redesign.

## 11. Final validation

Exact commands:

```bash
uv lock --check
uv sync --frozen
uv sync --frozen --group dev --group docs
uv run --no-sync pytest -ra
DISABLE_MKDOCS_2_WARNING=true uv run --no-sync mkdocs build --strict
uv build
```

### Tests

| Outcome | Count |
|---|---|
| Passed | **482** |
| Failed | **0** |
| Skipped | **48** |

Skip categories (expanded `SKIPPED [N]`; sum = 48):

| Count | Reason |
|---|---|
| 29 | ABF fixture missing (`tests/acqstore/data/abf`) |
| 11 | kymograph OIR fixture missing |
| 4 | specific OIR unique-metadata fixture missing |
| 2 | oir-debug 0010 missing |
| 1 | Z-stack OIR fixture missing |
| 1 | DFF0 uploaded example sidecars unavailable |

### Docs

MkDocs `--strict` succeeded.

### Package

Artifacts: `dist/acqstore-0.1.0.tar.gz`, `dist/acqstore-0.1.0-py3-none-any.whl`.

Wheel top-level: `acqstore/`, `acqstore-0.1.0.dist-info/` only. Excludes
repository-level `examples/`, `tests/`, `docs/`, `scripts/`, `tmp/`, and local
data. Package-internal colocated `dff0_diameter_analysis/{docs,tests}/` remain
inside the wheel (same as monolith source layout).

## 12. Sibling repository integrity

```bash
git -C ../cloudscope status --short
# (empty)

git -C ../cloudscope-data status --short
# D dist/demo-small-v1.zip
# ?? data-samples/
# ?? data/
```

`cloudscope/` unchanged. `cloudscope-data/` shows **pre-existing** local
dirtiness only; not modified by this work.

## 13. Demo status

| Item | Status |
|---|---|
| Path | `examples/app/dff0_diameter_analysis/` |
| In package / deps / lock / CI / validation | **No** |
| Manual validation | Deferred |

## 14. Remaining deferred items

**None that block Phase 1 publication.**

Optional follow-ups (not blockers):

1. Manual demo run in an environment with NiceGUI + NiceWidgets.
2. Env-var rename away from `CLOUDSCOPE_*` (compatibility decision).
3. Optional large fixtures for loader tests (currently clean skipifs).
4. Git init / GitHub publication (explicitly out of this handoff).

## 15. NiceWidgets extraction notes

1. Keep demo NiceWidgets usage out of AcqStore runtime deps.
2. Publish NiceWidgets before adding any optional AcqStore examples extra.
3. Do not move GUI defaults into AcqStore to satisfy the demo.
