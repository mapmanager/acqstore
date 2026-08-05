# AcqStore — Agent Instructions

## Repository role

`acqstore` is an independent Git repository containing the backend
acquisition/data layer for CloudScope and related tools.

- Distribution name: `acqstore`
- Python import package: `acqstore`
- Source: `src/acqstore/`
- Tests: `tests/`
- User documentation: `docs/`
- Development notes: `docs-dev/`
- Examples: `examples/`
- Scripts: `scripts/`

AcqStore owns acquisition file loading, image/trace models, ROIs, metadata,
schemas, analysis, and persistence. It is **backend only**: it has no GUI and no
knowledge of CloudScope, NiceGUI widgets, or the server layer.

> Coexistence note: This file is the primary instruction file when this repo is
> the working root (e.g. a Codex project with `acqstore/` primary). When this
> repo is opened as part of the outer `cs_project/` workspace (e.g. in Cursor),
> the outer `cs_project/AGENTS.md` plus `cs_project/.cursor/rules/` provide
> workspace-wide guidance and take precedence for cross-repo scope; this file
> stays repo-local and must not contradict it.

## Attached sibling repositories

`acqstore` has **no** source dependencies on sibling repositories. Consumers
depend on it, not the other way around.

| Repository | Local path | Relationship |
|---|---|---|
| AcqStore Server | `../acqstore-server/` | Depends on `acqstore` (editable path dep) |
| CloudScope App | `../cloudscope-app/` | Depends on `acqstore` (editable path dep) |

`acqstore` MUST NOT import from `nicewidgets`, `acqstore-server`, or
`cloudscope-app`. If a change appears to require importing a consumer, stop and
report — the dependency direction is wrong.

## Package boundaries

Put code in `acqstore` when it implements:

- acquisition file loaders and format handling;
- image, pixel, and trace models;
- ROIs, metadata, and schemas;
- scientific analysis (velocity, diameter, heart rate, events, batch, etc.);
- sample-data discovery and persistence helpers.

Do not place these in `acqstore`:

- GUI code, NiceGUI widgets, or anything importing `nicegui`;
- server endpoints or HTTP transport behavior;
- CloudScope application orchestration, controllers, or views.

Backend APIs must use backend-native values and must not leak GUI assumptions.

## Default task scope

Work only in `acqstore` unless the task explicitly includes another repository.

- Start with files named by the user and their direct dependencies.
- Do not make unrelated cleanup or consistency edits.
- Do not add abstractions for hypothetical future requirements.
- Do not add or change production dependencies without asking first.
- Ask a focused question, with a recommended answer, when a material decision
  remains ambiguous after inspecting the relevant code.

## Curated public API (frozen)

New or updated `__init__.py` files are **empty by default**. Do not add imports,
`__all__`, docstrings, or side effects unless the task explicitly extends a
curated public API surface.

Frozen curated allowlist (do not modify without an explicit request that names
the file and symbols):

- `src/acqstore/acq_image/__init__.py`
- `src/acqstore/acq_image/analysis/__init__.py`
- `src/acqstore/acq_image/analysis/batch/__init__.py`

The import contract is tested in `tests/acqstore/test_public_imports.py`; keep
it passing when touching allowlisted files. Elsewhere, import via full module
paths (e.g. `from acqstore.acq_image.acq_image import AcqImage`).

## Environment and commands

Run commands from the `acqstore/` repository root. Use `uv run`.

```bash
uv sync
uv run pytest path/to/test_file.py   # focused first
uv run pytest                        # full suite
uv run ruff check src tests
```

## Verification

- Source or API changes: run focused tests, then the full suite when practical.
- Formatting or lint-sensitive changes: run the relevant Ruff check.
- New functionality and edge cases must have deterministic tests.
- Do not weaken a meaningful test merely to make it pass. Determine whether the
  implementation or the test expectation is wrong; if the API is wrong, warn and
  fix the source.

## Coding conventions

- Keep changes small, direct, and maintainable.
- Prefer KISS and DRY without speculative shared modules.
- Use type annotations and Google-style docstrings (`Args`, `Returns`,
  `Raises`) for public APIs.
- Fail clearly on invalid input rather than silently guessing.
- Preserve existing architecture and naming unless the task is a deliberate
  refactor.

## Documentation and ticket reports

Do not update the repository-root `README.md` unless the task explicitly
requests a README change.

Do not create a `docs-dev/cursor_tickets/` report by default. Create one only
when the user explicitly identifies the work as a tracked implementation ticket
or requests a report. Use the next unused three-digit prefix. The
`cursor_tickets/` name is a project convention regardless of which agent writes
the report. Record: requested scope; repositories and files changed; important
decisions; verification performed; unresolved or unverified behavior.

## Search exclusions

Unless the task explicitly requires them, do not inspect or search:

- `.venv/`, `venv/`, `__pycache__/`, and tool caches;
- `build/`, `dist/`, `site/`, and generated output;
- `*.zip`, `*.tar`, `*.tar.gz`, and `*.whl`;
- `.git/`;
- large generated data or binary assets, and `docs/notebooks/` outputs.

Do not treat old monorepo paths or archived development notes as current
architecture when they conflict with the present repository.

## Git discipline

This directory is an independent Git repository.

- Check `git status` before and after material work.
- Preserve unrelated user changes.
- Do not commit, push, create branches, or open pull requests unless explicitly
  requested.
- For cross-repository work, report and verify changes separately in each
  affected repository.
