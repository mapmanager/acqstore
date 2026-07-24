# MkDocs style guide (AcqStore)

Conventions for authoring AcqStore documentation with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
Use the pattern names below when discussing doc changes in issues and PRs.

Site config: `mkdocs.yml`. Source pages: `docs/`.

---

## Pattern names (quick reference)

| Name | When to use | Markdown |
|---|---|---|
| **Inline Material icon** | Match an icon in prose | `:material-menu:{ .middle }` |
| **Tip block** | Optional shortcut or friendly suggestion | `!!! tip "Title"` |
| **Info block** | Neutral context, metadata, “validated on” notes | `!!! info "Title"` |
| **Warning block** | Required step; skipping it causes failure or bad UX | `!!! warning "Title"` |
| **Platform tabs** | Same topic, different steps per OS (Windows / macOS) | `=== "Windows"` / `=== "macOS"` |
| **Audience scope** | Data Scientist vs Developer tone and detail | [Audience and scope](#audience-and-scope) |
| **Home cards** | Landing-page navigation tiles | `<div class="grid cards" markdown>` |

---

## Admonitions (alert blocks)

Material admonitions render as colored callout boxes. AcqStore uses a small, consistent set:

### Tip block — optional, helpful

Use for “you might prefer this” guidance that is not required.

```markdown
!!! tip "Install with uv"
    From the repository root, run `uv sync --group docs`.
```

### Info block — context only

Use for provenance, version notes, “validated on” footers, and non-actionable detail.

```markdown
!!! info "Validated on"
    macOS Tahoe 26.2, Apple Silicon.
```

### Warning block — required or failure-prone

Use when the reader **must** do something, or when skipping a step commonly breaks the workflow.

```markdown
!!! warning "Keep package boundaries"
    Do not import `cloudscope` or `nicewidgets` from `src/acqstore/`.
```

Reference: [Material admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/).

---

## Platform tabs

Use Material content tabs when install or path steps differ by OS.

```markdown
=== "macOS"

    ```bash
    uv sync
    ```

=== "Windows"

    ```bash
    uv sync
    ```
```

Reference: [Material content tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/).

---

## Icons

AcqStore uses [Material for MkDocs icon emoji](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/) syntax. The site enables this via `pymdownx.emoji` with `material.extensions.emoji.to_svg` in `mkdocs.yml`.

---

## Screenshots

When a screenshot is needed:

```markdown
![Description](../assets/example.png){ .cs-screenshot .cs-screenshot-center width="760" loading=lazy }
```

Use `.cs-screenshot` from `docs/assets/stylesheets/extra.css`.

---

## Audience and scope

| Audience | Focus |
|---|---|
| **Data Scientist** | Algorithms, `AcqImage` / `AcqImageList`, notebooks, scripting |
| **Developer** | Repository layout, tests, docs build, package boundaries |

Prefer linking to [Reproducibility](../scientists/reproducibility.md) and the API pages over repeating architecture slogans.

---

## API pages

API reference pages under `docs/api/` are generated with mkdocstrings from Google-style source docstrings. Prefer improving the source docstring over duplicating API text in prose pages.
