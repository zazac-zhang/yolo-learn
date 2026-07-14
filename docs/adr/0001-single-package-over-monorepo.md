# ADR-0001: Single package over monorepo

## Status

Accepted

## Context

The project needs a clear organizational structure that:
- Supports a reusable learning tool + experimentation platform
- Allows future extensibility (new training strategies, export formats, datasets)
- Could potentially split into independent packages if needed

The `transmit-bug` org already has a monorepo project (`mujoco`) using `uv workspace` with independent packages per robot. We considered whether `yolo-learn` should follow the same pattern.

## Decision

Use a **single package** with sub-package organization:

```
src/yolo_learn/
├── data/       # dataset, augmentation, download
├── models/     # training, export, inference
├── eval/       # metrics, evaluation
└── viz/        # visualization
```

Not a monorepo. Not flat files.

## Consequences

**Positive:**
- Simple dependency management — one `pyproject.toml`, one install
- All modules share the same domain vocabulary (CONTEXT.md)
- Easy to get started: `pip install yolo-learn` gets everything

**Negative:**
- Users who only want visualization must install ultralytics too
- No independent versioning of sub-packages

**Splitting signal:**
Split into a monorepo only when a real user says "I want `yolo-learn-viz` without installing ultralytics." Until then, the single-package simplicity wins.

## Alternatives considered

- **Monorepo with uv workspace** (like mujoco): overkill for a single-domain project with 338 lines of code
- **Flat files** (no sub-packages): doesn't scale past ~10 files; unclear where new code goes
