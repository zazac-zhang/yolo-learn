# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the domain glossary (Box, Annotation, Prediction, EvaluationResult, etc.)
- **`docs/adr/`** — architectural decisions (package structure, training layer, etc.)

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront.

## File structure

Single-context repo:

```
/
├── CONTEXT.md                    ← Domain glossary (Box, Annotation, Prediction, etc.)
├── AGENTS.md                     ← Agent skills reference
├── docs/
│   ├── adr/                      ← Architectural decisions
│   │   ├── 0001-single-package-over-monorepo.md
│   │   └── 0002-thin-wrapper-ultralytics.md
│   └── agents/                   ← Agent skill docs (this file)
├── src/yolo_learn/               ← Core engine
│   ├── data/                     ← Dataset, augmentation, download
│   ├── models/                   ← Training, export, inference
│   ├── eval/                     ← Metrics, evaluation
│   └── viz/                      ← Visualization
├── scripts/                      ← Thin CLI shells
├── notebooks/                    ← Teaching layers
└── configs/                      ← YAML configs (data, train)
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms.

Key terms:
- **Box**: `[x1, y1, x2, y2]` pixel coordinates (xyxy format). NOT YOLO format.
- **Annotation**: `class_id` + `box` (no confidence)
- **Prediction**: `class_id` + `box` + `confidence`
- **EvaluationResult**: `mAP` + `per_class_ap` + metadata

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding.
