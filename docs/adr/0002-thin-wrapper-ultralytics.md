# ADR-0002: Thin wrapper around ultralytics for training

## Status

Accepted

## Context

The project uses ultralytics as the YOLO engine. The question is how much to wrap it:

- **Option A**: Use ultralytics directly in notebooks/scripts — no wrapper
- **Option B**: Thin wrapper for config, logging, output management — core training is still ultralytics
- **Option C**: Full abstraction with custom Trainer class — multi-backend support

## Decision

**Option B — thin wrapper.**

`src/yolo_learn/models/train.py` provides:
- Config loading from YAML → ultralytics parameters
- Output directory management (timestamped runs under `outputs/`)
- Consistent logging

It does NOT:
- Rewrite training loops
- Abstract away ultralytics-specific features
- Support multiple backends

## Consequences

**Positive:**
- Consistent experiment tracking (every run has config snapshot + outputs)
- Notebooks stay clean — one function call instead of 10 parameters
- Users still learn ultralytics (the wrapper is thin, not opaque)

**Negative:**
- Another layer to maintain
- If ultralytics changes its API, the wrapper needs updating

## Alternatives considered

- **No wrapper (A)**: Simple, but every notebook duplicates the same `model.train(...)` boilerplate with different output dirs
- **Full abstraction (C)**: Overkill — ultralytics is mature; wrapping it deeply adds complexity without real benefit for a learning project
