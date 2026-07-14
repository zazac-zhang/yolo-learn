# YOLO Learn

系统学习 YOLO 目标检测 —— 从原理到实践。

## Architecture

Single-package structure with sub-packages:

```
src/yolo_learn/
├── data/       # dataset, augmentation, download
├── models/     # training, export, inference
├── eval/       # metrics (IoU, mAP, NMS), evaluation
└── viz/        # visualization
```

- `scripts/` are thin CLI shells — core logic lives in `src/`
- `notebooks/` are teaching layers — show principles, then import from `src/`
- `configs/` holds YAML configs for datasets and training
- `docs/adr/` holds architectural decisions
- `CONTEXT.md` holds the domain glossary

See `docs/adr/0001-single-package-over-monorepo.md` and `docs/adr/0002-thin-wrapper-ultralytics.md`.

## Agent skills

### Issue tracker

GitHub Issues via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at root + `docs/adr/` for architectural decisions. See `docs/agents/domain.md`.
