# YOLO Learn

A reusable learning tool and experimentation platform for YOLO object detection.

## Language

**Box**:
A bounding box in `xyxy` format — `[x1, y1, x2, y2]` in pixel coordinates. This is the internal standard across all modules. COCO and Pascal VOC use this format; most tool libraries default to it.
_Avoid_: mixing formats in function signatures; always convert at I/O boundaries.

**YOLO Format**:
A bounding box in `[cx, cy, w, h]` normalized to 0–1. Used ONLY for file I/O (YOLO `.txt` label files). Convert to/from `Box` at the file-reading boundary.
_Avoid_: using YOLO format in internal calculations.

**Annotation**:
A ground-truth label: `class_id: int` + `box: Box`. No confidence score — ground truth is certain by definition.

**Prediction**:
A model output: `class_id: int` + `box: Box` + `confidence: float`. Structurally different from Annotation because confidence carries meaning.

**EvaluationResult**:
The output of one evaluation run: `mAP: float`, `per_class_ap: dict[int, float]`, and optionally PR curve data. Captures "how well did this model do on this dataset" in a single, comparable structure.

**Dataset**:
A collection of images + annotations, organized into train/val/test splits. Described by a `data.yaml` file (YOLO convention). COCO128 is the default learning dataset; custom datasets follow the same structure.

**Training Run**:
A single training session: `config` (YAML) + `model` (ultralytics YOLO) + `outputs` (weights, logs, metrics). Identified by a timestamped directory under `outputs/`.

**Experiment**:
A higher-level grouping: one or more Training Runs with the same model architecture but different hyperparameters, datasets, or augmentation strategies. For comparing "which approach works better."

## Principles

**Use ultralytics as the engine.** Training, inference, and model architecture come from ultralytics. Don't reinvent what it already provides well. Wrap it for config management and output organization, not for core ML logic.

**Internal standard is xyxy.** All functions accept and return `Box` in `[x1, y1, x2, y2]` pixel coordinates. Convert YOLO format at I/O boundaries only.

**src/ is the core engine.** Notebooks and scripts import from `yolo_learn.*`. Notebooks may include inline "teaching implementations" marked with `# 教学演示`, but the production path always goes through src/.

**Single package, not monorepo.** All modules live under one `yolo_learn` package. Split into a monorepo only when someone genuinely needs a subset without installing all dependencies.
