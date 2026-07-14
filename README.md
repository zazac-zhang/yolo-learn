# YOLO Learn

系统学习 YOLO 目标检测 —— 从原理到实践。

## 快速开始

```bash
# 安装核心依赖
uv sync

# 安装完整依赖（含 notebook、部署工具）
uv sync --extra full

# 启动 Jupyter
uv run jupyter notebook

# 打开 notebooks/01_what_is_object_detection.ipynb 开始学习
```

## 学习路径

| # | Notebook | 核心概念 |
|---|----------|----------|
| 01 | 什么是目标检测 | 检测 vs 分类 vs 分割 |
| 02 | 数据集与标注 | YOLO 格式、数据组织 |
| 03 | YOLO 架构 | Backbone → Neck → Head |
| 04 | 用预训练模型推理 | 前处理 → 推理 → 后处理 |
| 05 | 训练基础 | Loss、优化器、超参数 |
| 06 | 迁移学习 | 微调预训练模型 |
| 07 | 评估指标 | IoU、mAP、PR 曲线 |
| 08 | 数据增强 | Mosaic、MixUp 等策略 |
| 09 | 自定义数据集 | 标注、格式转换 |
| 10 | 导出与部署 | ONNX、CoreML |

## 目录结构

```
yolo_learn/
├── src/yolo_learn/         # 核心引擎
│   ├── data/               #   数据集、增强、下载
│   ├── models/             #   训练、导出、推理
│   ├── eval/               #   指标 (IoU, mAP, NMS)、评估
│   └── viz/                #   可视化
├── notebooks/              # 学习 Notebook（教学层）
├── scripts/                # CLI 入口（薄壳）
├── configs/                # 训练和数据配置
├── practice/               # 实战挑战（安全帽检测）
├── tests/                  # 单元测试
├── docs/                   # 文档、ADR、Agent 配置
├── outputs/                # 训练输出（gitignore）
└── data/                   # 数据集（gitignore）
```

## 使用 src 模块

```python
# 数据
from yolo_learn.data.dataset import read_data_yaml, print_dataset_summary
from yolo_learn.data.augment import get_augment_params
from yolo_learn.data.download import download_coco128

# 训练
from yolo_learn.models.train import train
from yolo_learn.models.export import export_model
from yolo_learn.models.predict import predict

# 评估
from yolo_learn.eval.metrics import compute_iou, non_max_suppression
from yolo_learn.eval.evaluate import Annotation, Prediction, evaluate_predictions

# 可视化
from yolo_learn.viz.visualize import draw_boxes_pil, yolo_to_xyxy
```

## 实战挑战

完成 10 个 Notebook 后，试试 [practice/](practice/) 里的安全帽检测挑战。

## 依赖

| 组 | 安装 | 内容 |
|---|---|---|
| 核心 | `uv sync` | ultralytics, numpy, matplotlib, pandas, pyyaml |
| Notebook | `uv sync --extra notebook` | jupyter, ipywidgets, seaborn |
| 部署 | `uv sync --extra deploy` | onnx, coremltools |
| Roboflow | `uv sync --extra roboflow` | roboflow SDK |
| 全部 | `uv sync --extra full` | 以上所有 + dev 工具 |

## 环境

- Python 3.13+
- Apple Silicon Mac (MPS 加速) / CUDA GPU / CPU
- ultralytics (YOLOv8/v11)
