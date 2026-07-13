# YOLO Learn

系统学习 YOLO 目标检测 —— 从原理到实践。

## 快速开始

```bash
# 安装依赖
uv sync

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
├── notebooks/              # 核心学习内容
├── src/yolo_learn/         # 自定义工具模块
├── configs/                # 训练和数据配置
├── scripts/                # 实用脚本
├── docs/                   # 理论文档
├── outputs/                # 训练输出（gitignore）
└── data/                   # 数据集（gitignore）
```

## 数据集

使用 **COCO128** —— COCO 的 128 张图片子集，ultralytics 自动下载。

## 环境

- Python 3.13+
- Apple Silicon Mac (MPS 加速)
- YOLOv8/v11 (ultralytics)
