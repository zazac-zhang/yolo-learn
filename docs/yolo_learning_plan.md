# YOLO 学习计划

## 目标

通过 10 个递进式 Notebook，从原理到实践系统掌握 YOLO 目标检测。

## 硬件环境

- Apple Silicon Mac (MPS 加速)
- YOLOv8/v11 (ultralytics)

## 数据集选择：COCO128

**为什么选 COCO128？**

| 数据集 | 图片数 | 类别数 | 训练时间(MPS) | 适合阶段 |
|--------|--------|--------|---------------|----------|
| COCO8 | 8 | 80 | ~1min | 仅验证代码能跑通 |
| **COCO128** | **128** | **80** | **~10min** | **学习原理+看效果** |
| Pascal VOC | 17k | 20 | ~2h | 进阶训练 |
| COCO | 118k | 80 | ~几天 | 正式训练 |

COCO128 是 ultralytics 官方维护的 COCO 子集：
- 128 张图片，覆盖 80 个类别
- 自动下载，无需手动处理
- 足够小可以在 MPS 上训练，但足够丰富能看到真实效果
- YOLO 格式标注，可以直接学习数据组织方式

## 学习路径

```
01_what_is_object_detection.ipynb    ← 概念基础：检测 vs 分类 vs 分割
02_dataset_and_annotations.ipynb     ← 数据：YOLO 标注格式、数据组织方式
03_yolo_architecture.ipynb           ← 架构：Backbone → Neck → Head
04_inference_with_pretrained.ipynb   ← 推理：用预训练模型做预测
05_training_basics.ipynb             ← 训练：loss 函数、优化器、超参数
06_transfer_learning.ipynb           ← 迁移学习：在小数据集上微调
07_evaluation_metrics.ipynb          ← 评估：IoU、mAP、PR 曲线
08_data_augmentation.ipynb           ← 增强：数据增强策略及效果
09_custom_dataset.ipynb              ← 自定义：准备自己的数据集
10_export_and_deploy.ipynb           ← 部署：导出 ONNX、优化推理
```

## 目录结构

```
yolo_learn/
├── notebooks/              # 核心学习内容（Jupyter Notebooks）
│   ├── 01_*.ipynb
│   ├── ...
│   └── 10_*.ipynb
├── src/                    # 自定义 Python 模块
│   └── yolo_learn/
│       ├── __init__.py
│       ├── visualize.py    # 可视化工具（画框、标注、对比）
│       ├── dataset.py      # 数据集工具（格式转换、统计分析）
│       └── metrics.py      # 指标计算（IoU、mAP 实现）
├── configs/                # 训练配置
│   ├── data/
│   │   ├── coco128.yaml
│   │   └── custom_example.yaml
│   └── train/
│       ├── baseline.yaml
│       └── finetune.yaml
├── scripts/                # 实用脚本
│   ├── download_dataset.py
│   ├── train.py
│   └── export.py
├── docs/                   # 学习笔记和理论文档
│   ├── yolo_learning_plan.md
│   └── yolo_theory.md
├── outputs/                # 训练输出（.gitignore）
├── data/                   # 数据集（.gitignore）
├── pyproject.toml
└── .gitignore
```

## 核心概念清单

每个 Notebook 覆盖的概念：

### 01 - 什么是目标检测
- 图像分类 vs 目标检测 vs 实例分割
- 检测的输出：bounding box + class + confidence
- 为什么需要目标检测（应用场景）

### 02 - 数据集与标注
- YOLO 标注格式：`class_id x_center y_center width height`
- 归一化坐标 vs 像素坐标
- 数据集组织：images/ labels/ 目录结构
- data.yaml 配置文件
- 类别映射

### 03 - YOLO 架构
- 单阶段检测器 vs 两阶段检测器（YOLO vs Faster R-CNN）
- 网格划分思想：将图片分成 S×S 网格
- Backbone：特征提取（CSPDarknet → C2f）
- Neck：特征融合（FPN → PANet → C2f）
- Head：检测头（anchor-based → anchor-free → DFL）
- 多尺度检测：P3/P4/P5 特征图

### 04 - 用预训练模型推理
- 加载预训练权重
- 推理流程：前处理 → 推理 → 后处理
- NMS（非极大值抑制）原理
- 置信度阈值的影响
- 可视化检测结果

### 05 - 训练基础
- Loss 函数组成：box_loss + cls_loss + dfl_loss
- 优化器：SGD vs AdamW
- 学习率调度：cosine annealing
- batch size、epoch 的影响
- 训练曲线解读

### 06 - 迁移学习
- 预训练权重的作用
- 冻结层 vs 微调全部层
- 在 COCO128 上微调预训练模型
- 过拟合的识别与应对

### 07 - 评估指标
- IoU（Intersection over Union）计算
- Precision、Recall、F1
- AP（Average Precision）
- mAP@0.5 vs mAP@0.5:0.95
- PR 曲线绘制
- 混淆矩阵

### 08 - 数据增强
- 常见增强：翻转、旋转、缩放、色彩变换
- Mosaic 增强原理
- MixUp 增强原理
- 增强策略对训练的影响

### 09 - 自定义数据集
- 数据收集与标注工具（Labelimg、CVAT）
- 标注格式转换（VOC → YOLO、COCO → YOLO）
- 数据集划分（train/val/test）
- 类别不平衡处理

### 10 - 导出与部署
- 模型导出：ONNX、TensorRT、CoreML
- 推理速度对比
- CoreML 在 Mac 上的加速
- 简单部署示例
