# 速查表

## 常用 CLI

```bash
# 训练
uv run python scripts/train.py --model yolo11n.pt --data configs/data/coco128.yaml --epochs 100

# 导出
uv run python scripts/export.py --model runs/train/weights/best.pt --format onnx

# 下载数据集
uv run python scripts/download_dataset.py

# 实战挑战评估
uv run python practice/evaluation.py runs/safety_helmet/weights/best.pt
```

## Python API

```python
# 训练
from yolo_learn.models.train import train
results = train(model="yolo11n.pt", data="data.yaml", epochs=100, name="exp1")

# 推理
from yolo_learn.models.predict import predict
results = predict(model="runs/train/weights/best.pt", source="image.jpg")

# 导出
from yolo_learn.models.export import export_model, list_formats
path = export_model("best.pt", format="onnx")

# 评估
from yolo_learn.eval.evaluate import evaluate_predictions, load_class_names
from yolo_learn.eval.metrics import compute_iou, non_max_suppression

# 数据
from yolo_learn.data.dataset import read_data_yaml, print_dataset_summary
from yolo_learn.data.augment import get_augment_params
from yolo_learn.data.download import download_coco128

# 可视化
from yolo_learn.viz.visualize import draw_boxes_pil, yolo_to_xyxy

# 教学辅助
from yolo_learn.pedagogy.checkpoint import Quiz
from yolo_learn.pedagogy.scaffold import ExerciseSet
from yolo_learn.pedagogy.reflection import MistakeLibrary
```

## 超参速查

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `lr0` | 0.01 | 0.001-0.1 | 初始学习率 |
| `lrf` | 0.01 | - | 最终学习率 = lr0 × lrf |
| `momentum` | 0.937 | 0.9-0.999 | SGD 动量 |
| `weight_decay` | 0.0005 | 0-0.001 | 权重衰减 |
| `warmup_epochs` | 3.0 | 0-5 | 预热轮数 |
| `batch` | 16 | 8-64 | batch size |
| `imgsz` | 640 | 320-1280 | 输入尺寸 |
| `epochs` | 100 | 10-300 | 训练轮数 |
| `mosaic` | 1.0 | 0-1 | Mosaic 增强概率 |
| `mixup` | 0.0 | 0-0.5 | MixUp 增强概率 |
| `copy_paste` | 0.0 | 0-0.5 | 复制粘贴增强 |
| `degrees` | 0.0 | 0-180 | 旋转角度范围 |
| `translate` | 0.1 | 0-0.9 | 平移范围 |
| `scale` | 0.5 | 0-2 | 缩放范围 |
| `fliplr` | 0.5 | 0-1 | 水平翻转概率 |
| `hsv_h` | 0.015 | 0-0.1 | 色调抖动 |
| `hsv_s` | 0.7 | 0-0.9 | 饱和度抖动 |
| `hsv_v` | 0.4 | 0-0.9 | 亮度抖动 |

## 增强预设

```python
from yolo_learn.data.augment import get_augment_params

none   = get_augment_params("none")    # {}
light  = get_augment_params("light")   # 基础翻转+色调
medium = get_augment_params("medium")  # +mosaic+mixup
heavy  = get_augment_params("heavy")   # +更强的增强
```

## 模型大小

| 模型 | 参数 | mAP | 速度 | 适用场景 |
|------|------|-----|------|----------|
| YOLO11n | 2.6M | 39.5 | 最快 | 移动端/嵌入式 |
| YOLO11s | 9.4M | 47.0 | 快 | 普通 GPU |
| YOLO11m | 20.1M | 51.5 | 中 | 服务器 |
| YOLO11l | 25.3M | 53.4 | 慢 | 高精度需求 |
| YOLO11x | 56.9M | 54.7 | 最慢 | 精度优先 |

## 坐标格式

```python
# xyxy → xywh
x, y, w, h = x1, y1, x2 - x1, y2 - y1

# xywh → xyxy
x1, y1, x2, y2 = x, y, x + w, y + h

# xyxy → cxcywh
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
w, h = x2 - x1, y2 - y1

# 归一化 (pixel → yolo)
cx_norm = cx / img_w
cy_norm = cy / img_h
w_norm = w / img_w
h_norm = h / img_h
```

## 调试命令

```bash
# 检查环境
uv run python -c "import torch; print(torch.backends.mps.is_available())"
uv run python -c "from ultralytics import YOLO; print('OK')"

# 查看训练日志
cat outputs/exp1/results.csv

# 启动 TensorBoard (如果安装)
uv run tensorboard --logdir outputs/

# ruff 检查
uv run ruff check src/ scripts/ tests/

# 运行测试
uv run pytest tests/ -v
```
