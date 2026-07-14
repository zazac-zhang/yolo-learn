# 常见问题 (FAQ)

## 安装与环境

### `uv sync` 报错 Python 版本不满足

```
The current Python version (3.12.x) is not compatible with the required version (>=3.13)
```

需要 Python 3.13+。检查版本：
```bash
python3 --version
```

升级 Python：
```bash
# macOS
brew install python@3.13

# 或用 uv 安装
uv python install 3.13
```

### ultralytics 安装失败

```bash
# 清除缓存重装
uv cache clean
uv sync

# 如果网络问题，用镜像
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### PyTorch 版本冲突

ultralytics 依赖 PyTorch。如果版本冲突：
```bash
uv remove torch torchvision
uv sync
```

## 训练相关

### CUDA Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

解决方法（按优先级）：
1. **减小 batch size** — `batch=8` 或 `batch=4`
2. **减小图片尺寸** — `imgsz=416`（默认 640）
3. **减小模型** — 用 `yolo11n.pt` 而不是 `yolo11m.pt`
4. **关闭增强** — `augment=False`（减少中间变量）

```bash
uv run python scripts/train.py --model yolo11n.pt --data data.yaml --batch 4 --imgsz 416
```

### MPS 不支持某些操作 (Apple Silicon)

```
NotImplementedError: Could not run 'aten::...' with arguments from the 'MPS' backend
```

解决方法：
```bash
# 强制使用 CPU
uv run python scripts/train.py --device cpu

# 或设置环境变量
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### 训练 loss 不下降

检查项：
1. **学习率** — 太大震荡，太小不动。试试 `lr0=0.001`
2. **数据** — 标注是否正确？用 `print_dataset_summary` 检查
3. **batch size** — 太小梯度不稳定。试试 `batch=16`
4. **数据增强** — 太强可能干扰。先关掉 `augment=False`

### 训练中断后恢复

```bash
# 自动从最后 checkpoint 恢复
uv run python scripts/train.py --resume

# 指定 checkpoint
uv run python scripts/train.py --resume runs/exp1/weights/last.pt
```

## 数据相关

### 标注格式错误

```
WARNING: ignoring corrupt label...
```

检查标注文件：
- 每行 5 个值：`class_id x_center y_center width height`
- 坐标在 0-1 之间（归一化值）
- class_id 从 0 开始

```python
# 批量检查
from pathlib import Path
for f in Path("data/your_dataset/train/labels").glob("*.txt"):
    for i, line in enumerate(f.read_text().strip().split("\n")):
        parts = line.split()
        if len(parts) != 5:
            print(f"{f}:{i} bad format: {line}")
        coords = list(map(float, parts[1:]))
        if any(c < 0 or c > 1 for c in coords):
            print(f"{f}:{i} coords out of [0,1]: {coords}")
```

### data.yaml 路径错误

```
FileNotFoundError: data.yaml not found
```

确保：
- `path` 是数据集根目录的绝对路径（或相对路径）
- `train`/`val` 是相对于 `path` 的路径
- 路径分隔符用 `/`，不要用 `\`

```yaml
# 正确
path: /Users/yourname/datasets/my_dataset
train: train/images
val: val/images

# 错误
path: ~/datasets/my_dataset  # ~ 不展开
train: train\images          # 用 /
```

### 类别 ID 从 1 开始

训练时类别错位，mAP 很低。检查：
```python
# 读取标注文件，确认 class_id 从 0 开始
line = "0 0.5 0.5 0.3 0.4"  # ✅ 正确
line = "1 0.5 0.5 0.3 0.4"  # ❌ 错误，应该是 0
```

## 评估相关

### mAP 为 0

检查项：
1. **模型是否训练好** — 检查 loss 是否下降
2. **data.yaml 是否正确** — val 路径是否存在
3. **标注格式** — 是否和训练时一致
4. **IoU 阈值** — 默认 0.5，试试降低

### mAP@0.5 和 mAP@0.5:0.95 搞混

```
metrics.box.map50      # mAP@0.5（宽松）
metrics.box.map        # mAP@0.5:0.95（严格）
```

### 混淆矩阵不显示

需要安装 seaborn：
```bash
uv sync --extra notebook
```

## 导出相关

### ONNX 导出失败

```
RuntimeError: Exporting the operator 'aten::...' to ONNX opset version XX is not supported
```

解决方法：
1. 更新 ultralytics — `uv sync`
2. 降低 opset 版本 — `model.export(format="onnx", opset=11)`
3. 用其他格式 — `model.export(format="torchscript")`

### CoreML 导出需要 macOS

CoreML 导出只能在 macOS 上运行。Linux 用户用 ONNX 替代。

## Jupyter 相关

### 内核找不到

```bash
# 安装 ipykernel
uv sync --extra dev

# 注册内核
uv run python -m ipykernel install --user --name yolo-learn
```

### notebook 中 import 找不到模块

确保 notebook 从项目根目录运行，或添加路径：
```python
import sys
sys.path.insert(0, "..")  # 或者用绝对路径
```

## 更多帮助

- [ultralytics 文档](https://docs.ultralytics.com/)
- [YOLO GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- 项目 Issue：在 GitHub 上提 issue
