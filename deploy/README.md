# 部署指南

## Docker 部署

### CPU 版本

```bash
# 构建镜像
docker build -f deploy/docker/Dockerfile -t yolo-learn .

# 运行容器
docker run -p 8000:8000 -e MODEL_PATH=yolo11n.pt yolo-learn
```

### GPU 版本

```bash
# 构建镜像
docker build -f deploy/docker/Dockerfile.gpu -t yolo-learn:gpu .

# 运行容器（需要 nvidia-docker）
docker run --gpus all -p 8000:8000 -e MODEL_PATH=yolo11n.pt yolo-learn:gpu
```

## API 使用

### 推理

```bash
# 单张图片
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -F "conf=0.25"

# 批量图片
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

### 响应格式

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "box": [100, 50, 300, 400]
    }
  ],
  "count": 1,
  "model_path": "yolo11n.pt"
}
```

### 健康检查

```bash
curl http://localhost:8000/health
```

### 交互式文档

浏览器打开 `http://localhost:8000/docs`

## 模型导出

```python
from yolo_learn.models.export import export_model

# 导出 ONNX（推荐，通用格式）
onnx_path = export_model("best.pt", format="onnx", half=True)

# 导出 CoreML（iOS）
coreml_path = export_model("best.pt", format="coreml")

# 导出 TFLite（Android）
tflite_path = export_model("best.pt", format="tflite")

# 导出 NCNN（边缘设备）
ncnn_path = export_model("best.pt", format="ncnn")
```

## 性能基准

```python
from yolo_learn.models.benchmark import benchmark_model, benchmark_onnx, compare_formats

# 测试 PyTorch 模型
result = benchmark_model("best.pt", num_runs=100)
print(result.summary())

# 测试 ONNX 模型
result = benchmark_onnx("best.onnx", num_runs=100)
print(result.summary())

# 对比不同格式
results = compare_formats("best.pt", formats=["pytorch", "onnx"])
for r in results:
    print(r.summary())
```

## 边缘设备部署

### 树莓派 / Jetson

1. 导出 NCNN 格式：
   ```python
   export_model("best.pt", format="ncnn")
   ```

2. 将 `_ncnn_model/` 目录复制到设备

3. 使用 NCNN C++ API 加载模型

### OpenVINO (Intel CPU)

1. 导出 OpenVINO 格式：
   ```python
   export_model("best.pt", format="openvino")
   ```

2. 使用 OpenVINO Runtime 推理

## 移动端部署

### iOS (CoreML)

1. 导出 CoreML 格式：
   ```python
   export_model("best.pt", format="coreml")
   ```

2. 将 `.mlpackage` 拖入 Xcode 项目

3. 使用 Vision Framework 调用

### Android (TFLite)

1. 导出 TFLite 格式：
   ```python
   export_model("best.pt", format="tflite", half=True)
   ```

2. 将 `.tflite` 文件放入 `assets/`

3. 使用 TensorFlow Lite Interpreter 调用
