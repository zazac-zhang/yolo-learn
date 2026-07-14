# MLOps 工程能力设计 (v2)

日期：2026-07-14
状态：草稿

## 设计原则

1. **不放大量内容在根目录** — 部署相关文件放 `deploy/` 下
2. **优先使用成熟库** — 不重复造轮子
3. **最小化新增模块** — 能扩展现有模块就不新建

## 子项目（3 个）

### 1. 模型导出 + 验证 + 基准

扩展现有 `models/export.py`，新增 `models/benchmark.py`。

**export.py 扩展：**
- `export_model()` 增加 `half` 参数（FP16）
- 增加 `validate` 参数，导出后自动验证精度
- 支持格式：onnx, torchscript, coreml, tflite, ncnn, openvino

**benchmark.py（新增）：**
- 使用 `onnxruntime` 做 ONNX 推理基准
- 使用 `time.perf_counter` 做原生模型基准
- `benchmark_model(path, imgsz, runs, warmup) -> BenchmarkResult`
- `compare_formats(model, formats, imgsz) -> DataFrame`

**依赖：** `onnxruntime>=1.16`

### 2. 推理服务

新增 `serve/server.py`，使用 FastAPI。

**server.py：**
- `POST /predict` — 单张推理
- `POST /predict/batch` — 批量推理
- `GET /health` — 健康检查
- 使用 `python-multipart` 处理图片上传
- 使用 `uvicorn` 运行

**依赖：** `fastapi>=0.100`, `uvicorn>=0.20`, `python-multipart`

### 3. 实验记录

新增 `experiment/tracker.py`，使用 MLflow（成熟库，用户要求优先用成熟库）。

**tracker.py：**
- `start_run(name, tags)` — 包装 `mlflow.start_run`
- `log_params(params)` — 记录超参
- `log_metrics(metrics, step)` — 记录指标
- `log_artifact(path)` — 记录产物（模型、图表）
- `get_best_run(metric, direction)` — 获取最佳实验
- 薄封装，不重新发明轮子

**依赖：** `mlflow>=2.0`

## 目录结构

```
yolo_learn/
├── src/yolo_learn/
│   ├── models/
│   │   ├── export.py        ← 扩展：多格式 + FP16 + 验证
│   │   ├── benchmark.py     ← 新增：性能基准
│   │   ├── train.py
│   │   └── predict.py
│   ├── serve/
│   │   └── server.py        ← 新增：FastAPI 推理服务
│   ├── experiment/
│   │   └── tracker.py       ← 新增：MLflow 封装
│   ├── data/
│   ├── eval/
│   ├── viz/
│   └── pedagogy/
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile       ← CPU 镜像
│   │   └── Dockerfile.gpu   ← GPU 镜像
│   └── README.md            ← 部署说明
├── configs/
├── scripts/
├── notebooks/
├── tests/
└── docs/
```

## 依赖变更

```toml
[project.optional-dependencies]
# 现有
notebook = [...]
deploy = [...]
roboflow = [...]
dev = [...]

# 新增
serve = ["fastapi>=0.100", "uvicorn>=0.20", "python-multipart"]
experiment = ["mlflow>=2.0"]
benchmark = ["onnxruntime>=1.16"]
mlops = ["yolo-learn[serve,experiment,benchmark]"]
full = ["yolo-learn[notebook,deploy,roboflow,dev,mlops]"]
```

## 实施顺序

```
Phase 1: benchmark.py          ← 独立，不依赖其他
Phase 2: export.py 扩展        ← 依赖 benchmark（导出后自动跑基准）
Phase 3: experiment/tracker.py ← 独立
Phase 4: serve/server.py       ← 独立
Phase 5: deploy/docker/        ← 依赖 server.py
```

## 不做的内容

| 不做 | 原因 |
|------|------|
| 模型剪枝 | 学习项目用不到 |
| INT8 量化 | 需要校准数据，复杂度高 |
| 分布式训练 | 单机足够 |
| K8s 编排 | 轻量 MLOps 不需要 |
| 自建实验追踪 | 用 MLflow |
| 自建基准测试框架 | 用 onnxruntime |
