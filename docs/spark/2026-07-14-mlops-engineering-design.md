# MLOps 工程能力设计

日期：2026-07-14
状态：草稿

## 当前状态

项目已有基础工程能力：
- 训练脚本 + 配置管理
- 导出脚本（基本的 ONNX/CoreML）
- 推理模块（单张图片）
- 评估模块（mAP/IoU）
- 单元测试 + CI

缺失的工程能力：实验追踪、模型验证、性能基准、模型优化、推理服务、容器化、边缘/移动部署。

## 设计目标

为 yolo-learn 项目补充完整的轻量 MLOps 工程链路，覆盖从训练到部署的全流程。

## 子项目分解

MLOps 范围太大，分解为 5 个独立子项目，每个可独立实现和测试：

```
┌─────────────────────────────────────────────────────────┐
│                    MLOps 工程链路                        │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ 1.实验追踪 │ 2.导出验证 │ 3.模型优化 │ 4.推理服务 │ 5.边缘部署  │
│ MLflow   │ 多格式+   │ FP16/INT8│ REST API │ NCNN/       │
│          │ 基准测试  │          │ Docker   │ CoreML/     │
│          │          │          │          │ TFLite      │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

建议实施顺序：1 → 2 → 3 → 4 → 5

---

## 子项目 1：MLflow 实验追踪

### 目标

训练过程自动记录到 MLflow，支持实验对比和模型注册。

### 模块设计

新增 `src/yolo_learn/experiment/` 子包：

```
src/yolo_learn/experiment/
├── tracker.py        ← MLflow 集成：start_run, log_params, log_metrics, log_model
├── registry.py       ← 模型注册：register_model, load_model, list_versions
└── compare.py        ← 实验对比：compare_runs, plot_comparison
```

**tracker.py 核心接口：**
- `start_experiment(name, tags) -> Run` — 开始实验
- `log_params(params: dict)` — 记录超参
- `log_metrics(metrics: dict, step: int)` — 记录指标（每个 epoch）
- `log_model(model_path, name)` — 记录模型产物
- `log_dataset(data_path, name)` — 记录数据集版本（hash）

**registry.py 核心接口：**
- `register_model(run_id, name, stage)` — 注册模型
- `load_model(name, stage) -> model` — 加载模型（Production/Staging/Archived）
- `list_versions(name) -> list` — 列出所有版本
- `promote_version(name, version, stage)` — 提升模型阶段

**compare.py 核心接口：**
- `compare_runs(run_ids) -> DataFrame` — 对比多个实验的指标
- `plot_comparison(run_ids, metric)` — 可视化对比

### 与现有代码的集成

修改 `scripts/train.py` 和 `src/yolo_learn/models/train.py`：
- 训练完成后自动调用 `tracker.log_*`
- 可选启用/禁用 MLflow（`--no-mlflow`）

### 依赖

新增 `[experiment]` 可选依赖组：`mlflow>=2.0`

### 测试

- `tests/test_experiment/test_tracker.py` — mock MLflow，测试 log 调用
- `tests/test_experiment/test_compare.py` — 测试对比逻辑

---

## 子项目 2：模型导出验证 + 基准测试

### 目标

导出模型后自动验证精度一致，测量推理性能。

### 模块设计

扩展 `src/yolo_learn/models/` 子包：

```
src/yolo_learn/models/
├── export.py          ← 扩展：支持更多格式 + 验证
├── benchmark.py       ← 新增：性能基准测试
└── validate_export.py ← 新增：导出后精度验证
```

**export.py 扩展：**

当前 `export_model()` 只做基本导出。扩展为：
- 支持格式：onnx, torchscript, coreml, tflite, ncnn, openvino, tensorrt
- 每个格式有独立的导出参数
- 导出后可选自动验证

**benchmark.py 核心接口：**
- `benchmark_model(model_path, imgsz, num_runs, warmup) -> BenchmarkResult` — 测量延迟
- `BenchmarkResult` dataclass：`avg_ms, p50_ms, p95_ms, p99_ms, throughput_fps, memory_mb`
- `compare_formats(model_path, formats) -> DataFrame` — 对比不同格式的性能
- `benchmark_batch_sizes(model_path, batch_sizes) -> DataFrame` — 不同 batch size 的吞吐量

**validate_export.py 核心接口：**
- `validate_export(original_path, exported_path, test_images, tolerance) -> ValidationResult` — 验证导出模型精度
- `ValidationResult` dataclass：`max_diff, mean_diff, passed, details`
- 逻辑：用同一组图片分别跑原始模型和导出模型，对比输出差异

### 测试

- `tests/test_models/test_benchmark.py` — 测试 BenchmarkResult 计算
- `tests/test_models/test_validate_export.py` — 测试验证逻辑

### 依赖

无新增依赖（ultralytics 已支持大部分格式）

---

## 子项目 3：模型优化

### 目标

支持 FP16/INT8 量化和模型剪枝，减小模型体积、提升推理速度。

### 模块设计

新增 `src/yolo_learn/optimize/` 子包：

```
src/yolo_learn/optimize/
├── quantize.py    ← 量化：FP16, INT8 (PTQ)
├── prune.py       ← 剪枝：结构化/非结构化剪枝
└── profile.py     ← 模型分析：参数量、FLOPs、层耗时
```

**quantize.py 核心接口：**
- `quantize_fp16(model_path) -> Path` — FP16 量化（导出时 half=True）
- `quantize_int8(model_path, calibration_data) -> Path` — INT8 PTQ 量化
- `compare_precision(original, quantized, test_data) -> dict` — 对比量化前后精度

**prune.py 核心接口：**
- `prune_model(model_path, ratio, method) -> Path` — 模型剪枝
- `method`: "l1" (L1 范数剪枝), "structured" (通道剪枝)
- `prune_and_finetune(model_path, data, ratio, epochs) -> Path` — 剪枝 + 微调

**profile.py 核心接口：**
- `profile_model(model_path) -> ModelProfile` — 分析模型
- `ModelProfile` dataclass：`params, flops, layers, layer_times`
- `compare_models(model_paths) -> DataFrame` — 对比多个模型的复杂度

### 依赖

新增 `[optimize]` 可选依赖组：`torch-pruning>=1.0`

### 测试

- `tests/test_optimize/test_profile.py` — 测试 ModelProfile 计算
- `tests/test_optimize/test_quantize.py` — mock 测试量化流程

---

## 子项目 4：推理服务

### 目标

提供 REST API 推理服务，支持 Docker 部署。

### 模块设计

新增 `src/yolo_learn/serve/` 子包 + `deploy/` 目录：

```
src/yolo_learn/serve/
├── api.py         ← FastAPI 应用：/predict, /health, /metrics
├── pipeline.py    ← 推理管道：前处理→推理→后处理
└── middleware.py   ← 中间件：日志、限流、错误处理

deploy/
├── Dockerfile     ← 生产镜像
├── Dockerfile.gpu ← GPU 镜像
└── docker-compose.yml
```

**api.py 核心接口：**
- `POST /predict` — 单张图片推理（multipart/form-data）
- `POST /predict/batch` — 批量推理
- `GET /health` — 健康检查
- `GET /metrics` — Prometheus 指标
- `GET /model/info` — 模型信息

**pipeline.py 核心接口：**
- `InferencePipeline(model_path, device, imgsz)` — 推理管道
- `pipeline.predict(image) -> DetectionResult` — 单张推理
- `pipeline.predict_batch(images) -> list[DetectionResult]` — 批量推理
- `pipeline.preprocess(image) -> tensor` — 前处理
- `pipeline.postprocess(output) -> DetectionResult` — 后处理

**Dockerfile 设计：**
- 基于 `python:3.13-slim`
- 安装 `yolo-learn[serve]`
- 暴露 8000 端口
- 支持 GPU（nvidia/cuda 基础镜像）

### 依赖

新增 `[serve]` 可选依赖组：`fastapi>=0.100`, `uvicorn>=0.20`, `python-multipart`

### 测试

- `tests/test_serve/test_api.py` — TestClient 测试 API 端点
- `tests/test_serve/test_pipeline.py` — 测试推理管道

---

## 子项目 5：边缘/移动部署

### 目标

提供针对边缘设备和移动端的模型优化和部署指南。

### 模块设计

新增 `src/yolo_learn/deploy/` 子包：

```
src/yolo_learn/deploy/
├── edge.py        ← 边缘设备：NCNN, OpenVINO
├── mobile.py      ← 移动端：CoreML (iOS), TFLite (Android)
└── guide.py       ← 部署指南生成
```

**edge.py 核心接口：**
- `export_ncnn(model_path, imgsz) -> Path` — 导出 NCNN 格式
- `export_openvino(model_path, imgsz) -> Path` — 导出 OpenVINO 格式
- `benchmark_edge(model_path, device) -> BenchmarkResult` — 边缘设备基准

**mobile.py 核心接口：**
- `export_coreml(model_path, imgsz, half) -> Path` — 导出 CoreML
- `export_tflite(model_path, imgsz, int8) -> Path` — 导出 TFLite
- `validate_mobile_export(original, exported, test_images) -> bool` — 验证精度

**guide.py 核心接口：**
- `generate_deploy_guide(model_path, target) -> str` — 生成部署指南
- `target`: "ios", "android", "raspberry_pi", "jetson"
- 输出：步骤说明 + 代码示例 + 性能预期

### 依赖

无新增核心依赖（ultralytics 已支持大部分格式导出）

### 测试

- `tests/test_deploy/test_edge.py` — 测试导出逻辑
- `tests/test_deploy/test_mobile.py` — 测试移动端导出

---

## 实施顺序

```
Phase 1: MLflow 实验追踪 (子项目1)
  ├── tracker.py + registry.py + compare.py
  ├── 集成到 train.py
  └── 单元测试

Phase 2: 导出验证 + 基准测试 (子项目2)
  ├── benchmark.py
  ├── validate_export.py
  └── 扩展 export.py

Phase 3: 模型优化 (子项目3)
  ├── quantize.py
  ├── prune.py
  └── profile.py

Phase 4: 推理服务 (子项目4)
  ├── api.py + pipeline.py
  ├── Dockerfile
  └── 集成测试

Phase 5: 边缘/移动部署 (子项目5)
  ├── edge.py + mobile.py
  └── guide.py
```

每个 Phase 约 1-2 天，总计约 1-2 周。

## 依赖汇总

```toml
[project.optional-dependencies]
experiment = ["mlflow>=2.0"]
optimize = ["torch-pruning>=1.0"]
serve = ["fastapi>=0.100", "uvicorn>=0.20", "python-multipart"]
deploy = ["onnxruntime>=1.16"]  # ONNX 推理验证
mlops = ["yolo-learn[experiment,optimize,serve,deploy]"]
```

## 不在范围内

- Kubernetes 编排（轻量 MLOps 不需要）
- 分布式训练（单机足够）
- 数据版本控制（DVC）— 可以后续添加
- 超参搜索（Optuna）— 可以后续添加
- 生产级监控（Prometheus + Grafana）— 可以后续添加
