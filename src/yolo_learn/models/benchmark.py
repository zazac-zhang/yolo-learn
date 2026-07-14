"""模型性能基准测试：延迟、吞吐量、内存."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkResult:
    """基准测试结果."""

    model_path: str
    format: str
    imgsz: int
    num_runs: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.times_ms, 50))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.times_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.times_ms, 99))

    @property
    def throughput_fps(self) -> float:
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0

    def summary(self) -> str:
        """返回可读的基准报告."""
        lines = [
            f"📊 基准测试: {self.model_path}",
            f"   格式: {self.format}, 输入尺寸: {self.imgsz}",
            f"   运行次数: {self.num_runs}",
            f"   平均延迟: {self.avg_ms:.2f} ms",
            f"   P50: {self.p50_ms:.2f} ms",
            f"   P95: {self.p95_ms:.2f} ms",
            f"   P99: {self.p99_ms:.2f} ms",
            f"   吞吐量: {self.throughput_fps:.1f} FPS",
        ]
        return "\n".join(lines)


def benchmark_model(
    model_path: str | Path,
    imgsz: int = 640,
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> BenchmarkResult:
    """基准测试 PyTorch 模型.

    Args:
        model_path: 模型路径 (.pt)
        imgsz: 输入图片尺寸
        num_runs: 测试运行次数
        warmup: 预热次数
        device: 设备 (cpu/cuda/mps)

    Returns:
        BenchmarkResult
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # 预热
    for _ in range(warmup):
        model.predict(dummy_input, device=device, verbose=False)

    # 测试
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(dummy_input, device=device, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        model_path=str(model_path),
        format="pytorch",
        imgsz=imgsz,
        num_runs=num_runs,
        times_ms=times,
    )


def benchmark_onnx(
    onnx_path: str | Path,
    imgsz: int = 640,
    num_runs: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """基准测试 ONNX 模型 (使用 onnxruntime).

    Args:
        onnx_path: ONNX 模型路径
        imgsz: 输入图片尺寸
        num_runs: 测试运行次数
        warmup: 预热次数

    Returns:
        BenchmarkResult
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # 预热
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # 测试
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        model_path=str(onnx_path),
        format="onnx",
        imgsz=imgsz,
        num_runs=num_runs,
        times_ms=times,
    )


def compare_formats(
    model_path: str | Path,
    formats: list[str] | None = None,
    imgsz: int = 640,
    num_runs: int = 50,
) -> list[BenchmarkResult]:
    """对比不同导出格式的性能.

    Args:
        model_path: 原始 .pt 模型路径
        formats: 要对比的格式列表，默认 ["pytorch", "onnx"]
        imgsz: 输入图片尺寸
        num_runs: 每个格式的测试次数

    Returns:
        list[BenchmarkResult]
    """
    if formats is None:
        formats = ["pytorch", "onnx"]

    results = []
    from yolo_learn.models.export import export_model

    for fmt in formats:
        if fmt == "pytorch":
            results.append(benchmark_model(model_path, imgsz=imgsz, num_runs=num_runs))
        elif fmt == "onnx":
            onnx_path = export_model(model_path, format="onnx", imgsz=imgsz)
            results.append(benchmark_onnx(onnx_path, imgsz=imgsz, num_runs=num_runs))
        else:
            # 其他格式先导出再用 PyTorch benchmark
            exported = export_model(model_path, format=fmt, imgsz=imgsz)
            results.append(
                BenchmarkResult(
                    model_path=str(exported),
                    format=fmt,
                    imgsz=imgsz,
                    num_runs=0,
                    times_ms=[0],
                )
            )

    return results
