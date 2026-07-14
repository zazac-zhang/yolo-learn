"""Tests for yolo_learn.models.benchmark."""

from yolo_learn.models.benchmark import BenchmarkResult


class TestBenchmarkResult:
    def test_avg_ms(self):
        result = BenchmarkResult(
            model_path="test.pt", format="pytorch", imgsz=640, num_runs=3,
            times_ms=[10.0, 20.0, 30.0],
        )
        assert result.avg_ms == 20.0

    def test_p50_ms(self):
        result = BenchmarkResult(
            model_path="test.pt", format="pytorch", imgsz=640, num_runs=3,
            times_ms=[10.0, 20.0, 30.0],
        )
        assert result.p50_ms == 20.0

    def test_p95_ms(self):
        result = BenchmarkResult(
            model_path="test.pt", format="pytorch", imgsz=640, num_runs=100,
            times_ms=list(range(1, 101)),
        )
        assert result.p95_ms > 90

    def test_throughput_fps(self):
        result = BenchmarkResult(
            model_path="test.pt", format="pytorch", imgsz=640, num_runs=1,
            times_ms=[10.0],
        )
        assert result.throughput_fps == 100.0

    def test_throughput_zero_latency(self):
        result = BenchmarkResult(
            model_path="test.pt", format="pytorch", imgsz=640, num_runs=1,
            times_ms=[0.0],
        )
        assert result.throughput_fps == 0

    def test_summary(self):
        result = BenchmarkResult(
            model_path="test.pt", format="pytorch", imgsz=640, num_runs=1,
            times_ms=[10.0],
        )
        summary = result.summary()
        assert "test.pt" in summary
        assert "pytorch" in summary
        assert "10.00" in summary
