"""Tests for yolo_learn.serve — API endpoints."""

import pytest

from yolo_learn.serve import app, predict_image


class TestPredictImage:
    def test_predict_returns_dict(self):
        """predict_image 返回正确结构."""
        import io

        from PIL import Image

        # 创建测试图片
        img = Image.new("RGB", (640, 640), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = predict_image(image_bytes, conf=0.25)

        assert "detections" in result
        assert "count" in result
        assert isinstance(result["detections"], list)
        assert isinstance(result["count"], int)


class TestAPI:
    @pytest.fixture
    def client(self):
        """创建测试客户端."""
        if app is None:
            pytest.skip("FastAPI not installed")
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health(self, client):
        """健康检查端点."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_model_info(self, client):
        """模型信息端点."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data
