"""训练流程集成测试 — 测试 train 函数的外部行为."""

from unittest.mock import MagicMock, patch

from yolo_learn.models.train import train


class TestTrainFunction:
    """train() 函数测试."""

    @patch("ultralytics.YOLO")
    def test_train_returns_results(self, mock_yolo_cls, tmp_path, tmp_dataset):
        """训练返回结果对象."""
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = str(tmp_path / "runs" / "exp1")
        mock_model.train.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        result = train(
            model="yolo11n.pt",
            data=str(tmp_dataset / "data.yaml"),
            epochs=1,
            output_dir=str(tmp_path / "outputs"),
            name="exp1",
        )

        assert result is not None
        mock_model.train.assert_called_once()

    @patch("ultralytics.YOLO")
    def test_train_passes_params(self, mock_yolo_cls, tmp_path, tmp_dataset):
        """训练正确传递参数."""
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = str(tmp_path / "runs")
        mock_model.train.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        train(
            model="yolo11n.pt",
            data=str(tmp_dataset / "data.yaml"),
            epochs=5,
            batch=4,
            imgsz=320,
            output_dir=str(tmp_path / "outputs"),
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["epochs"] == 5
        assert call_kwargs["batch"] == 4
        assert call_kwargs["imgsz"] == 320

    @patch("ultralytics.YOLO")
    def test_train_creates_output_dir(self, mock_yolo_cls, tmp_path, tmp_dataset):
        """训练创建输出目录."""
        mock_model = MagicMock()
        mock_results = MagicMock()
        out_dir = tmp_path / "outputs" / "test_run"
        mock_results.save_dir = str(out_dir)
        mock_model.train.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        train(
            model="yolo11n.pt",
            data=str(tmp_dataset / "data.yaml"),
            epochs=1,
            output_dir=str(tmp_path / "outputs"),
            name="test_run",
        )

        # 验证 train() 被调用时包含 project 和 name
        call_kwargs = mock_model.train.call_args[1]
        assert "project" in call_kwargs
        assert "name" in call_kwargs
