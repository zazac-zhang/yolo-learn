"""测试共享 fixtures."""

import pytest


@pytest.fixture
def tmp_dataset(tmp_path):
    """创建临时数据集结构."""
    # 创建目录
    train_imgs = tmp_path / "train" / "images"
    train_labels = tmp_path / "train" / "labels"
    val_imgs = tmp_path / "val" / "images"
    val_labels = tmp_path / "val" / "labels"
    for d in [train_imgs, train_labels, val_imgs, val_labels]:
        d.mkdir(parents=True)

    # 创建空图片文件
    (train_imgs / "img001.jpg").touch()
    (val_imgs / "img002.jpg").touch()

    # 创建标注文件 (YOLO 格式)
    (train_labels / "img001.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    (val_labels / "img002.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    # 创建 data.yaml
    import yaml

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "path": str(tmp_path),
                "train": "train/images",
                "val": "val/images",
                "nc": 1,
                "names": {0: "object"},
            }
        )
    )

    return tmp_path


@pytest.fixture
def tmp_model_path(tmp_path):
    """返回不存在的模型路径（用于测试错误处理）."""
    return tmp_path / "nonexistent" / "model.pt"


@pytest.fixture
def sample_boxes():
    """示例边界框数据 (xyxy 格式)."""
    return [
        [10, 10, 50, 50],
        [30, 30, 70, 70],
        [100, 100, 150, 150],
    ]


@pytest.fixture
def sample_scores():
    """示例置信度分数."""
    return [0.9, 0.8, 0.7]


@pytest.fixture
def sample_annotations():
    """示例标注数据."""
    return [
        {"class_id": 0, "box": [10, 10, 50, 50]},
        {"class_id": 1, "box": [100, 100, 150, 150]},
    ]


@pytest.fixture
def sample_predictions():
    """示例预测数据."""
    return [
        {"class_id": 0, "box": [12, 12, 52, 52], "confidence": 0.95},
        {"class_id": 1, "box": [100, 100, 150, 150], "confidence": 0.85},
    ]
