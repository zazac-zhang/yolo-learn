"""训练模块：封装 ultralytics 训练，统一配置加载和输出管理."""

from datetime import datetime
from pathlib import Path

import yaml


def load_train_config(config_path: str | Path) -> dict:
    """加载训练配置文件.

    Args:
        config_path: YAML 配置文件路径

    Returns:
        dict: 训练参数
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def make_output_dir(base_dir: str | Path = "outputs", name: str | None = None) -> Path:
    """创建带时间戳的输出目录.

    Args:
        base_dir: 输出根目录
        name: 实验名称（可选）

    Returns:
        Path: 输出目录路径
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{name}_{timestamp}" if name else timestamp
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config_snapshot(config: dict, output_dir: Path) -> None:
    """保存配置快照到输出目录.

    Args:
        config: 训练配置
        output_dir: 输出目录
    """
    snapshot_path = output_dir / "config.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def train(
    model: str = "yolo11n.pt",
    data: str = "configs/data/coco128.yaml",
    config: str | dict | None = None,
    *,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    output_dir: str | Path = "outputs",
    name: str | None = None,
    **kwargs,
):
    """训练 YOLO 模型.

    封装 ultralytics 的训练流程，统一配置加载和输出管理。

    Args:
        model: 模型名称或路径 (如 "yolo11n.pt")
        data: 数据集配置文件路径
        config: 额外配置文件路径或 dict（可选，覆盖其他参数）
        epochs: 训练轮数
        imgsz: 输入图片尺寸
        batch: batch size
        device: 设备 ("auto", "cpu", "mps", "0", "0,1")
        output_dir: 输出根目录
        name: 实验名称
        **kwargs: 其他 ultralytics 参数

    Returns:
        训练结果对象
    """
    from ultralytics import YOLO

    # 加载额外配置
    train_config = {}
    if isinstance(config, str):
        train_config = load_train_config(config)
    elif isinstance(config, dict):
        train_config = config.copy()

    # 合并参数（显式参数覆盖配置文件）
    train_config.update(
        {
            "data": data,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
        }
    )
    train_config.update(kwargs)

    # 创建输出目录
    out_dir = make_output_dir(output_dir, name)
    train_config["project"] = str(out_dir.parent)
    train_config["name"] = out_dir.name

    # 保存配置快照
    save_config_snapshot(train_config, out_dir)

    # 训练
    yolo_model = YOLO(model)
    results = yolo_model.train(**train_config)

    return results
