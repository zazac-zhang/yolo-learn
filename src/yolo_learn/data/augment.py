"""数据增强策略：封装 ultralytics 的增强参数，提供预设配置."""

from pathlib import Path

import yaml

# 常用增强预设
AUGMENT_PRESETS = {
    "none": {},
    "light": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
    },
    "medium": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    },
    "heavy": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 15.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 5.0,
        "perspective": 0.001,
        "flipud": 0.1,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.1,
    },
}


def get_augment_params(preset: str = "medium") -> dict:
    """获取增强预设参数.

    Args:
        preset: 预设名称 ("none", "light", "medium", "heavy")

    Returns:
        dict: ultralytics 增强参数

    Raises:
        ValueError: 未知的预设名称
    """
    if preset not in AUGMENT_PRESETS:
        raise ValueError(f"未知的增强预设: {preset}. 可选: {list(AUGMENT_PRESETS.keys())}")
    return AUGMENT_PRESETS[preset].copy()


def save_augment_config(preset: str, output_path: Path) -> None:
    """将增强配置保存为 YAML 文件.

    Args:
        preset: 预设名称
        output_path: 输出文件路径
    """
    params = get_augment_params(preset)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump({"augment": preset, "params": params}, f, default_flow_style=False)
