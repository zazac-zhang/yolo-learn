"""数据集下载工具：从 Roboflow 或直接 URL 下载数据集."""

from pathlib import Path


def download_coco128(target_dir: str | Path = "data") -> Path:
    """下载 COCO128 数据集.

    使用 ultralytics 内置的下载功能。

    Args:
        target_dir: 目标目录

    Returns:
        Path: 数据集根目录路径
    """
    from ultralytics.utils.downloads import download

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = "https://ultralytics.com/assets/coco128.zip"
    download(url, dir=target_dir / "coco128", unzip=True)

    return target_dir / "coco128"


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    format: str = "yolov8",
    target_dir: str | Path = "data",
) -> Path:
    """从 Roboflow 下载数据集.

    Args:
        api_key: Roboflow API key
        workspace: 工作空间名称
        project: 项目名称
        version: 版本号
        format: 导出格式 (默认 yolov8)
        target_dir: 目标目录

    Returns:
        Path: 数据集根目录路径
    """
    from roboflow import Roboflow

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)
    dataset = version_obj.download(format, location=str(target_dir))

    return Path(dataset.location)
