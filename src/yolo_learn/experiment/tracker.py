"""实验追踪：MLflow 封装."""

from __future__ import annotations

from pathlib import Path


def start_run(name: str, tags: dict | None = None):
    """开始实验运行.

    Args:
        name: 实验名称
        tags: 标签 (如 {"model": "yolo11n", "dataset": "coco128"})

    Returns:
        mlflow.ActiveRun
    """
    import mlflow

    return mlflow.start_run(run_name=name, tags=tags)


def log_params(params: dict) -> None:
    """记录超参数.

    Args:
        params: 参数字典
    """
    import mlflow

    mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """记录指标.

    Args:
        metrics: 指标字典
        step: 步骤号 (如 epoch)
    """
    import mlflow

    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    """记录产物文件 (模型、图表等).

    Args:
        path: 文件路径
        artifact_path: artifact 子目录
    """
    import mlflow

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_model(model_path: str | Path, name: str = "model") -> None:
    """记录模型产物.

    Args:
        model_path: 模型文件路径
        name: 模型名称
    """
    import mlflow

    mlflow.log_artifact(str(model_path), artifact_path=name)


def get_best_run(
    experiment_name: str,
    metric: str = "metrics/mAP50",
    direction: str = "max",
) -> dict | None:
    """获取最佳实验运行.

    Args:
        experiment_name: 实验名称
        metric: 排序指标
        direction: "max" 或 "min"

    Returns:
        最佳 run 的信息 dict，无结果返回 None
    """
    import mlflow

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        order = "DESC" if direction == "max" else "ASC"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"{metric} {order}"],
            max_results=1,
        )

        if runs.empty:
            return None

        run = runs.iloc[0]
        return {
            "run_id": run["run_id"],
            "metrics": {k: run[k] for k in runs.columns if k.startswith("metrics/")},
            "params": {k: run[k] for k in runs.columns if k.startswith("params/")},
        }
    except Exception:
        return None
