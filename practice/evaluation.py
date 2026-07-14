"""
自动评估脚本 - 检查挑战完成情况.

使用方法:
    uv run python practice/evaluation.py [model_path]

评估标准:
    - 及格: mAP@0.5 > 0.50
    - 良好: mAP@0.5 > 0.60
    - 优秀: mAP@0.5 > 0.65
"""

import sys
from pathlib import Path

from ultralytics import YOLO

from yolo_learn.eval.evaluate import load_class_names


CLASS_NAMES = {0: "helmet", 1: "person", 2: "vest"}


def evaluate(model_path="runs/safety_helmet/weights/best.pt"):
    """评估训练好的模型."""
    model_path = Path(model_path)
    data_yaml = Path(__file__).parent / "data" / "safety_helmet" / "data.yaml"

    # 检查文件
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先完成训练")
        return False
    if not data_yaml.exists():
        print(f"❌ 数据集配置不存在: {data_yaml}")
        print("   请先运行: uv run python practice/download.py")
        return False

    print("=" * 50)
    print("🏗️  安全帽检测挑战 - 评估")
    print("=" * 50)

    # 加载模型并评估
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), device="auto", verbose=False)

    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    print(f"\n📊 评估结果:")
    print(f"   mAP@0.5   : {map50:.4f}")
    print(f"   mAP@0.5:0.95: {map50_95:.4f}")
    print(f"   Precision : {metrics.box.mp:.4f}")
    print(f"   Recall    : {metrics.box.mr:.4f}")

    # 各类别 AP
    print(f"\n📋 各类别 AP@0.5:")
    for i, ap in enumerate(metrics.box.ap50):
        name = CLASS_NAMES.get(i, f"class_{i}")
        status = "✅" if ap > 0.6 else "⚠️" if ap > 0.4 else "❌"
        print(f"   {status} {name:<10} {ap:.4f}")

    # 评级
    print(f"\n{'=' * 50}")
    if map50 >= 0.65:
        print("🏆 优秀！mAP@0.5 > 0.65")
        print("   你已经掌握了 YOLO 训练和调优技巧")
    elif map50 >= 0.60:
        print("🟢 良好！mAP@0.5 > 0.60")
        print("   尝试分析混淆矩阵，进一步提升")
    elif map50 >= 0.50:
        print("🟡 及格！mAP@0.5 > 0.50")
        print("   建议：检查类不平衡问题，调整增强策略")
    else:
        print("❌ 未及格，mAP@0.5 < 0.50")
        print("   建议：")
        print("   1. 检查数据集是否正确加载")
        print("   2. 增加训练轮数 (epochs)")
        print("   3. 使用预训练权重 (yolo11n.pt)")
    print("=" * 50)

    return map50 >= 0.50


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/safety_helmet/weights/best.pt"
    success = evaluate(model_path)
    sys.exit(0 if success else 1)
