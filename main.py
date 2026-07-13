"""YOLO Learn - 主入口."""


def main():
    print("=" * 60)
    print("  YOLO Learn - 系统学习 YOLO 目标检测")
    print("=" * 60)
    print()
    print("  学习路径:")
    print("  1. 打开 notebooks/ 目录")
    print("  2. 按编号顺序学习 01-10 的 Notebook")
    print("  3. 每个 Notebook 都有详细的注释和练习")
    print()
    print("  快速开始:")
    print("  $ uv run jupyter notebook notebooks/01_what_is_object_detection.ipynb")
    print()
    print("  训练模型:")
    print("  $ uv run python scripts/train.py --epochs 10")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
