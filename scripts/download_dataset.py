"""下载数据集脚本 - 薄 CLI 壳，核心逻辑在 yolo_learn.data.download."""

import argparse

from yolo_learn.data.download import download_coco128, download_roboflow_dataset


def main():
    parser = argparse.ArgumentParser(description="下载 YOLO 数据集")
    subparsers = parser.add_subparsers(dest="command", help="数据集来源")

    # COCO128
    coco_parser = subparsers.add_parser("coco128", help="下载 COCO128 数据集")
    coco_parser.add_argument("--target-dir", default="data", help="目标目录")

    # Roboflow
    rf_parser = subparsers.add_parser("roboflow", help="从 Roboflow 下载数据集")
    rf_parser.add_argument("--api-key", required=True, help="Roboflow API key")
    rf_parser.add_argument("--workspace", required=True, help="工作空间名称")
    rf_parser.add_argument("--project", required=True, help="项目名称")
    rf_parser.add_argument("--version", type=int, required=True, help="版本号")
    rf_parser.add_argument("--target-dir", default="data", help="目标目录")

    args = parser.parse_args()

    if args.command == "coco128":
        path = download_coco128(args.target_dir)
        print(f"数据集下载完成: {path}")
    elif args.command == "roboflow":
        path = download_roboflow_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            target_dir=args.target_dir,
        )
        print(f"数据集下载完成: {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
