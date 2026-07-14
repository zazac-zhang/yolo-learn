#!/bin/bash
# 启动推理服务
# 环境变量:
#   MODEL_PATH - 模型路径 (默认 yolo11n.pt)
#   HOST       - 监听地址 (默认 0.0.0.0)
#   PORT       - 监听端口 (默认 8000)

MODEL_PATH=${MODEL_PATH:-yolo11n.pt}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

exec python -c "
from yolo_learn.serve import start_server
start_server('${MODEL_PATH}', '${HOST}', ${PORT})
"
