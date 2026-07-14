FROM python:3.13-slim

WORKDIR /app

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制项目文件
COPY pyproject.toml README.md ./
COPY src/ src/

# 安装依赖
RUN uv pip install --system ".[serve]"

# 复制启动脚本
COPY deploy/docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
