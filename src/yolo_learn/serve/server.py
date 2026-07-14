"""推理服务：FastAPI REST API."""

from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="YOLO Learn 推理服务", version="0.1.0")
except ImportError:
    app = None

# 全局模型实例
_model = None
_model_path = None


def _get_model():
    """延迟加载模型."""
    global _model, _model_path
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO("yolo11n.pt")
    return _model


def _set_model(model_path: str | Path):
    """设置模型."""
    global _model, _model_path
    from ultralytics import YOLO

    _model = YOLO(str(model_path))
    _model_path = str(model_path)


def predict_image(image_bytes: bytes, conf: float = 0.25) -> dict:
    """单张图片推理.

    Args:
        image_bytes: 图片字节
        conf: 置信度阈值

    Returns:
        检测结果 dict
    """
    import io

    from PIL import Image

    model = _get_model()
    image = Image.open(io.BytesIO(image_bytes))
    results = model.predict(image, conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        detections.append(
            {
                "class_id": int(box.cls[0]),
                "class_name": results[0].names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": box.xyxy[0].tolist(),
            }
        )

    return {
        "detections": detections,
        "count": len(detections),
        "model_path": _model_path,
    }


# FastAPI 路由（仅在 FastAPI 可用时注册）
if app is not None:

    @app.post("/predict")
    async def predict_endpoint(file: UploadFile = File(...), conf: float = 0.25):
        """单张图片推理."""
        contents = await file.read()
        result = predict_image(contents, conf=conf)
        return JSONResponse(content=result)

    @app.post("/predict/batch")
    async def predict_batch_endpoint(files: list[UploadFile] = File(...), conf: float = 0.25):
        """批量推理."""
        results = []
        for file in files:
            contents = await file.read()
            results.append(predict_image(contents, conf=conf))
        return JSONResponse(content={"results": results, "count": len(results)})

    @app.get("/health")
    async def health():
        """健康检查."""
        return {"status": "ok", "model_path": _model_path}

    @app.get("/model/info")
    async def model_info():
        """模型信息."""
        return {"model_path": _model_path, "loaded": _model is not None}


def start_server(
    model_path: str | Path = "yolo11n.pt",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """启动推理服务.

    Args:
        model_path: 模型路径
        host: 监听地址
        port: 监听端口
    """
    import uvicorn

    _set_model(model_path)
    print(f"🚀 启动推理服务: http://{host}:{port}")
    print(f"   模型: {model_path}")
    print(f"   文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)
