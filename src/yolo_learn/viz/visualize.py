"""可视化工具：画 bounding box、标注、对比图等"""

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


def draw_boxes_pil(image, boxes, labels=None, scores=None, colors=None, width=2):
    """在 PIL Image 上画 bounding box.

    Args:
        image: PIL Image
        boxes: list of [x1, y1, x2, y2] (像素坐标)
        labels: list of class name strings
        scores: list of confidence scores
        colors: list of (R, G, B) tuples
        width: 边框宽度
    """
    if Image is None:
        raise ImportError("需要安装 Pillow: pip install Pillow")

    img = image.copy()
    draw = ImageDraw.Draw(img)

    default_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = (colors[i] if colors else default_colors[i % len(default_colors)])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        text_parts = []
        if labels and i < len(labels):
            text_parts.append(labels[i])
        if scores and i < len(scores):
            text_parts.append(f"{scores[i]:.2f}")

        if text_parts:
            text = " ".join(text_parts)
            draw.text((x1, y1 - 15), text, fill=color)

    return img


def yolo_to_xyxy(box, img_width, img_height):
    """将 YOLO 格式 (cx, cy, w, h) 转换为 (x1, y1, x2, y2) 像素坐标.

    Args:
        box: [x_center, y_center, width, height] 归一化坐标 (0-1)
        img_width: 图片宽度
        img_height: 图片高度
    """
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return [x1, y1, x2, y2]


def read_yolo_label(label_path, img_width, img_height):
    """读取 YOLO 格式标注文件，返回像素坐标的 boxes.

    Args:
        label_path: .txt 标注文件路径
        img_width: 图片宽度
        img_height: 图片高度

    Returns:
        list of (class_id, [x1, y1, x2, y2])
    """
    results = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            box = [float(x) for x in parts[1:5]]
            xyxy = yolo_to_xyxy(box, img_width, img_height)
            results.append((class_id, xyxy))
    return results
