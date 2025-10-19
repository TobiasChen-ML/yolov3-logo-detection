import os
import glob
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from ultralytics import YOLO


def find_best_weights(root_dir: str) -> Optional[str]:
    """
    在项目根目录下查找最新的 best.pt 权重文件。
    优先返回最新修改时间的权重路径；若找不到则返回 None。
    """
    pattern = os.path.join(root_dir, "runs", "train", "**", "weights", "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def infer_image(
    image_path: str,
    weights: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用 YOLO .pt 权重对单张图片进行推理，返回每个检测框的类别、名称、置信度和像素坐标（xyxy）。

    参数：
    - image_path: 图片路径。
    - weights: 模型权重路径，若为 None 则自动查找 runs/train/**/weights/best.pt，找不到则回退到 yolov8n.pt。
    - conf: 置信度阈值。
    - iou: IOU 阈值。
    - imgsz: 推理图像尺寸（按需缩放）。
    - device: 指定设备，例如 'cuda' 或 'cpu'；默认自动选择。

    返回：
    - 列表，元素为字典：{
        'class_id': int,
        'class_name': str,
        'confidence': float,
        'bbox_xyxy': [x1, y1, x2, y2]
      }
    若无检测结果，返回空列表。
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # 选择权重
    selected_weights = (
        weights
        or find_best_weights(root_dir)
        or os.path.join(root_dir, "yolov8n.pt")
    )
    print(selected_weights)
    # 选择设备
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = YOLO(selected_weights)

    # 推理（不保存可视化图）
    results = model.predict(
        source=image_path,
        save=False,
        conf=conf,
        iou=iou,
        device=dev,
        imgsz=imgsz,
        verbose=False,
    )

    if not results:
        return []

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else np.array([])
    clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") and boxes.cls is not None else np.array([])

    names = getattr(res, "names", None) or getattr(model, "names", {})

    outputs: List[Dict[str, Any]] = []
    for i in range(len(xyxy)):
        c = int(clss[i]) if clss.size > i else -1
        name = names.get(c, str(c)) if isinstance(names, dict) else (names[c] if isinstance(names, (list, tuple)) and c >= 0 and c < len(names) else str(c))
        box = [float(v) for v in xyxy[i].tolist()]
        conf_v = float(confs[i]) if confs.size > i else float("nan")
        outputs.append(
            {
                "class_id": c,
                "class_name": name,
                "confidence": conf_v,
                "bbox_xyxy": box,
            }
        )

    return outputs


__all__ = ["infer_image"]


print(infer_image('sample.jpg'))