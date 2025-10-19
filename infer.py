import os
import sys
import glob
from ultralytics import YOLO
import torch


def find_best_weights(root_dir: str):
    pattern = os.path.join(root_dir, "runs", "train", "**", "weights", "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # 默认对验证集推理，可传入 source 参数覆盖
    source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root_dir, "data", "images", "val")

    best_weights = find_best_weights(root_dir)
    weights = best_weights if best_weights else "yolov8n.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用权重: {weights}")
    print(f"使用设备: {device}")

    model = YOLO(weights)

    project = os.path.join(root_dir, "runs", "predict")
    name = "logo_val"

    conf = float(os.environ.get("CONF", "0.25"))
    iou = float(os.environ.get("IOU", "0.45"))
    imgsz = int(os.environ.get("IMGSZ", "640"))

    results = model.predict(
        source=source,
        save=True,
        conf=conf,
        iou=iou,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        imgsz=imgsz,
    )

    try:
        save_dir = results[0].save_dir if results else os.path.join(project, name)
        print(f"预测结果已保存到: {save_dir}")
    except Exception:
        pass

    # 打印检测类别与坐标（xyxy）
    try:
        if results:
            res = results[0]
            names = getattr(res, "names", None) or getattr(model, "names", {})
            import numpy as np
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                print("无检测结果")
            else:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0,4))
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else np.array([])
                clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") and boxes.cls is not None else np.array([])
                if len(xyxy) == 0:
                    print("无检测结果")
                else:
                    print("检测结果（cls, name, conf, x1,y1,x2,y2）:")
                    for i in range(len(xyxy)):
                        c = int(clss[i]) if clss.size > i else -1
                        name = names.get(c, str(c)) if isinstance(names, dict) else str(c)
                        box = [float(v) for v in xyxy[i].tolist()]
                        conf = float(confs[i]) if confs.size > i else float('nan')
                        print(f"{c}\t{name}\t{conf:.3f}\t{box}")
    except Exception as e:
        print(f"解析检测结果失败: {e}")


if __name__ == "__main__":
    main()