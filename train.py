import os
import sys
from ultralytics import YOLO
import torch


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(root_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"data.yaml 未找到: {data_yaml}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 选择小而快的模型：yolov8n
    model = YOLO("yolov8n.pt")

    # 可通过环境变量调整训练参数
    epochs = int(os.environ.get("EPOCHS", "5"))
    imgsz = int(os.environ.get("IMGSZ", "640"))
    batch = int(os.environ.get("BATCH", "16" if torch.cuda.is_available() else "8"))

    project = os.path.join(root_dir, "runs", "train")
    name = "logo_yolov8n"

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,  # Windows 更稳定
        cache=True,
        project=project,
        name=name,
        exist_ok=True,
        patience=10,
        verbose=True,
    )

    # 保存最佳权重路径到文件，便于后续推理脚本读取
    best = getattr(model.trainer, "best", None)
    if best:
        best_str = str(best)
        if os.path.exists(best_str):
            print(f"最佳权重: {best_str}")
            with open(os.path.join(root_dir, "best_model.txt"), "w", encoding="utf-8") as f:
                f.write(best_str)
        else:
            print(f"未找到最佳权重文件: {best_str}")
    else:
        print("未找到最佳权重，可能训练未完成或路径变更。")

    # 使用验证集做一次评估
    try:
        model.val(data=data_yaml, split="val", device=device)
    except Exception as e:
        print(f"验证失败: {e}")


if __name__ == "__main__":
    main()