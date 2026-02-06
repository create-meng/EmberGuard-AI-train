"""训练YOLO火灾检测模型."""

import os

# 设置使用项目本地的Ultralytics配置
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

from ultralytics import YOLO


def train():
    """训练模型."""
    model = YOLO("models/yolov8n.pt")
    model.train(data="configs/yolo_fire.yaml", workers=0, epochs=50, batch=48)


if __name__ == "__main__":
    train()
