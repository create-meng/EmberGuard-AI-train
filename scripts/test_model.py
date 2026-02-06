"""
测试YOLO火灾检测模型
支持图片、视频和摄像头.
"""

import os

# 设置使用项目本地的Ultralytics配置
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

import argparse

from ultralytics import YOLO


def test_model(source, model_path="models/yolov8n.pt", save=True, conf=0.25):
    """测试模型.

    Args:
        source: 输入源
            - 图片路径: 'image.jpg'
            - 视频路径: 'video.mp4'
            - 摄像头: 0 (默认摄像头) 或 1, 2...
            - 图片文件夹: 'path/to/images/'
        model_path: 模型路径
        save: 是否保存结果
        conf: 置信度阈值
    """
    model = YOLO(model_path, task="detect")
    results = model(source=source, save=save, conf=conf)
    return results


def main():
    parser = argparse.ArgumentParser(description="测试YOLO火灾检测模型")
    parser.add_argument("--source", type=str, default=0, help="输入源: 图片/视频路径, 摄像头编号(0), 或文件夹路径")
    parser.add_argument("--model", type=str, default="runs/detect/train2/weights/best.pt", help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (0-1)")
    parser.add_argument("--no-save", action="store_true", help="不保存检测结果")

    args = parser.parse_args()

    # 如果source是数字字符串，转换为整数（摄像头编号）
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    print(f"使用模型: {args.model}")
    print(f"输入源: {source}")
    print(f"置信度阈值: {args.conf}")

    test_model(source=source, model_path=args.model, save=not args.no_save, conf=args.conf)


if __name__ == "__main__":
    main()
