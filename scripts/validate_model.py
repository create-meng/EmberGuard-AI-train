"""
验证YOLO火灾检测模型性能
"""
import os

# 设置使用项目本地的Ultralytics配置
os.environ['ULTRALYTICS_CONFIG_DIR'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')

from ultralytics import YOLO

def validate():
    """在验证集上验证模型"""
    # 加载训练好的模型
    model = YOLO("runs/detect/train2/weights/best.pt")
    
    # 在验证集上进行验证
    results = model.val(data="configs/yolo_fire.yaml")
    
    # 打印验证结果
    print(f"\n验证结果:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

if __name__ == "__main__":
    validate()
