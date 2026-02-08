"""
快速测试 - 测试LSTM模型在测试图片上的效果
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from emberguard.feature_extractor import FeatureExtractor
from emberguard.lstm_model import LSTMTrainer
from ultralytics import YOLO


def quick_test():
    """快速测试LSTM模型"""
    print("\n" + "🔥" * 30)
    print("EmberGuard AI - LSTM快速测试")
    print("🔥" * 30)
    
    # 加载模型
    print("\n加载模型...")
    yolo_model = YOLO('runs/detect/train2/weights/best.pt')
    lstm_model = LSTMTrainer.load_model('models/lstm/best.pt')
    feature_extractor = FeatureExtractor()
    
    print("✅ 模型加载完成")
    
    # 测试图片
    test_images = [
        'test_picture/1.png',
        'test_picture/2.jpg',
        'test_picture/3.jpg',
        'test_picture/4.jpg'
    ]
    
    class_names = {0: "无火", 1: "烟雾", 2: "火焰"}
    
    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"\n⚠️  图片不存在: {img_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"测试图片: {img_path}")
        print(f"{'='*60}")
        
        # 读取图片
        img = cv2.imread(img_path)
        
        # YOLO检测
        results = yolo_model(img, verbose=False)
        
        # 提取特征
        features = feature_extractor.get_best_detection(results, img.shape)
        
        print(f"\nYOLO检测:")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = 'fire' if cls == 0 else 'smoke'
                print(f"  检测到: {cls_name}, 置信度: {conf:.3f}")
        else:
            print(f"  未检测到火/烟")
        
        # 创建30帧序列（重复当前特征）
        sequence = np.array([features] * 30)
        
        # LSTM预测
        pred_class, probs = lstm_model.predict(sequence)
        pred_class = pred_class[0]
        probs = probs[0]
        
        print(f"\nLSTM预测:")
        print(f"  预测类别: {class_names[pred_class]}")
        print(f"  置信度: {probs[pred_class]:.3f}")
        print(f"  概率分布:")
        print(f"    无火: {probs[0]:.3f}")
        print(f"    烟雾: {probs[1]:.3f}")
        print(f"    火焰: {probs[2]:.3f}")


def test_video_sample():
    """测试视频采样（从视频中采样30帧）"""
    print("\n" + "🔥" * 30)
    print("EmberGuard AI - 视频采样测试")
    print("🔥" * 30)
    
    # 选择一个测试视频
    test_video = "datasets/fire_videos_organized/mixed/archive_fire and smoke.mp4"
    
    if not Path(test_video).exists():
        print(f"\n❌ 测试视频不存在: {test_video}")
        # 尝试其他视频
        mixed_dir = Path("datasets/fire_videos_organized/mixed")
        videos = list(mixed_dir.glob("*.mp4")) + list(mixed_dir.glob("*.avi"))
        if videos:
            test_video = str(videos[0])
            print(f"使用: {test_video}")
        else:
            print("未找到测试视频")
            return
    
    print(f"\n测试视频: {test_video}")
    
    # 加载模型
    print("\n加载模型...")
    yolo_model = YOLO('runs/detect/train2/weights/best.pt')
    lstm_model = LSTMTrainer.load_model('models/lstm/best.pt')
    feature_extractor = FeatureExtractor()
    
    # 打开视频
    cap = cv2.VideoCapture(test_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频总帧数: {total_frames}")
    
    # 采样30帧
    sample_indices = np.linspace(0, total_frames-1, 30, dtype=int)
    features_list = []
    
    print("\n提取特征...")
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO检测
        results = yolo_model(frame, verbose=False)
        
        # 提取特征
        features = feature_extractor.get_best_detection(results, frame.shape)
        features_list.append(features)
    
    cap.release()
    
    if len(features_list) < 30:
        print(f"⚠️  只提取了 {len(features_list)} 帧")
        return
    
    # 创建序列
    sequence = np.array(features_list)
    
    # LSTM预测
    pred_class, probs = lstm_model.predict(sequence)
    pred_class = pred_class[0]
    probs = probs[0]
    
    class_names = {0: "无火", 1: "烟雾", 2: "火焰"}
    
    print(f"\n{'='*60}")
    print("LSTM预测结果")
    print(f"{'='*60}")
    print(f"预测类别: {class_names[pred_class]}")
    print(f"置信度: {probs[pred_class]:.3f}")
    print(f"\n概率分布:")
    print(f"  无火: {probs[0]:.3f}")
    print(f"  烟雾: {probs[1]:.3f}")
    print(f"  火焰: {probs[2]:.3f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='快速测试LSTM模型')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video'],
                       help='测试模式: image(图片) 或 video(视频采样)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'image':
            quick_test()
        else:
            test_video_sample()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
