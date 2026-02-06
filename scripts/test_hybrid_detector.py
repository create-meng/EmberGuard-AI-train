"""
测试YOLO+LSTM混合检测器
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from emberguard.models.hybrid_detector import HybridFireDetector


def test_image(detector, image_path):
    """测试单张图片"""
    import cv2
    
    print(f"\n📸 测试图片: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    annotated_frame, result = detector.process_frame(frame, use_lstm=False)
    
    print(f"   YOLO检测: {len(result['yolo_detections'])} 个目标")
    print(f"   预测结果: {result['lstm_prediction']} (置信度: {result['lstm_confidence']:.2f})")
    
    # 显示结果
    cv2.imshow("Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(detector, video_path, output_path=None):
    """测试视频"""
    print(f"\n🎬 测试视频: {video_path}")
    stats = detector.process_video(video_path, output_path, display=True)
    
    print(f"\n📊 检测统计:")
    print(f"   总帧数: {stats['total_frames']}")
    print(f"   火焰帧: {stats['fire_frames']} ({stats['fire_frames']/stats['total_frames']*100:.1f}%)")
    print(f"   烟雾帧: {stats['smoke_frames']} ({stats['smoke_frames']/stats['total_frames']*100:.1f}%)")
    print(f"   正常帧: {stats['no_fire_frames']} ({stats['no_fire_frames']/stats['total_frames']*100:.1f}%)")


def test_webcam(detector):
    """测试摄像头"""
    print(f"\n📹 测试摄像头实时检测")
    detector.process_webcam(camera_id=0)


def main():
    """主函数"""
    print("=" * 60)
    print("EmberGuard AI - YOLO+LSTM混合检测器测试")
    print("=" * 60)
    
    # 配置路径
    yolo_model_path = "runs/detect/train2/weights/best.pt"  # 你训练的YOLO模型
    lstm_model_path = None  # 暂时没有训练LSTM模型
    
    # 检查YOLO模型是否存在
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO模型不存在: {yolo_model_path}")
        print("   请先训练YOLO模型或修改路径")
        return
    
    # 初始化检测器
    print(f"\n🚀 初始化检测器...")
    print(f"   YOLO模型: {yolo_model_path}")
    print(f"   LSTM模型: {lstm_model_path if lstm_model_path else '未加载（将创建新模型）'}")
    
    detector = HybridFireDetector(
        yolo_model_path=yolo_model_path,
        lstm_model_path=lstm_model_path,
        seq_length=30,
        conf_threshold=0.25
    )
    
    print("\n✅ 检测器初始化完成!")
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 测试图片")
    print("2. 测试视频")
    print("3. 测试摄像头")
    print("4. 退出")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == '1':
        # 测试图片
        test_images = [
            "1.png",
            "2.jpg",
            "3.jpg",
            "4.jpg"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image(detector, img_path)
            else:
                print(f"⏭️  跳过不存在的图片: {img_path}")
    
    elif choice == '2':
        # 测试视频
        video_path = input("请输入视频路径: ").strip()
        if os.path.exists(video_path):
            output_path = "detection_saves/output_video.mp4"
            test_video(detector, video_path, output_path)
        else:
            print(f"❌ 视频文件不存在: {video_path}")
    
    elif choice == '3':
        # 测试摄像头
        test_webcam(detector)
    
    elif choice == '4':
        print("👋 再见!")
    
    else:
        print("❌ 无效选项")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
