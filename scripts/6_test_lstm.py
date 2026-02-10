"""
测试LSTM模型
快速测试训练好的LSTM模型在测试视频上的表现
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from emberguard.pipeline import FireDetectionPipeline


def test_on_video(video_path, model_path, yolo_path='runs/detect/train2/weights/best.pt'):
    """
    在单个视频上测试
    
    Args:
        video_path: 测试视频路径
        model_path: LSTM模型路径
        yolo_path: YOLO模型路径
    """
    print(f"\n{'='*60}")
    print(f"测试视频: {video_path}")
    print(f"{'='*60}")
    
    # 创建检测管道
    pipeline = FireDetectionPipeline(
        yolo_model_path=yolo_path,
        lstm_model_path=model_path,
        sequence_length=30
    )
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"视频信息: {total_frames} 帧, {fps} fps")
    print(f"\n开始检测...")
    
    # 统计
    frame_count = 0
    yolo_detections = 0
    lstm_predictions = {0: 0, 1: 0, 2: 0}  # 无火、烟雾、火焰
    lstm_confidences = []
    
    # 重置缓冲区
    pipeline.reset_buffer()
    
    # 处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        result = pipeline.detect_frame(frame)
        
        # 统计YOLO检测
        if result['has_detection']:
            yolo_detections += 1
        
        # 统计LSTM预测（缓冲区满后）
        if 'lstm_prediction' in result:
            pred = result['lstm_prediction']
            lstm_predictions[pred] += 1
            lstm_confidences.append(result['lstm_confidence'])
        
        frame_count += 1
        
        # 每30帧显示一次进度
        if frame_count % 30 == 0:
            print(f"  处理进度: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)", end='\r')
    
    cap.release()
    
    # 输出结果
    print(f"\n\n{'='*60}")
    print("检测结果")
    print(f"{'='*60}")
    
    print(f"\nYOLO检测:")
    print(f"  检测到火/烟的帧数: {yolo_detections}/{frame_count} ({100*yolo_detections/frame_count:.1f}%)")
    
    if lstm_confidences:
        print(f"\nLSTM预测 (缓冲区满后):")
        total_lstm = sum(lstm_predictions.values())
        print(f"  总预测次数: {total_lstm}")
        print(f"  无火 (0): {lstm_predictions[0]} ({100*lstm_predictions[0]/total_lstm:.1f}%)")
        print(f"  烟雾 (1): {lstm_predictions[1]} ({100*lstm_predictions[1]/total_lstm:.1f}%)")
        print(f"  火焰 (2): {lstm_predictions[2]} ({100*lstm_predictions[2]/total_lstm:.1f}%)")
        print(f"  平均置信度: {np.mean(lstm_confidences):.3f}")
        
        # 实时火灾检测判断逻辑：一旦检测到火焰/烟雾就报警
        has_fire = lstm_predictions[2] > 0
        has_smoke = lstm_predictions[1] > 0
        
        print(f"\n⚠️  实时火灾检测判断:")
        if has_fire:
            fire_ratio = 100 * lstm_predictions[2] / total_lstm
            print(f"  🔥 检测到火焰！({lstm_predictions[2]}次, {fire_ratio:.1f}%)")
            print(f"  ⚠️  建议：立即报警！")
        if has_smoke:
            smoke_ratio = 100 * lstm_predictions[1] / total_lstm
            print(f"  💨 检测到烟雾！({lstm_predictions[1]}次, {smoke_ratio:.1f}%)")
            if not has_fire:
                print(f"  ⚠️  建议：发出预警，密切监控！")
        
        if not has_fire and not has_smoke:
            print(f"  ✓ 未检测到火灾迹象")
            
    else:
        print(f"\n⚠️  视频太短，LSTM缓冲区未满（需要至少30帧）")


def test_on_mixed_videos():
    """测试mixed目录中的4个测试视频"""
    mixed_dir = Path("datasets/fire_videos_organized/mixed")
    
    if not mixed_dir.exists():
        print(f"❌ 测试目录不存在: {mixed_dir}")
        return
    
    # 获取所有视频
    videos = list(mixed_dir.glob("*.avi")) + list(mixed_dir.glob("*.mp4"))
    
    if not videos:
        print(f"❌ 未找到测试视频")
        return
    
    print(f"\n🔥🔥🔥 EmberGuard AI - LSTM模型测试 🔥🔥🔥")
    print(f"\n找到 {len(videos)} 个测试视频")
    
    # 测试每个视频
    model_path = "models/lstm/best.pt"
    
    if not Path(model_path).exists():
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("请先训练模型")
        return
    
    for i, video in enumerate(videos, 1):
        print(f"\n\n{'#'*60}")
        print(f"测试 {i}/{len(videos)}")
        print(f"{'#'*60}")
        
        test_on_video(str(video), model_path)
        
        input("\n按Enter键继续下一个视频...")


def test_single_video(video_path):
    """测试单个视频"""
    model_path = "models/lstm/best.pt"
    
    if not Path(model_path).exists():
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("请先训练模型")
        return
    
    if not Path(video_path).exists():
        print(f"\n❌ 视频文件不存在: {video_path}")
        return
    
    print(f"\n🔥🔥🔥 EmberGuard AI - LSTM模型测试 🔥🔥🔥")
    test_on_video(video_path, model_path)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试LSTM模型')
    parser.add_argument('--video', type=str, default=None,
                       help='测试单个视频的路径')
    parser.add_argument('--model', type=str, default='models/lstm/best.pt',
                       help='LSTM模型路径')
    
    args = parser.parse_args()
    
    if args.video:
        # 测试单个视频
        test_single_video(args.video)
    else:
        # 测试mixed目录中的所有视频
        test_on_mixed_videos()


if __name__ == "__main__":
    main()
