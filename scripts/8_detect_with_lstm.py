"""
YOLO+LSTM火灾检测脚本
支持图片、视频、摄像头检测
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from emberguard.pipeline import FireDetectionPipeline


def detect_image(pipeline, image_path, output_path=None, show=True):
    """
    检测单张图片
    
    Args:
        pipeline: 检测管道
        image_path: 图片路径
        output_path: 输出路径（可选）
        show: 是否显示结果
    """
    print(f"\n{'='*60}")
    print(f"检测图片: {image_path}")
    print(f"{'='*60}")
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    # 重置缓冲区
    pipeline.reset_buffer()
    
    # 检测（重复30次填充缓冲区）
    for _ in range(30):
        result = pipeline.detect_frame(img)
    
    # 打印结果
    print(f"\n🔥 检测结果:")
    print(f"{'='*60}")
    
    # YOLO检测
    if result['yolo_detections']:
        print(f"\n📹 YOLO检测:")
        for det in result['yolo_detections']:
            print(f"  - {det['class_name']}: 置信度 {det['confidence']:.3f}")
    else:
        print(f"\n📹 YOLO检测: 未检测到火/烟")
    
    # LSTM预测
    if 'lstm_prediction' in result:
        print(f"\n🧠 LSTM预测:")
        print(f"  - 类别: {result['lstm_class_name']}")
        print(f"  - 置信度: {result['lstm_confidence']:.3f}")
        print(f"  - 概率分布:")
        for name, prob in result['lstm_probabilities'].items():
            print(f"    {name}: {prob:.3f}")
    else:
        print(f"\n🧠 LSTM预测: 缓冲区未满（需要30帧）")
    
    # 绘制结果
    img_vis = pipeline._draw_results(img, result)
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, img_vis)
        print(f"\n💾 结果已保存: {output_path}")
    
    # 显示结果
    if show:
        cv2.imshow('EmberGuard AI - Detection Result', img_vis)
        print(f"\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_video(pipeline, video_path, output_path=None, show=True):
    """
    检测视频
    
    Args:
        pipeline: 检测管道
        video_path: 视频路径
        output_path: 输出路径（可选）
        show: 是否显示结果
    """
    print(f"\n{'='*60}")
    print(f"检测视频: {video_path}")
    print(f"{'='*60}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 视频信息:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧率: {fps} fps")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {total_frames/fps:.1f} 秒")
    
    # 输出视频
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 重置缓冲区
    pipeline.reset_buffer()
    
    # 统计
    frame_count = 0
    lstm_predictions = {0: 0, 1: 0, 2: 0}  # 无火、烟雾、火焰
    
    print(f"\n🚀 开始检测...")
    print(f"{'='*60}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        result = pipeline.detect_frame(frame)
        
        # 统计LSTM预测
        if 'lstm_prediction' in result:
            pred = result['lstm_prediction']
            lstm_predictions[pred] += 1
        
        # 绘制结果
        frame_vis = pipeline._draw_results(frame, result)
        
        # 保存
        if writer:
            writer.write(frame_vis)
        
        # 显示
        if show:
            cv2.imshow('EmberGuard AI - Video Detection', frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n⚠️  用户中断检测")
                break
        
        frame_count += 1
        
        # 显示进度
        if frame_count % 30 == 0:
            progress = 100 * frame_count / total_frames
            print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')
    
    # 释放资源
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # 输出统计
    print(f"\n\n{'='*60}")
    print(f"🔥 检测完成")
    print(f"{'='*60}")
    print(f"\n📊 统计结果:")
    print(f"  - 处理帧数: {frame_count}")
    
    if lstm_predictions[0] + lstm_predictions[1] + lstm_predictions[2] > 0:
        total_pred = sum(lstm_predictions.values())
        print(f"\n🧠 LSTM预测分布:")
        print(f"  - 无火: {lstm_predictions[0]} ({100*lstm_predictions[0]/total_pred:.1f}%)")
        print(f"  - 烟雾: {lstm_predictions[1]} ({100*lstm_predictions[1]/total_pred:.1f}%)")
        print(f"  - 火焰: {lstm_predictions[2]} ({100*lstm_predictions[2]/total_pred:.1f}%)")
        
        # 最终判断
        final_pred = max(lstm_predictions, key=lstm_predictions.get)
        class_names = {0: "无火", 1: "烟雾", 2: "火焰"}
        print(f"\n✅ 最终判断: {class_names[final_pred]} (出现 {lstm_predictions[final_pred]} 次)")
    
    if output_path:
        print(f"\n💾 结果已保存: {output_path}")


def detect_camera(pipeline, camera_id=0):
    """
    检测摄像头
    
    Args:
        pipeline: 检测管道
        camera_id: 摄像头ID
    """
    print(f"\n{'='*60}")
    print(f"检测摄像头: {camera_id}")
    print(f"{'='*60}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头: {camera_id}")
        return
    
    # 摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n📹 摄像头信息:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧率: {fps} fps")
    
    # 重置缓冲区
    pipeline.reset_buffer()
    
    print(f"\n🚀 开始实时检测...")
    print(f"{'='*60}")
    print(f"按 'q' 退出，按 's' 截图保存")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n❌ 无法读取摄像头画面")
            break
        
        # 检测
        result = pipeline.detect_frame(frame)
        
        # 绘制结果
        frame_vis = pipeline._draw_results(frame, result)
        
        # 添加帧数信息
        cv2.putText(frame_vis, f"Frame: {frame_count}", (10, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示
        cv2.imshow('EmberGuard AI - Camera Detection (Press Q to quit, S to save)', frame_vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n⚠️  用户退出")
            break
        elif key == ord('s'):
            # 保存截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"detection_saves/camera_{timestamp}.jpg"
            Path("detection_saves").mkdir(exist_ok=True)
            cv2.imwrite(save_path, frame_vis)
            print(f"\n💾 截图已保存: {save_path}")
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"🔥 检测结束")
    print(f"{'='*60}")
    print(f"总帧数: {frame_count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EmberGuard AI - YOLO+LSTM火灾检测')
    parser.add_argument('--source', type=str, required=True,
                       help='输入源：图片路径、视频路径、摄像头ID（0,1,2...）')
    parser.add_argument('--yolo', type=str, default='runs/detect/train2/weights/best.pt',
                       help='YOLO模型路径')
    parser.add_argument('--lstm', type=str, default='models/lstm/train/best.pt',
                       help='LSTM模型路径（可选，不指定则只用YOLO）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径（可选）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示结果窗口')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='YOLO置信度阈值')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 EmberGuard AI - 火灾检测系统")
    print("=" * 60)
    print(f"\n📋 配置:")
    print(f"  - YOLO模型: {args.yolo}")
    print(f"  - LSTM模型: {args.lstm if args.lstm else '未使用'}")
    print(f"  - 输入源: {args.source}")
    print(f"  - 置信度阈值: {args.conf}")
    
    # 检查LSTM模型
    lstm_path = args.lstm if args.lstm and Path(args.lstm).exists() else None
    if args.lstm and not lstm_path:
        print(f"\n⚠️  LSTM模型不存在: {args.lstm}")
        print(f"将只使用YOLO检测")
    
    # 创建检测管道
    print(f"\n🔧 初始化检测管道...")
    pipeline = FireDetectionPipeline(
        yolo_model_path=args.yolo,
        lstm_model_path=lstm_path,
        sequence_length=30
    )
    
    # 判断输入类型
    source = args.source
    
    # 摄像头
    if source.isdigit():
        detect_camera(pipeline, int(source))
    
    # 图片
    elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        if not Path(source).exists():
            print(f"\n❌ 图片不存在: {source}")
            return
        detect_image(pipeline, source, args.output, not args.no_show)
    
    # 视频
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        if not Path(source).exists():
            print(f"\n❌ 视频不存在: {source}")
            return
        detect_video(pipeline, source, args.output, not args.no_show)
    
    else:
        print(f"\n❌ 不支持的输入格式: {source}")
        print(f"支持的格式:")
        print(f"  - 图片: .jpg, .jpeg, .png, .bmp, .webp")
        print(f"  - 视频: .mp4, .avi, .mov, .mkv, .flv")
        print(f"  - 摄像头: 0, 1, 2...")


if __name__ == "__main__":
    main()
