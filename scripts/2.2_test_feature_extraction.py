"""
测试特征提取效果
在开始完整训练前，先测试YOLO模型和特征提取器
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import random

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from emberguard.feature_extractor import FeatureExtractor
from ultralytics import YOLO


def test_yolo_detection():
    """测试YOLO模型检测效果 - 测试多个视频"""
    print("=" * 60)
    print("测试1: YOLO模型检测效果（每类测试5个视频）")
    print("=" * 60)
    
    # 加载模型
    print("\n加载YOLO模型...")
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    # 测试不同类型的视频
    test_videos = {
        'fire': 'datasets/fire_videos_organized/fire',
        'smoke': 'datasets/fire_videos_organized/smoke',
        'normal': 'datasets/fire_videos_organized/normal'
    }
    
    all_results = {}
    
    for category, video_dir in test_videos.items():
        video_dir = Path(video_dir)
        if not video_dir.exists():
            continue
        
        # 获取所有视频
        videos = list(video_dir.glob('*.avi')) + list(video_dir.glob('*.mp4'))
        if not videos:
            continue
        
        # 随机选择5个视频测试
        num_test_videos = min(5, len(videos))
        test_videos_list = random.sample(videos, num_test_videos)
        
        print(f"\n{'='*60}")
        print(f"测试 {category} 类别 (从 {len(videos)} 个视频中随机选择 {num_test_videos} 个)")
        print(f"{'='*60}")
        
        category_results = []
        
        for idx, test_video in enumerate(test_videos_list, 1):
            print(f"\n[{idx}/{num_test_videos}] {test_video.name}")
            
            # 读取视频
            cap = cv2.VideoCapture(str(test_video))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 随机采样50帧进行测试
            test_frame_count = min(50, total_frames)
            test_frames = sorted(random.sample(range(total_frames), test_frame_count))
            
            detections = []
            confidences = []
            
            for frame_idx in test_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO检测
                results = model(frame, verbose=False)
                
                if len(results[0].boxes) > 0:
                    detections.append(True)
                    # 获取最高置信度
                    max_conf = max([float(box.conf[0]) for box in results[0].boxes])
                    confidences.append(max_conf)
                else:
                    detections.append(False)
            
            cap.release()
            
            # 统计
            detection_rate = sum(detections) / len(detections) * 100 if detections else 0
            avg_conf = np.mean(confidences) if confidences else 0
            min_conf = min(confidences) if confidences else 0
            max_conf = max(confidences) if confidences else 0
            
            result = {
                'video': test_video.name,
                'total_frames': total_frames,
                'frames_tested': len(detections),
                'detection_rate': detection_rate,
                'avg_confidence': avg_conf,
                'min_confidence': min_conf,
                'max_confidence': max_conf,
                'detections_count': sum(detections)
            }
            category_results.append(result)
            
            print(f"  总帧数: {total_frames}, 测试: {len(detections)}帧, 检测: {sum(detections)}帧")
            print(f"  检测率: {detection_rate:.2f}%")
            print(f"  置信度: 平均={avg_conf:.3f}, 范围=[{min_conf:.3f}, {max_conf:.3f}]")
        
        # 计算该类别的平均统计
        avg_detection_rate = np.mean([r['detection_rate'] for r in category_results])
        avg_confidence = np.mean([r['avg_confidence'] for r in category_results if r['avg_confidence'] > 0])
        
        all_results[category] = {
            'videos_tested': num_test_videos,
            'individual_results': category_results,
            'avg_detection_rate': avg_detection_rate,
            'avg_confidence': avg_confidence
        }
        
        print(f"\n{category} 类别平均:")
        print(f"  平均检测率: {avg_detection_rate:.2f}%")
        print(f"  平均置信度: {avg_confidence:.3f}")
    
    return all_results
    return results_summary


def test_feature_extraction():
    """测试特征提取效果"""
    print("\n" + "=" * 60)
    print("测试2: 特征提取效果")
    print("=" * 60)
    
    # 加载模型和特征提取器
    model = YOLO('runs/detect/train2/weights/best.pt')
    extractor = FeatureExtractor()
    
    # 随机选择一个火灾视频
    fire_dir = Path('datasets/fire_videos_organized/fire')
    videos = list(fire_dir.glob('*.avi')) + list(fire_dir.glob('*.mp4'))
    
    if not videos:
        print("未找到测试视频")
        return
    
    test_video = random.choice(videos)
    print(f"\n测试视频: {test_video.name}")
    print(f"  (从 {len(videos)} 个火灾视频中随机选择)")
    
    # 读取视频
    cap = cv2.VideoCapture(str(test_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 随机采样60帧进行特征提取测试
    test_frame_count = min(60, total_frames)
    test_frames = sorted(random.sample(range(total_frames), test_frame_count))
    
    # 提取特征
    features_list = []
    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO检测
        results = model(frame, verbose=False)
        
        # 提取特征
        features = extractor.get_best_detection(results, frame.shape)
        features_list.append(features)
    
    cap.release()
    
    # 分析特征
    features_array = np.array(features_list)
    
    print(f"\n提取的特征序列:")
    print(f"  序列长度: {len(features_list)}")
    print(f"  特征维度: {features_array.shape}")
    print(f"\n特征统计:")
    print(f"  中心点x (cx): {features_array[:, 0].mean():.3f} ± {features_array[:, 0].std():.3f}")
    print(f"  中心点y (cy): {features_array[:, 1].mean():.3f} ± {features_array[:, 1].std():.3f}")
    print(f"  宽度 (w): {features_array[:, 2].mean():.3f} ± {features_array[:, 2].std():.3f}")
    print(f"  高度 (h): {features_array[:, 3].mean():.3f} ± {features_array[:, 3].std():.3f}")
    print(f"  面积 (area): {features_array[:, 4].mean():.3f} ± {features_array[:, 4].std():.3f}")
    print(f"  宽高比 (ratio): {features_array[:, 5].mean():.3f} ± {features_array[:, 5].std():.3f}")
    print(f"  置信度 (conf): {features_array[:, 6].mean():.3f} ± {features_array[:, 6].std():.3f}")
    print(f"  类别 (cls): {features_array[:, 7].mean():.3f}")
    
    # 检查特征是否有效
    print(f"\n特征有效性检查:")
    
    # 检查是否有检测
    has_detection = (features_array[:, 6] > 0).sum()
    detection_rate = has_detection / len(features_list) * 100
    print(f"  有检测的帧数: {has_detection}/{len(features_list)}")
    print(f"  检测率: {detection_rate:.2f}%")
    
    # 检查特征变化
    feature_variance = features_array.var(axis=0)
    print(f"  特征方差: {feature_variance}")
    
    if detection_rate >= 40:
        print(f"\n✅ 特征提取效果良好！")
        return True
    else:
        print(f"\n⚠️  检测率较低，可能影响训练效果")
        return False


def test_sequence_generation():
    """测试序列生成"""
    print("\n" + "=" * 60)
    print("测试3: 序列生成测试")
    print("=" * 60)
    
    # 直接从3_prepare_lstm_data导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_lstm_data", Path(__file__).parent / "3_prepare_lstm_data.py")
    prepare_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_module)
    LSTMDataPreparer = prepare_module.LSTMDataPreparer
    
    # 初始化
    preparer = LSTMDataPreparer(
        yolo_model_path='runs/detect/train2/weights/best.pt',
        sequence_length=30
    )
    
    # 随机选择一个视频
    fire_dir = Path('datasets/fire_videos_organized/fire')
    videos = list(fire_dir.glob('*.avi')) + list(fire_dir.glob('*.mp4'))
    
    if not videos:
        print("未找到测试视频")
        return
    
    test_video = random.choice(videos)
    print(f"\n测试视频: {test_video.name}")
    
    # 提取特征
    print("提取特征...")
    features = preparer.extract_features_from_video(str(test_video), stride=5)
    
    print(f"\n提取结果:")
    print(f"  特征向量数: {len(features)}")
    
    # 创建序列
    if len(features) >= 30:
        sequences, labels = preparer.create_sequences(features, label=2)
        print(f"  生成序列数: {len(sequences)}")
        print(f"  序列形状: {sequences.shape}")
        print(f"\n✅ 序列生成成功！")
        return True
    else:
        print(f"\n⚠️  特征数量不足，无法生成序列")
        return False


def estimate_training_time():
    """估算训练时间"""
    print("\n" + "=" * 60)
    print("测试4: 训练时间估算")
    print("=" * 60)
    
    import time
    # 直接从3_prepare_lstm_data导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_lstm_data", Path(__file__).parent / "3_prepare_lstm_data.py")
    prepare_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_module)
    LSTMDataPreparer = prepare_module.LSTMDataPreparer
    
    # 初始化
    preparer = LSTMDataPreparer(
        yolo_model_path='runs/detect/train2/weights/best.pt',
        sequence_length=30
    )
    
    # 随机选择一个视频测试处理时间
    fire_dir = Path('datasets/fire_videos_organized/fire')
    videos = list(fire_dir.glob('*.avi')) + list(fire_dir.glob('*.mp4'))
    
    if not videos:
        print("未找到测试视频")
        return
    
    test_video = random.choice(videos)
    
    # 获取视频信息
    cap = cv2.VideoCapture(str(test_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"\n测试视频: {test_video.name}")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps} fps")
    print(f"  时长: {duration:.1f} 秒")
    
    # 测试处理时间
    print(f"\n开始计时...")
    start_time = time.time()
    
    features = preparer.extract_features_from_video(str(test_video), stride=5)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n处理时间: {elapsed_time:.1f} 秒")
    print(f"处理速度: {total_frames/elapsed_time:.1f} 帧/秒")
    
    # 估算总时间
    total_videos = 240
    avg_video_time = elapsed_time
    estimated_total_time = avg_video_time * total_videos
    
    print(f"\n估算总时间:")
    print(f"  单个视频: {avg_video_time:.1f} 秒")
    print(f"  240个视频: {estimated_total_time/60:.1f} 分钟 ({estimated_total_time/3600:.1f} 小时)")
    
    return estimated_total_time


def main():
    """主函数"""
    print("\n" + "🔥" * 30)
    print("EmberGuard AI - 特征提取测试")
    print("🔥" * 30)
    
    print("\n这个测试会:")
    print("1. 测试YOLO模型在不同类别视频上的检测效果")
    print("2. 测试特征提取器的效果")
    print("3. 测试序列生成")
    print("4. 估算完整训练所需时间")
    
    input("\n按Enter键开始测试...")
    
    # 测试1: YOLO检测
    try:
        yolo_results = test_yolo_detection()
    except Exception as e:
        print(f"\n❌ YOLO测试失败: {e}")
        return
    
    # 测试2: 特征提取
    try:
        feature_ok = test_feature_extraction()
    except Exception as e:
        print(f"\n❌ 特征提取测试失败: {e}")
        return
    
    # 测试3: 序列生成
    try:
        sequence_ok = test_sequence_generation()
    except Exception as e:
        print(f"\n❌ 序列生成测试失败: {e}")
        return
    
    # 测试4: 时间估算
    try:
        estimated_time = estimate_training_time()
    except Exception as e:
        print(f"\n❌ 时间估算失败: {e}")
        estimated_time = None
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    print("\nYOLO检测效果（每类测试5个视频的平均值）:")
    for category, result in yolo_results.items():
        print(f"  {category}:")
        print(f"    平均检测率: {result['avg_detection_rate']:.2f}%")
        print(f"    平均置信度: {result['avg_confidence']:.3f}")
        print(f"    测试视频数: {result['videos_tested']}")
    
    # 评估模型质量
    print("\n模型质量评估:")
    fire_rate = yolo_results.get('fire', {}).get('avg_detection_rate', 0)
    smoke_rate = yolo_results.get('smoke', {}).get('avg_detection_rate', 0)
    normal_rate = yolo_results.get('normal', {}).get('avg_detection_rate', 0)
    
    issues = []
    if fire_rate < 70:
        issues.append(f"⚠️  火灾检测率偏低 ({fire_rate:.1f}%)")
    if smoke_rate < 70:
        issues.append(f"⚠️  烟雾检测率偏低 ({smoke_rate:.1f}%)")
    if normal_rate > 20:
        issues.append(f"⚠️  正常视频误报率偏高 ({normal_rate:.1f}%)")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ 模型性能良好")
    
    if feature_ok and sequence_ok:
        print("\n✅ 所有测试通过！")
        print("\n建议:")
        print("  - YOLO模型工作正常")
        print("  - 特征提取有效")
        if normal_rate > 20:
            print("  - 注意：正常视频有一定误报，LSTM需要学习区分")
        print("  - 可以开始完整训练")
        
        if estimated_time:
            print(f"\n预计数据准备时间: {estimated_time/3600:.1f} 小时")
        
        print("\n下一步:")
        print("  python scripts/3_prepare_lstm_data.py")
    else:
        print("\n⚠️  部分测试未通过")
        print("建议检查YOLO模型和数据质量")


if __name__ == "__main__":
    main()
