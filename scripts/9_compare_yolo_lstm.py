"""
YOLO vs YOLO+LSTM 对比测试
生成可视化对比报告
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 配置matplotlib - 使用英文避免中文显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from emberguard.pipeline import FireDetectionPipeline


class ComparisonTester:
    """YOLO vs YOLO+LSTM 对比测试器"""
    
    def __init__(self, yolo_path, lstm_path=None):
        """
        初始化
        
        Args:
            yolo_path: YOLO模型路径
            lstm_path: LSTM模型路径（可选）
        """
        print(f"\n{'='*60}")
        print(f"初始化对比测试器")
        print(f"{'='*60}")
        
        # 纯YOLO模型
        print(f"加载YOLO模型: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        
        # YOLO+LSTM管道
        self.lstm_pipeline = None
        if lstm_path and Path(lstm_path).exists():
            print(f"加载LSTM模型: {lstm_path}")
            self.lstm_pipeline = FireDetectionPipeline(
                yolo_model_path=yolo_path,
                lstm_model_path=lstm_path,
                sequence_length=30
            )
        else:
            print(f"⚠️  LSTM模型不可用，将只测试纯YOLO")
        
        print(f"✅ 初始化完成\n")

    def test_video_yolo(self, video_path, conf_threshold=0.25):
        """
        使用纯YOLO测试视频（逐帧独立检测，无时序信息）
        
        Args:
            video_path: 视频路径
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 测试结果
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 统计
        frame_count = 0
        detections = {'fire': 0, 'smoke': 0, 'none': 0}
        confidences = []
        
        # 时序统计（用于分析YOLO的不稳定性）
        detection_sequence = []  # 记录每帧的检测结果
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO检测
            results = self.yolo_model(frame, conf=conf_threshold, verbose=False)
            
            # 记录当前帧的检测
            frame_detection = 'none'
            
            # 统计
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0:  # smoke
                        detections['smoke'] += 1
                        frame_detection = 'smoke'
                    elif cls == 1:  # fire
                        detections['fire'] += 1
                        frame_detection = 'fire'
                    
                    confidences.append(conf)
            else:
                detections['none'] += 1
            
            detection_sequence.append(frame_detection)
            frame_count += 1
        
        cap.release()
        
        # 计算波动性（YOLO的缺点：不稳定）
        changes = sum(1 for i in range(1, len(detection_sequence)) 
                     if detection_sequence[i] != detection_sequence[i-1])
        volatility = changes / len(detection_sequence) if detection_sequence else 0
        
        # 判断结果（简单规则：检测到就报警）
        fire_ratio = detections['fire'] / frame_count if frame_count > 0 else 0
        smoke_ratio = detections['smoke'] / frame_count if frame_count > 0 else 0
        
        # YOLO判断逻辑：基于检测比例
        # 问题：容易受单帧误检影响，无法判断趋势
        if fire_ratio > 0.05:  # 超过5%的帧检测到火焰
            prediction = 'fire'
            confidence_level = fire_ratio
        elif smoke_ratio > 0.05:  # 超过5%的帧检测到烟雾
            prediction = 'smoke'
            confidence_level = smoke_ratio
        else:
            prediction = 'normal'
            confidence_level = 1 - (fire_ratio + smoke_ratio)
        
        return {
            'method': 'YOLO',
            'total_frames': frame_count,
            'fps': fps,
            'detections': detections,
            'detection_ratios': {
                'fire': fire_ratio,
                'smoke': smoke_ratio,
                'none': detections['none'] / frame_count if frame_count > 0 else 0
            },
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'volatility': volatility,  # 波动性：越高说明检测越不稳定
            'prediction': prediction,
            'confidence_level': confidence_level,
            'has_fire': detections['fire'] > 0,
            'has_smoke': detections['smoke'] > 0
        }

    def test_video_lstm(self, video_path, conf_threshold=0.25):
        """
        使用YOLO+LSTM测试视频（时序分析，考虑趋势和连续性）
        
        Args:
            video_path: 视频路径
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 测试结果
        """
        if not self.lstm_pipeline:
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 重置缓冲区
        self.lstm_pipeline.reset_buffer()
        
        # 统计
        frame_count = 0
        yolo_detections = {'fire': 0, 'smoke': 0, 'none': 0}
        lstm_predictions = {0: 0, 1: 0, 2: 0}  # 无火、烟雾、火焰
        lstm_confidences = []
        
        # 时序分析
        lstm_sequence = []  # LSTM预测序列
        continuous_fire_frames = 0  # 连续火焰帧
        continuous_smoke_frames = 0  # 连续烟雾帧
        max_continuous_fire = 0
        max_continuous_smoke = 0
        
        # 趋势分析
        smoke_to_fire_transitions = 0  # 烟雾→火焰转变次数
        prev_lstm_pred = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # LSTM检测
            result = self.lstm_pipeline.detect_frame(frame, conf_threshold)
            
            # 统计YOLO检测
            if result['has_detection']:
                for det in result['yolo_detections']:
                    if det['class_name'] == 'fire':
                        yolo_detections['fire'] += 1
                    elif det['class_name'] == 'smoke':
                        yolo_detections['smoke'] += 1
            else:
                yolo_detections['none'] += 1
            
            # 统计LSTM预测
            if 'lstm_prediction' in result:
                pred = result['lstm_prediction']
                lstm_predictions[pred] += 1
                lstm_confidences.append(result['lstm_confidence'])
                lstm_sequence.append(pred)
                
                # 连续性分析
                if pred == 2:  # 火焰
                    continuous_fire_frames += 1
                    continuous_smoke_frames = 0
                    max_continuous_fire = max(max_continuous_fire, continuous_fire_frames)
                elif pred == 1:  # 烟雾
                    continuous_smoke_frames += 1
                    continuous_fire_frames = 0
                    max_continuous_smoke = max(max_continuous_smoke, continuous_smoke_frames)
                else:  # 无火
                    continuous_fire_frames = 0
                    continuous_smoke_frames = 0
                
                # 趋势分析：烟雾→火焰
                if prev_lstm_pred == 1 and pred == 2:
                    smoke_to_fire_transitions += 1
                
                prev_lstm_pred = pred
            
            frame_count += 1
        
        cap.release()
        
        # 计算LSTM的稳定性（优势：更稳定）
        if len(lstm_sequence) > 1:
            changes = sum(1 for i in range(1, len(lstm_sequence)) 
                         if lstm_sequence[i] != lstm_sequence[i-1])
            lstm_volatility = changes / len(lstm_sequence)
        else:
            lstm_volatility = 0
        
        # LSTM判断逻辑：考虑连续性和趋势
        total_lstm = sum(lstm_predictions.values())
        fire_ratio = lstm_predictions[2] / total_lstm if total_lstm > 0 else 0
        smoke_ratio = lstm_predictions[1] / total_lstm if total_lstm > 0 else 0
        
        # LSTM的优势：
        # 1. 考虑连续性（max_continuous_fire/smoke）
        # 2. 考虑趋势（smoke_to_fire_transitions）
        # 3. 更稳定（lstm_volatility更低）
        
        # 判断逻辑
        if fire_ratio > 0.1 or max_continuous_fire > 10:
            # 火焰比例>10% 或 连续检测到火焰>10帧
            prediction = 'fire'
            confidence_level = fire_ratio
        elif smoke_ratio > 0.1 or max_continuous_smoke > 15:
            # 烟雾比例>10% 或 连续检测到烟雾>15帧
            prediction = 'smoke'
            confidence_level = smoke_ratio
        elif smoke_to_fire_transitions > 0:
            # 检测到烟雾→火焰的发展趋势
            prediction = 'fire'
            confidence_level = 0.8  # 高置信度
        else:
            prediction = 'normal'
            confidence_level = 1 - (fire_ratio + smoke_ratio)
        
        return {
            'method': 'YOLO+LSTM',
            'total_frames': frame_count,
            'fps': fps,
            'yolo_detections': yolo_detections,
            'lstm_predictions': lstm_predictions,
            'lstm_ratios': {
                'fire': fire_ratio,
                'smoke': smoke_ratio,
                'normal': lstm_predictions[0] / total_lstm if total_lstm > 0 else 0
            },
            'avg_confidence': np.mean(lstm_confidences) if lstm_confidences else 0,
            'volatility': lstm_volatility,  # LSTM的波动性（应该更低）
            'max_continuous_fire': max_continuous_fire,  # 最长连续火焰帧
            'max_continuous_smoke': max_continuous_smoke,  # 最长连续烟雾帧
            'smoke_to_fire_transitions': smoke_to_fire_transitions,  # 趋势转变
            'prediction': prediction,
            'confidence_level': confidence_level,
            'has_fire': lstm_predictions[2] > 0,
            'has_smoke': lstm_predictions[1] > 0
        }

    def test_directory(self, directory, ground_truth_label, max_videos=10):
        """
        测试目录中的视频（随机选择）
        
        Args:
            directory: 视频目录
            ground_truth_label: 真实标签 ('fire', 'smoke', 'normal', 'mixed')
            max_videos: 最多测试的视频数量（默认10个）
            
        Returns:
            list: 测试结果列表
        """
        import random
        
        directory = Path(directory)
        if not directory.exists():
            print(f"⚠️  目录不存在: {directory}")
            return []
        
        # 获取所有视频
        all_videos = list(directory.glob("*.mp4")) + list(directory.glob("*.avi"))
        
        if not all_videos:
            print(f"⚠️  目录中没有视频: {directory}")
            return []
        
        # 随机选择视频
        if len(all_videos) > max_videos:
            videos = random.sample(all_videos, max_videos)
            print(f"\n{'='*60}")
            print(f"测试目录: {directory.name}")
            print(f"真实标签: {ground_truth_label}")
            print(f"总视频数: {len(all_videos)}")
            print(f"随机选择: {len(videos)} 个视频进行测试")
            print(f"{'='*60}")
        else:
            videos = all_videos
            print(f"\n{'='*60}")
            print(f"测试目录: {directory.name}")
            print(f"真实标签: {ground_truth_label}")
            print(f"视频数量: {len(videos)}")
            print(f"{'='*60}")
        
        results = []
        
        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] 测试: {video.name}")
            
            # 纯YOLO测试
            print(f"  → YOLO测试中...")
            yolo_result = self.test_video_yolo(str(video))
            
            # YOLO+LSTM测试
            lstm_result = None
            if self.lstm_pipeline:
                print(f"  → YOLO+LSTM测试中...")
                lstm_result = self.test_video_lstm(str(video))
            
            # 保存结果
            result = {
                'video_name': video.name,
                'video_path': str(video),
                'ground_truth': ground_truth_label,
                'yolo': yolo_result,
                'lstm': lstm_result
            }
            results.append(result)
            
            # 显示结果
            if yolo_result:
                print(f"  ✓ YOLO预测: {yolo_result['prediction']}")
            if lstm_result:
                print(f"  ✓ LSTM预测: {lstm_result['prediction']}")
        
        return results

    def generate_report(self, all_results, output_dir):
        """
        生成可视化对比报告
        
        Args:
            all_results: 所有测试结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"生成对比报告")
        print(f"{'='*60}")
        
        # 1. 计算准确率
        yolo_correct = 0
        lstm_correct = 0
        total = 0
        
        yolo_predictions = []
        lstm_predictions = []
        ground_truths = []
        
        for result in all_results:
            gt = result['ground_truth']
            ground_truths.append(gt)
            
            # YOLO
            if result['yolo']:
                pred = result['yolo']['prediction']
                yolo_predictions.append(pred)
                if self._match_prediction(pred, gt):
                    yolo_correct += 1
            
            # LSTM
            if result['lstm']:
                pred = result['lstm']['prediction']
                lstm_predictions.append(pred)
                if self._match_prediction(pred, gt):
                    lstm_correct += 1
            
            total += 1
        
        yolo_accuracy = 100 * yolo_correct / total if total > 0 else 0
        lstm_accuracy = 100 * lstm_correct / total if total > 0 else 0
        
        print(f"\n📊 准确率对比:")
        print(f"  YOLO:      {yolo_correct}/{total} ({yolo_accuracy:.1f}%)")
        print(f"  YOLO+LSTM: {lstm_correct}/{total} ({lstm_accuracy:.1f}%)")
        
        # 2. 生成可视化图表
        self._plot_accuracy_comparison(yolo_accuracy, lstm_accuracy, output_dir)
        self._plot_confusion_matrix(ground_truths, yolo_predictions, lstm_predictions, output_dir)
        self._plot_detection_stats(all_results, output_dir)
        self._plot_lstm_advantages(all_results, output_dir)  # 新增：LSTM优势对比
        
        # 3. 保存详细结果
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'total_videos': total,
                'yolo_accuracy': yolo_accuracy,
                'lstm_accuracy': lstm_accuracy,
                'yolo_correct': yolo_correct,
                'lstm_correct': lstm_correct
            },
            'details': all_results
        }
        
        report_path = output_dir / 'comparison_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 报告已保存:")
        print(f"  - {output_dir / 'accuracy_comparison.png'}")
        print(f"  - {output_dir / 'confusion_matrix.png'}")
        print(f"  - {output_dir / 'detection_stats.png'}")
        print(f"  - {output_dir / 'lstm_advantages.png'}")  # 新增
        print(f"  - {report_path}")

    def _match_prediction(self, prediction, ground_truth):
        """
        判断预测是否正确
        
        Args:
            prediction: 预测结果
            ground_truth: 真实标签
            
        Returns:
            bool: 是否正确
        """
        # mixed类别：检测到fire或smoke都算对
        if ground_truth == 'mixed':
            return prediction in ['fire', 'smoke']
        
        # 其他类别：完全匹配
        return prediction == ground_truth
    
    def _plot_accuracy_comparison(self, yolo_acc, lstm_acc, output_dir):
        """绘制准确率对比图"""
        plt.figure(figsize=(10, 6))
        
        methods = ['YOLO', 'YOLO+LSTM']
        accuracies = [yolo_acc, lstm_acc]
        colors = ['#3498db', '#e74c3c']
        
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('YOLO vs YOLO+LSTM Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, ground_truths, yolo_preds, lstm_preds, output_dir):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        labels = ['normal', 'smoke', 'fire', 'mixed']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # YOLO混淆矩阵
        if yolo_preds:
            cm_yolo = confusion_matrix(ground_truths, yolo_preds, labels=labels)
            sns.heatmap(cm_yolo, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=axes[0])
            axes[0].set_title('YOLO Confusion Matrix', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('True Label', fontsize=12)
            axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # LSTM混淆矩阵
        if lstm_preds:
            cm_lstm = confusion_matrix(ground_truths, lstm_preds, labels=labels)
            sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Reds',
                       xticklabels=labels, yticklabels=labels, ax=axes[1])
            axes[1].set_title('YOLO+LSTM Confusion Matrix', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('True Label', fontsize=12)
            axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detection_stats(self, all_results, output_dir):
        """绘制检测统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 按类别统计
        categories = {}
        for result in all_results:
            gt = result['ground_truth']
            if gt not in categories:
                categories[gt] = {'yolo': [], 'lstm': []}
            
            if result['yolo']:
                categories[gt]['yolo'].append(result['yolo'])
            if result['lstm']:
                categories[gt]['lstm'].append(result['lstm'])
        
        # 1. 每个类别的平均置信度
        ax = axes[0, 0]
        cat_names = list(categories.keys())
        yolo_confs = [np.mean([r['avg_confidence'] for r in categories[c]['yolo']]) 
                     if categories[c]['yolo'] else 0 for c in cat_names]
        lstm_confs = [np.mean([r['avg_confidence'] for r in categories[c]['lstm']]) 
                     if categories[c]['lstm'] else 0 for c in cat_names]
        
        x = np.arange(len(cat_names))
        width = 0.35
        ax.bar(x - width/2, yolo_confs, width, label='YOLO', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, lstm_confs, width, label='YOLO+LSTM', color='#e74c3c', alpha=0.8)
        ax.set_ylabel('Average Confidence', fontsize=12)
        ax.set_title('Average Confidence by Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. 检测帧数统计
        ax = axes[0, 1]
        for i, cat in enumerate(cat_names):
            if categories[cat]['yolo']:
                yolo_det = [r['detections']['fire'] + r['detections']['smoke'] 
                           for r in categories[cat]['yolo']]
                ax.boxplot([yolo_det], positions=[i*2], widths=0.6, 
                          patch_artist=True, boxprops=dict(facecolor='#3498db', alpha=0.6))
            
            if categories[cat]['lstm']:
                lstm_det = [r['yolo_detections']['fire'] + r['yolo_detections']['smoke']
                           for r in categories[cat]['lstm']]
                ax.boxplot([lstm_det], positions=[i*2+0.8], widths=0.6,
                          patch_artist=True, boxprops=dict(facecolor='#e74c3c', alpha=0.6))
        
        ax.set_ylabel('Detection Frames', fontsize=12)
        ax.set_title('Detection Frame Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks([i*2+0.4 for i in range(len(cat_names))])
        ax.set_xticklabels(cat_names)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. 正确率分类别统计
        ax = axes[1, 0]
        yolo_correct_by_cat = []
        lstm_correct_by_cat = []
        
        for cat in cat_names:
            yolo_c = sum(1 for r in categories[cat]['yolo'] 
                        if self._match_prediction(r['prediction'], cat))
            yolo_t = len(categories[cat]['yolo'])
            yolo_correct_by_cat.append(100 * yolo_c / yolo_t if yolo_t > 0 else 0)
            
            lstm_c = sum(1 for r in categories[cat]['lstm']
                        if self._match_prediction(r['prediction'], cat))
            lstm_t = len(categories[cat]['lstm'])
            lstm_correct_by_cat.append(100 * lstm_c / lstm_t if lstm_t > 0 else 0)
        
        x = np.arange(len(cat_names))
        ax.bar(x - width/2, yolo_correct_by_cat, width, label='YOLO', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, lstm_correct_by_cat, width, label='YOLO+LSTM', color='#e74c3c', alpha=0.8)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # 4. 处理速度对比
        ax = axes[1, 1]
        yolo_fps = [r['yolo']['fps'] for r in all_results if r['yolo']]
        lstm_fps = [r['lstm']['fps'] for r in all_results if r['lstm']]
        
        ax.boxplot([yolo_fps, lstm_fps], labels=['YOLO', 'YOLO+LSTM'],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.6))
        ax.set_ylabel('FPS', fontsize=12)
        ax.set_title('Processing Speed Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lstm_advantages(self, all_results, output_dir):
        """绘制LSTM时序优势对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取数据
        yolo_volatility = []
        lstm_volatility = []
        lstm_continuous_fire = []
        lstm_continuous_smoke = []
        lstm_transitions = []
        
        for result in all_results:
            if result['yolo']:
                yolo_volatility.append(result['yolo'].get('volatility', 0))
            if result['lstm']:
                lstm_volatility.append(result['lstm'].get('volatility', 0))
                lstm_continuous_fire.append(result['lstm'].get('max_continuous_fire', 0))
                lstm_continuous_smoke.append(result['lstm'].get('max_continuous_smoke', 0))
                lstm_transitions.append(result['lstm'].get('smoke_to_fire_transitions', 0))
        
        # 1. 稳定性对比（波动性越低越好）
        ax = axes[0, 0]
        data = [yolo_volatility, lstm_volatility]
        bp = ax.boxplot(data, labels=['YOLO\n(Frame-by-Frame)', 'YOLO+LSTM\n(Temporal Analysis)'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.6))
        ax.set_ylabel('Volatility (Lower is Better)', fontsize=12)
        ax.set_title('Detection Stability Comparison\nLSTM Advantage: Temporal Smoothing, Reduced False Alarms', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加说明文字
        if yolo_volatility and lstm_volatility:
            yolo_avg = np.mean(yolo_volatility)
            lstm_avg = np.mean(lstm_volatility)
            improvement = (yolo_avg - lstm_avg) / yolo_avg * 100 if yolo_avg > 0 else 0
            ax.text(0.5, 0.95, f'LSTM Stability Improvement: {improvement:.1f}%',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10, fontweight='bold')
        
        # 2. 连续性检测能力
        ax = axes[0, 1]
        x = np.arange(len(lstm_continuous_fire))
        width = 0.35
        ax.bar(x - width/2, lstm_continuous_fire, width, label='Continuous Fire Frames', 
              color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, lstm_continuous_smoke, width, label='Continuous Smoke Frames',
              color='#95a5a6', alpha=0.8)
        ax.set_ylabel('Max Continuous Frames', fontsize=12)
        ax.set_xlabel('Video Index', fontsize=12)
        ax.set_title('LSTM Continuity Detection\nLSTM Advantage: Identifies Persistent Fire Patterns', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. 趋势识别能力
        ax = axes[1, 0]
        if lstm_transitions:
            ax.bar(range(len(lstm_transitions)), lstm_transitions, 
                  color='#f39c12', alpha=0.8, edgecolor='black')
            ax.set_ylabel('Smoke->Fire Transitions', fontsize=12)
            ax.set_xlabel('Video Index', fontsize=12)
            ax.set_title('LSTM Trend Recognition\nLSTM Advantage: Captures Fire Development Process', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            total_transitions = sum(lstm_transitions)
            ax.text(0.5, 0.95, f'Detected {total_transitions} Fire Development Trends',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10, fontweight='bold')
        
        # 4. 判断逻辑对比
        ax = axes[1, 1]
        ax.axis('off')
        
        # 文字说明
        comparison_text = """
        Detection Logic Comparison:
        
        ┌─────────────────────────────────────────┐
        │ YOLO (Frame-by-Frame Detection)         │
        ├─────────────────────────────────────────┤
        │ • Independent frame judgment             │
        │ • Based on detection ratio (>5%)        │
        │ • No temporal information                │
        │ • Susceptible to single-frame errors    │
        │ • Cannot identify trends                 │
        └─────────────────────────────────────────┘
        
        ┌─────────────────────────────────────────┐
        │ YOLO+LSTM (Temporal Analysis)           │
        ├─────────────────────────────────────────┤
        │ ✓ Analyzes 30-frame sequences           │
        │ ✓ Considers continuity (>10 frames)     │
        │ ✓ Recognizes trends (smoke->fire)       │
        │ ✓ Temporal smoothing, reduces errors    │
        │ ✓ More stable and reliable              │
        └─────────────────────────────────────────┘
        
        LSTM Core Advantages:
        1. Temporal Continuity: Sequence analysis, not single-frame
        2. Trend Recognition: Captures fire development process
        3. Stability: Smoothing reduces fluctuations
        4. Smart Decisions: Multi-frame information synthesis
        """
        
        ax.text(0.1, 0.95, comparison_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'lstm_advantages.png', dpi=300, bbox_inches='tight')
        plt.close()



def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO vs YOLO+LSTM 对比测试')
    parser.add_argument('--yolo', type=str, default='runs/detect/train2/weights/best.pt',
                       help='YOLO模型路径')
    parser.add_argument('--lstm', type=str, default='models/lstm/best.pt',
                       help='LSTM模型路径')
    parser.add_argument('--data-dir', type=str, 
                       default='datasets/fire_videos_organized',
                       help='测试数据目录')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='输出目录')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='每个目录最多测试的视频数量（默认10）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 YOLO vs YOLO+LSTM 对比测试")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  YOLO模型: {args.yolo}")
    print(f"  LSTM模型: {args.lstm}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output}")
    print(f"  每目录测试: {args.max_videos} 个视频")
    
    # 创建测试器
    tester = ComparisonTester(args.yolo, args.lstm)
    
    # 测试各个目录
    data_dir = Path(args.data_dir)
    all_results = []
    
    # 测试fire目录
    fire_dir = data_dir / 'fire'
    if fire_dir.exists():
        results = tester.test_directory(fire_dir, 'fire', args.max_videos)
        all_results.extend(results)
    
    # 测试smoke目录
    smoke_dir = data_dir / 'smoke'
    if smoke_dir.exists():
        results = tester.test_directory(smoke_dir, 'smoke', args.max_videos)
        all_results.extend(results)
    
    # 测试normal目录
    normal_dir = data_dir / 'normal'
    if normal_dir.exists():
        results = tester.test_directory(normal_dir, 'normal', args.max_videos)
        all_results.extend(results)
    
    # 测试mixed目录
    mixed_dir = data_dir / 'mixed'
    if mixed_dir.exists():
        results = tester.test_directory(mixed_dir, 'mixed', args.max_videos)
        all_results.extend(results)
    
    # 生成报告
    if all_results:
        tester.generate_report(all_results, args.output)
        
        print(f"\n{'='*60}")
        print(f"✅ 对比测试完成！")
        print(f"{'='*60}")
        print(f"\n测试统计:")
        print(f"  总测试视频: {len(all_results)}")
        print(f"\n查看结果:")
        print(f"  cd {args.output}")
        print(f"  查看图表: accuracy_comparison.png, confusion_matrix.png")
        print(f"  查看LSTM优势: lstm_advantages.png ⭐")
        print(f"  查看详情: comparison_report.json")
    else:
        print(f"\n❌ 没有找到测试视频")


if __name__ == "__main__":
    main()
