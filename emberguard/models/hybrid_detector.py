"""
混合检测器 - 整合YOLO和LSTM
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional
from .yolo_detector import YOLOFireDetector
from .lstm_classifier import LSTMFireClassifier


class HybridFireDetector:
    """
    YOLO + LSTM 混合火灾检测器
    结合空间特征（YOLO）和时序特征（LSTM）进行准确检测
    """
    
    def __init__(self, yolo_model_path: str, lstm_model_path: str = None,
                 seq_length: int = 30, conf_threshold: float = 0.25):
        """
        初始化混合检测器
        
        Args:
            yolo_model_path: YOLO模型路径
            lstm_model_path: LSTM模型路径（可选）
            seq_length: LSTM序列长度
            conf_threshold: YOLO置信度阈值
        """
        # 初始化YOLO检测器
        self.yolo_detector = YOLOFireDetector(yolo_model_path, conf_threshold)
        
        # 初始化LSTM分类器
        self.lstm_classifier = LSTMFireClassifier(lstm_model_path, seq_length)
        
        # 特征缓冲区 - 存储最近的特征序列
        self.feature_buffer = deque(maxlen=seq_length)
        
        # 检测状态
        self.current_prediction = "no_fire"
        self.current_confidence = 0.0
        
    def process_frame(self, frame: np.ndarray, use_lstm: bool = True) -> Tuple[np.ndarray, dict]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像 (BGR格式)
            use_lstm: 是否使用LSTM进行时序分析
            
        Returns:
            (标注后的图像, 检测结果字典)
        """
        # Step 1: YOLO检测
        detections = self.yolo_detector.detect(frame)
        
        # Step 2: 提取特征向量
        features = self.yolo_detector.extract_features_vector(detections)
        self.feature_buffer.append(features)
        
        # Step 3: LSTM时序分析（如果启用且缓冲区足够）
        if use_lstm and len(self.feature_buffer) >= self.lstm_classifier.seq_length:
            feature_seq = np.array(list(self.feature_buffer))
            pred_class, pred_conf, all_probs = self.lstm_classifier.predict(feature_seq)
            
            self.current_prediction = pred_class
            self.current_confidence = pred_conf
        else:
            # 仅使用YOLO结果
            if detections:
                best_det = max(detections, key=lambda x: x['confidence'])
                self.current_prediction = best_det['class_name']
                self.current_confidence = best_det['confidence']
            else:
                self.current_prediction = "no_fire"
                self.current_confidence = 0.0
        
        # Step 4: 绘制检测结果
        annotated_frame = self.yolo_detector.draw_detections(frame, detections)
        
        # 添加LSTM预测结果
        self._draw_lstm_prediction(annotated_frame)
        
        # 构建结果字典
        result = {
            'yolo_detections': detections,
            'lstm_prediction': self.current_prediction,
            'lstm_confidence': self.current_confidence,
            'buffer_size': len(self.feature_buffer),
            'is_fire_detected': self.current_prediction in ['fire', 'smoke']
        }
        
        return annotated_frame, result
    
    def _draw_lstm_prediction(self, frame: np.ndarray):
        """
        在图像上绘制LSTM预测结果
        
        Args:
            frame: 输入图像（会被直接修改）
        """
        # 根据预测结果选择颜色
        if self.current_prediction == 'fire':
            color = (0, 0, 255)  # 红色
            emoji = "🔥"
        elif self.current_prediction == 'smoke':
            color = (0, 255, 255)  # 黄色
            emoji = "💨"
        else:
            color = (0, 255, 0)  # 绿色
            emoji = "✅"
        
        # 绘制预测结果
        text = f"{emoji} LSTM: {self.current_prediction.upper()} ({self.current_confidence:.2f})"
        cv2.putText(frame, text, (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # 绘制缓冲区状态
        buffer_text = f"Buffer: {len(self.feature_buffer)}/{self.lstm_classifier.seq_length}"
        cv2.putText(frame, buffer_text, (30, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def reset_buffer(self):
        """重置特征缓冲区"""
        self.feature_buffer.clear()
        self.current_prediction = "no_fire"
        self.current_confidence = 0.0
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     display: bool = True) -> dict:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            display: 是否显示处理过程
            
        Returns:
            处理统计信息
        """
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 统计信息
        stats = {
            'total_frames': total_frames,
            'fire_frames': 0,
            'smoke_frames': 0,
            'no_fire_frames': 0
        }
        
        frame_count = 0
        
        print(f"🎬 开始处理视频: {video_path}")
        print(f"   分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            annotated_frame, result = self.process_frame(frame)
            
            # 更新统计
            if result['lstm_prediction'] == 'fire':
                stats['fire_frames'] += 1
            elif result['lstm_prediction'] == 'smoke':
                stats['smoke_frames'] += 1
            else:
                stats['no_fire_frames'] += 1
            
            # 写入输出视频
            if writer:
                writer.write(annotated_frame)
            
            # 显示处理过程
            if display:
                cv2.imshow("EmberGuard AI - Fire Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⚠️  用户中断处理")
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"✅ 视频处理完成!")
        print(f"   火焰帧: {stats['fire_frames']}, 烟雾帧: {stats['smoke_frames']}, 正常帧: {stats['no_fire_frames']}")
        
        return stats
    
    def process_webcam(self, camera_id: int = 0):
        """
        处理摄像头实时视频流
        
        Args:
            camera_id: 摄像头ID（默认0为主摄像头）
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            return
        
        print(f"📹 开始实时检测 (摄像头 {camera_id})")
        print("   按 'q' 退出, 按 'r' 重置缓冲区")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            # 处理帧
            annotated_frame, result = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow("EmberGuard AI - Live Detection", annotated_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_buffer()
                print("🔄 已重置特征缓冲区")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 实时检测结束")
