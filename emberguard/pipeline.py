"""
火灾检测管道 - 集成YOLO和LSTM
"""
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from .feature_extractor import FeatureExtractor
from .lstm_model import LSTMFireClassifier


class FireDetectionPipeline:
    """
    火灾检测管道
    
    集成YOLO目标检测和LSTM时序分析
    """
    
    def __init__(self, yolo_model_path, lstm_model_path=None, sequence_length=30, device=None):
        """
        初始化检测管道
        
        Args:
            yolo_model_path: YOLO模型路径
            lstm_model_path: LSTM模型路径（可选）
            sequence_length: 时序长度（帧数）
            device: 计算设备
        """
        # 设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载YOLO模型
        print(f"加载YOLO模型: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # LSTM模型（可选）
        self.lstm_model = None
        self.use_lstm = False
        if lstm_model_path is not None:
            try:
                from .lstm_model import LSTMTrainer
                self.lstm_model = LSTMTrainer.load_model(lstm_model_path, self.device)
                self.use_lstm = True
                print(f"LSTM模型已加载")
            except Exception as e:
                print(f"LSTM模型加载失败: {e}")
                print("将仅使用YOLO检测")
        
        # 时序缓冲区
        self.sequence_length = sequence_length
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # 类别映射
        self.class_names = {
            0: "无火",
            1: "烟雾", 
            2: "火焰"
        }
    
    def reset_buffer(self):
        """重置特征缓冲区"""
        self.feature_buffer.clear()
    
    def detect_frame(self, frame, conf_threshold=0.25):
        """
        检测单帧
        
        Args:
            frame: 输入图像
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 检测结果
        """
        # YOLO检测
        results = self.yolo_model(frame, conf=conf_threshold, verbose=False)
        
        # 提取特征
        features = self.feature_extractor.get_best_detection(results, frame.shape)
        
        # 添加到缓冲区
        self.feature_buffer.append(features)
        
        # 基础检测结果
        yolo_detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                yolo_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': cls,
                    'class_name': 'fire' if cls == 0 else 'smoke'
                })
        
        result = {
            'yolo_detections': yolo_detections,
            'has_detection': len(yolo_detections) > 0,
            'features': features,
            'buffer_size': len(self.feature_buffer)
        }
        
        # LSTM分析（如果可用且缓冲区已满）
        if self.use_lstm and len(self.feature_buffer) == self.sequence_length:
            lstm_result = self._lstm_classify()
            result.update(lstm_result)
        
        return result
    
    def _lstm_classify(self):
        """
        使用LSTM进行时序分类
        
        Returns:
            dict: LSTM分类结果
        """
        # 准备序列数据
        sequence = np.array(list(self.feature_buffer))  # (seq_len, 8)
        
        # 预测
        pred_class, probs = self.lstm_model.predict(sequence)
        
        pred_class = pred_class[0]
        probs = probs[0]
        
        return {
            'lstm_prediction': int(pred_class),
            'lstm_class_name': self.class_names[pred_class],
            'lstm_probabilities': {
                '无火': float(probs[0]),
                '烟雾': float(probs[1]),
                '火焰': float(probs[2])
            },
            'lstm_confidence': float(probs[pred_class])
        }
    
    def process_video(self, video_path, output_path=None, conf_threshold=0.25, show_progress=True):
        """
        处理视频
        
        Args:
            video_path: 视频路径
            output_path: 输出路径（可选）
            conf_threshold: 置信度阈值
            show_progress: 显示进度
            
        Returns:
            list: 每帧的检测结果
        """
        import cv2
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        
        # 输出视频（可选）
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 重置缓冲区
        self.reset_buffer()
        
        # 处理每一帧
        results_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            result = self.detect_frame(frame, conf_threshold)
            result['frame_idx'] = frame_idx
            results_list.append(result)
            
            # 绘制结果（如果需要输出视频）
            if writer is not None:
                frame_vis = self._draw_results(frame, result)
                writer.write(frame_vis)
            
            # 显示进度
            if show_progress and frame_idx % 30 == 0:
                print(f"处理进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
            
            frame_idx += 1
        
        # 释放资源
        cap.release()
        if writer is not None:
            writer.release()
            print(f"输出视频已保存: {output_path}")
        
        return results_list
    
    def _draw_results(self, frame, result):
        """
        在帧上绘制检测结果
        
        Args:
            frame: 输入帧
            result: 检测结果
            
        Returns:
            绘制后的帧
        """
        import cv2
        
        frame_vis = frame.copy()
        
        # 绘制YOLO检测框
        for det in result['yolo_detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            
            # 绘制框
            color = (0, 0, 255) if cls_name == 'fire' else (0, 255, 255)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame_vis, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制LSTM结果（如果有）
        if 'lstm_prediction' in result:
            lstm_text = f"LSTM: {result['lstm_class_name']} ({result['lstm_confidence']:.2f})"
            cv2.putText(frame_vis, lstm_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_vis
