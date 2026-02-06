"""
YOLO火灾检测器 - 空间特征提取
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple


class YOLOFireDetector:
    """
    基于YOLOv8的火灾检测器
    负责从单帧图像中提取空间特征
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型权重路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = ['fire', 'smoke']  # 根据训练数据集调整
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测单帧图像中的火灾/烟雾
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个结果包含边界框和特征
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                # 提取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                # 计算几何特征
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # 获取置信度和类别
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # 提取ROI区域的颜色特征
                roi = frame[y1:y2, x1:x2]
                color_features = self._extract_color_features(roi)
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'size': (w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'confidence': conf,
                    'class': cls,
                    'class_name': self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}',
                    'color_features': color_features
                }
                
                detections.append(detection)
                
        return detections
    
    def _extract_color_features(self, roi: np.ndarray) -> Dict[str, float]:
        """
        从ROI区域提取颜色特征
        火焰通常具有高红色、高饱和度特征
        
        Args:
            roi: 感兴趣区域图像
            
        Returns:
            颜色特征字典
        """
        if roi.size == 0:
            return {'mean_red': 0, 'mean_saturation': 0, 'mean_value': 0}
        
        # BGR转HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 计算平均值
        mean_red = np.mean(roi[:, :, 2])  # BGR中的R通道
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        return {
            'mean_red': float(mean_red),
            'mean_saturation': float(mean_saturation),
            'mean_value': float(mean_value)
        }
    
    def extract_features_vector(self, detections: List[Dict]) -> np.ndarray:
        """
        将检测结果转换为特征向量，用于LSTM输入
        
        Args:
            detections: 检测结果列表
            
        Returns:
            特征向量 (如果没有检测到，返回零向量)
        """
        if not detections:
            # 没有检测到目标，返回零向量
            return np.zeros(11)  # 11个特征
        
        # 取置信度最高的检测结果
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        features = [
            best_detection['center'][0],  # cx
            best_detection['center'][1],  # cy
            best_detection['size'][0],    # width
            best_detection['size'][1],    # height
            best_detection['area'],       # area
            best_detection['aspect_ratio'], # aspect_ratio
            best_detection['confidence'], # confidence
            best_detection['class'],      # class
            best_detection['color_features']['mean_red'],
            best_detection['color_features']['mean_saturation'],
            best_detection['color_features']['mean_value']
        ]
        
        return np.array(features, dtype=np.float32)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            绘制了检测框的图像
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            
            # 根据类别选择颜色
            color = (0, 0, 255) if class_name == 'fire' else (0, 255, 255)  # 火焰红色，烟雾黄色
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
