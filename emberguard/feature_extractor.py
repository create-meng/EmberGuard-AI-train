"""
特征提取器 - 从YOLO检测结果提取8维特征向量
"""
import numpy as np


class FeatureExtractor:
    """
    从YOLO检测结果提取特征
    
    特征向量 (8维):
    1. cx - 中心点x坐标 (归一化)
    2. cy - 中心点y坐标 (归一化)
    3. w - 宽度 (归一化)
    4. h - 高度 (归一化)
    5. area - 面积 (归一化)
    6. aspect_ratio - 宽高比
    7. conf - 置信度
    8. cls - 类别ID
    """
    
    def __init__(self, img_width=640, img_height=640):
        """
        初始化特征提取器
        
        Args:
            img_width: 图像宽度（用于归一化）
            img_height: 图像高度（用于归一化）
        """
        self.img_width = img_width
        self.img_height = img_height
    
    def extract(self, detection, img_shape=None):
        """
        从单个检测结果提取特征
        
        Args:
            detection: YOLO detection object (box)
            img_shape: 图像形状 (height, width)，如果提供则用于归一化
            
        Returns:
            np.array: 8维特征向量
        """
        # 获取边界框坐标
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        
        # 如果提供了图像形状，使用实际尺寸
        if img_shape is not None:
            img_h, img_w = img_shape[:2]
        else:
            img_w, img_h = self.img_width, self.img_height
        
        # 计算几何特征
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # 归一化坐标和尺寸
        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        area_norm = area / (img_w * img_h)
        
        # 获取检测特征
        conf = float(detection.conf[0])
        cls = int(detection.cls[0])
        
        # 组合特征向量
        features = np.array([
            cx_norm,
            cy_norm,
            w_norm,
            h_norm,
            area_norm,
            aspect_ratio,
            conf,
            cls
        ], dtype=np.float32)
        
        return features
    
    def extract_from_results(self, results, img_shape=None):
        """
        从YOLO结果中提取所有检测的特征
        
        Args:
            results: YOLO results object
            img_shape: 图像形状
            
        Returns:
            list: 特征向量列表
        """
        features_list = []
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                features = self.extract(box, img_shape)
                features_list.append(features)
        
        return features_list
    
    def get_best_detection(self, results, img_shape=None):
        """
        获取置信度最高的检测的特征
        
        Args:
            results: YOLO results object
            img_shape: 图像形状
            
        Returns:
            np.array: 8维特征向量，如果没有检测则返回零向量
        """
        if len(results[0].boxes) > 0:
            # 找到置信度最高的检测
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            return self.extract(best_box, img_shape)
        else:
            # 没有检测，返回零向量
            return np.zeros(8, dtype=np.float32)


if __name__ == "__main__":
    # 测试代码
    from ultralytics import YOLO
    import cv2
    
    print("测试特征提取器...")
    
    # 加载模型
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    # 加载测试图片
    img_path = 'test_picture/1.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法加载图片: {img_path}")
        exit(1)
    
    # 检测
    results = model(img, verbose=False)
    
    # 提取特征
    extractor = FeatureExtractor()
    
    # 方法1: 提取所有检测
    all_features = extractor.extract_from_results(results, img.shape)
    print(f"\n检测到 {len(all_features)} 个目标")
    for i, feat in enumerate(all_features):
        print(f"目标 {i+1} 特征: {feat}")
    
    # 方法2: 只提取最佳检测
    best_features = extractor.get_best_detection(results, img.shape)
    print(f"\n最佳检测特征: {best_features}")
    
    print("\n✅ 特征提取器测试完成！")
