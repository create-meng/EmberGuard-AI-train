# EmberGuard AI 核心模块

## 📦 模块结构

```
emberguard/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── yolo_detector.py       # YOLO空间特征检测器
│   ├── lstm_classifier.py     # LSTM时序分类器
│   └── hybrid_detector.py     # YOLO+LSTM混合检测器
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 基础使用 - 仅YOLO检测

```python
from emberguard.models import YOLOFireDetector
import cv2

# 初始化检测器
detector = YOLOFireDetector("runs/detect/train2/weights/best.pt")

# 检测图片
frame = cv2.imread("test.jpg")
detections = detector.detect(frame)

# 绘制结果
annotated = detector.draw_detections(frame, detections)
cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

### 2. 高级使用 - YOLO+LSTM混合检测

```python
from emberguard.models import HybridFireDetector

# 初始化混合检测器
detector = HybridFireDetector(
    yolo_model_path="runs/detect/train2/weights/best.pt",
    lstm_model_path="models/lstm_fire_model.h5",  # 可选
    seq_length=30,
    conf_threshold=0.25
)

# 处理视频
detector.process_video(
    video_path="test_video.mp4",
    output_path="output.mp4",
    display=True
)

# 或处理摄像头
detector.process_webcam(camera_id=0)
```

### 3. 训练LSTM模型

```python
from emberguard.models import LSTMFireClassifier
import numpy as np

# 创建分类器
classifier = LSTMFireClassifier(seq_length=30, num_features=11)

# 准备训练数据
# X_train shape: (n_samples, 30, 11)
# y_train shape: (n_samples, 3) - one-hot编码

# 训练
history = classifier.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32
)

# 保存模型
classifier.save("models/lstm_fire_model.h5")
```

## 🔧 核心组件说明

### YOLOFireDetector

**功能：** 从单帧图像中检测火灾/烟雾，提取空间特征

**特征提取：**
- 几何特征：中心坐标、宽高、面积、长宽比
- 检测特征：置信度、类别
- 颜色特征：平均红色值、饱和度、亮度

**输出：** 11维特征向量

### LSTMFireClassifier

**功能：** 分析连续30帧的特征序列，判断火灾类型

**架构：**
- 2层LSTM (128→64单元)
- Batch Normalization
- Dropout防止过拟合
- 3分类输出：no_fire, smoke, fire

**优势：** 能够区分短暂闪光、炊烟等误报场景

### HybridFireDetector

**功能：** 整合YOLO和LSTM，提供完整的检测流程

**工作流程：**
1. YOLO检测每帧的火焰/烟雾
2. 提取11维特征向量
3. 维护30帧的特征缓冲区
4. LSTM分析时序特征
5. 输出最终判断结果

**优势：**
- 结合空间和时间信息
- 降低误报率
- 适合实时视频流处理

## 📊 特征说明

每帧提取的11个特征：

| 索引 | 特征名称 | 说明 |
|------|---------|------|
| 0 | cx | 检测框中心X坐标 |
| 1 | cy | 检测框中心Y坐标 |
| 2 | width | 检测框宽度 |
| 3 | height | 检测框高度 |
| 4 | area | 检测框面积 |
| 5 | aspect_ratio | 长宽比 |
| 6 | confidence | YOLO置信度 |
| 7 | class | 类别ID (0=fire, 1=smoke) |
| 8 | mean_red | ROI平均红色值 |
| 9 | mean_saturation | ROI平均饱和度 |
| 10 | mean_value | ROI平均亮度 |

## 🎯 下一步计划

1. **数据收集：** 收集包含火灾、烟雾、炊烟、闪光等场景的视频数据
2. **特征标注：** 为每个30帧序列标注类别（no_fire/smoke/fire）
3. **模型训练：** 训练LSTM分类器
4. **性能优化：** 调整超参数，提升准确率
5. **部署测试：** 在实际场景中测试效果

## 📝 参考项目

本实现参考了以下开源项目：
- [yolo-lstm_fire_detection-in-cctv-videos](https://github.com/sureshkumark23/yolo-lstm_fire_detection-in-cctv-videos)
- [Fire-Detection](https://github.com/pedbrgs/Fire-Detection)
- [STCNet](https://github.com/Caoyichao/STCNet)
