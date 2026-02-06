# EmberGuard LSTM模块

LSTM时序分析模块，用于提升火灾检测准确率，降低误报率。

## 模块结构

```
emberguard/
├── __init__.py              # 模块初始化
├── feature_extractor.py     # 特征提取器
├── lstm_model.py           # LSTM模型
├── pipeline.py             # 检测管道
└── README.md               # 本文件
```

## 快速开始

### 1. 特征提取

```python
from emberguard import FeatureExtractor
from ultralytics import YOLO
import cv2

# 初始化
model = YOLO('runs/detect/train2/weights/best.pt')
extractor = FeatureExtractor()

# 检测并提取特征
img = cv2.imread('test.jpg')
results = model(img)
features = extractor.get_best_detection(results, img.shape)

print(features)  # 8维特征向量
```

### 2. 使用检测管道

```python
from emberguard import FireDetectionPipeline

# 创建管道（仅YOLO）
pipeline = FireDetectionPipeline(
    yolo_model_path='runs/detect/train2/weights/best.pt'
)

# 检测单帧
result = pipeline.detect_frame(frame)
print(result['has_detection'])
print(result['yolo_detections'])

# 处理视频
results = pipeline.process_video('input.mp4', 'output.mp4')
```

### 3. 使用LSTM模型

```python
from emberguard import FireDetectionPipeline

# 创建管道（YOLO + LSTM）
pipeline = FireDetectionPipeline(
    yolo_model_path='runs/detect/train2/weights/best.pt',
    lstm_model_path='models/lstm/best.pt'  # LSTM模型
)

# 检测（自动使用LSTM）
result = pipeline.detect_frame(frame)
print(result['lstm_prediction'])      # 0=无火, 1=烟雾, 2=火焰
print(result['lstm_class_name'])      # 类别名称
print(result['lstm_probabilities'])   # 各类别概率
```

## 数据准备

### 准备训练数据

```python
from scripts.prepare_lstm_data import LSTMDataPreparer

# 初始化
preparer = LSTMDataPreparer(
    yolo_model_path='runs/detect/train2/weights/best.pt',
    sequence_length=30
)

# 准备数据集
video_list = [
    ('videos/no_fire_1.mp4', 0),  # 无火
    ('videos/smoke_1.mp4', 1),     # 烟雾
    ('videos/fire_1.mp4', 2),      # 火焰
]

preparer.prepare_dataset(video_list, 'datasets/lstm_data')
```

### 训练LSTM模型

```bash
# 命令行训练
python scripts/train_lstm.py \
    --data_dir datasets/lstm_data \
    --output_dir models/lstm \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

或使用Python:

```python
from scripts.train_lstm import train_lstm_model

train_lstm_model(
    data_dir='datasets/lstm_data',
    output_dir='models/lstm',
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)
```

## 特征说明

### 8维特征向量

1. **cx** - 中心点x坐标（归一化）
2. **cy** - 中心点y坐标（归一化）
3. **w** - 宽度（归一化）
4. **h** - 高度（归一化）
5. **area** - 面积（归一化）
6. **aspect_ratio** - 宽高比
7. **conf** - 置信度
8. **cls** - 类别ID

### 时序序列

- 序列长度: 30帧
- 采样频率: 可配置（默认每帧）
- 缓冲机制: 滑动窗口

## LSTM模型架构

```
输入: (batch, 30, 8)
  ↓
LSTM(128) + Dropout(0.3)
  ↓
LSTM(64) + Dropout(0.3)
  ↓
Dense(64) + ReLU + Dropout(0.3)
  ↓
Dense(3) + Softmax
  ↓
输出: (batch, 3)  # [无火, 烟雾, 火焰]
```

**模型参数**: 211,203个

## 性能目标

- **准确率**: >99%
- **误报率**: <2%
- **推理速度**: >30 FPS (GPU)
- **延迟**: <100ms

## 测试

```bash
# 测试特征提取器
python emberguard/feature_extractor.py

# 测试LSTM模型
python emberguard/lstm_model.py

# 测试检测管道
python scripts/test_pipeline.py
```

## 常见问题

### Q: 如何调整序列长度？

A: 在创建管道时指定 `sequence_length` 参数：

```python
pipeline = FireDetectionPipeline(
    yolo_model_path='...',
    sequence_length=60  # 60帧
)
```

### Q: 如何重置特征缓冲区？

A: 调用 `reset_buffer()` 方法：

```python
pipeline.reset_buffer()
```

### Q: 没有LSTM模型可以使用吗？

A: 可以，管道会自动降级为仅使用YOLO检测：

```python
pipeline = FireDetectionPipeline(
    yolo_model_path='...',
    lstm_model_path=None  # 不使用LSTM
)
```

## 下一步

1. 收集视频数据
2. 标注数据（无火/烟雾/火焰）
3. 训练LSTM模型
4. 集成到GUI界面
5. 性能评估与优化

## 参考

- 技术研究: `docs/TECHNICAL_RESEARCH.md`
- 开发日志: `DEVELOPMENT_LOG.md`
- 项目文档: `README.md`
