# EmberGuard AI - 快速开始指南

## 🚀 立即开始 Phase 1

### 第一步：环境准备 (30分钟)

```bash
# 1. 确认Python版本
python --version  # 需要 3.11+

# 2. 安装核心依赖
pip install ultralytics opencv-python tensorflow numpy pandas scikit-learn

# 3. 验证安装
python -c "from ultralytics import YOLO; print('✅ YOLOv8 安装成功')"
python -c "import tensorflow as tf; print('✅ TensorFlow 安装成功')"
```

### 第二步：训练YOLOv8 (2小时)

```python
# scripts/train_yolo.py
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
results = model.train(
    data='configs/yolo_fire.yaml',
    epochs=50,  # 快速测试用50，正式训练用100+
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project='runs/detect',
    name='fire_v1'
)

print("✅ 训练完成！模型保存在: runs/detect/fire_v1/weights/best.pt")
```

### 第三步：实现特征提取 (1小时)

```python
# emberguard/feature_extractor.py
import numpy as np

class FeatureExtractor:
    """8维基础特征提取器"""
    
    def extract(self, detection):
        """
        从YOLO检测结果提取特征
        
        Args:
            detection: YOLO detection object
            
        Returns:
            np.array: 8维特征向量
        """
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        w = x2 - x1
        h = y2 - y1
        
        features = np.array([
            (x1 + x2) / 2,              # 中心点x
            (y1 + y2) / 2,              # 中心点y
            w,                          # 宽度
            h,                          # 高度
            w * h,                      # 面积
            w / h if h > 0 else 0,      # 宽高比
            float(detection.conf),      # 置信度
            int(detection.cls)          # 类别
        ])
        
        return features

# 测试
if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2
    
    model = YOLO('runs/detect/fire_v1/weights/best.pt')
    extractor = FeatureExtractor()
    
    # 测试图片
    frame = cv2.imread('test_images/fire_test.jpg')
    results = model(frame)
    
    for det in results[0].boxes:
        features = extractor.extract(det)
        print(f"特征向量: {features}")
        print(f"✅ 特征提取成功！")
```

### 第四步：构建LSTM模型 (2小时)

```python
# emberguard/lstm_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(seq_len=30, feature_dim=8, num_classes=3):
    """
    构建LSTM分类模型
    
    Args:
        seq_len: 序列长度（帧数）
        feature_dim: 特征维度
        num_classes: 类别数（0:无火, 1:烟雾, 2:火焰）
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, feature_dim)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 训练脚本
def train_lstm():
    # 准备数据（这里需要你的标注数据）
    # X_train: (samples, 30, 8)
    # y_train: (samples, 3)
    
    model = build_lstm_model()
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                'models/lstm_fire_model.h5',
                save_best_only=True
            )
        ]
    )
    
    print("✅ LSTM训练完成！")
    return model

if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
```

### 第五步：完整推理管道 (1小时)

```python
# emberguard/pipeline.py
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque

class FireDetectionPipeline:
    """完整的火灾检测管道"""
    
    def __init__(self, yolo_path, lstm_path, seq_len=30):
        self.yolo = YOLO(yolo_path)
        self.lstm = load_model(lstm_path)
        self.seq_len = seq_len
        self.feature_buffer = deque(maxlen=seq_len)
        self.extractor = FeatureExtractor()
        
        self.classes = ['无火', '烟雾', '火焰']
    
    def process_frame(self, frame):
        """处理单帧"""
        # YOLO检测
        results = self.yolo(frame, verbose=False)
        
        # 提取特征
        if len(results[0].boxes) > 0:
            # 取置信度最高的检测
            best_det = max(results[0].boxes, key=lambda x: x.conf)
            features = self.extractor.extract(best_det)
        else:
            # 无检测，填充零向量
            features = np.zeros(8)
        
        # 添加到缓冲区
        self.feature_buffer.append(features)
        
        # LSTM预测
        if len(self.feature_buffer) == self.seq_len:
            seq = np.array(list(self.feature_buffer))
            seq = np.expand_dims(seq, axis=0)
            
            pred = self.lstm.predict(seq, verbose=0)
            class_idx = np.argmax(pred[0])
            confidence = pred[0][class_idx]
            
            return {
                'class': self.classes[class_idx],
                'class_id': int(class_idx),
                'confidence': float(confidence),
                'detections': results[0].boxes
            }
        
        return None
    
    def process_video(self, video_path):
        """处理视频"""
        cap = cv2.VideoCapture(video_path)
        results = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            if result:
                results.append({
                    'frame': frame_count,
                    **result
                })
            
            frame_count += 1
        
        cap.release()
        return results

# 使用示例
if __name__ == "__main__":
    pipeline = FireDetectionPipeline(
        yolo_path='runs/detect/fire_v1/weights/best.pt',
        lstm_path='models/lstm_fire_model.h5'
    )
    
    # 测试视频
    results = pipeline.process_video('test_videos/fire_test.mp4')
    
    # 打印结果
    for r in results:
        if r['class_id'] > 0:  # 检测到火或烟
            print(f"帧 {r['frame']}: {r['class']} (置信度: {r['confidence']:.2%})")
```

### 第六步：简单GUI (30分钟)

```python
# scripts/simple_gui.py
import streamlit as st
import cv2
from emberguard.pipeline import FireDetectionPipeline

st.title("🔥 EmberGuard AI - 火灾检测")

# 初始化
@st.cache_resource
def load_pipeline():
    return FireDetectionPipeline(
        yolo_path='runs/detect/fire_v1/weights/best.pt',
        lstm_path='models/lstm_fire_model.h5'
    )

pipeline = load_pipeline()

# 上传视频
uploaded_file = st.file_uploader("上传视频", type=['mp4', 'avi'])

if uploaded_file:
    # 保存临时文件
    with open('temp_video.mp4', 'wb') as f:
        f.write(uploaded_file.read())
    
    if st.button("开始分析"):
        with st.spinner("分析中..."):
            results = pipeline.process_video('temp_video.mp4')
        
        # 显示结果
        st.success(f"分析完成！共 {len(results)} 帧")
        
        # 统计
        fire_frames = sum(1 for r in results if r['class_id'] == 2)
        smoke_frames = sum(1 for r in results if r['class_id'] == 1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总帧数", len(results))
        col2.metric("火焰帧", fire_frames)
        col3.metric("烟雾帧", smoke_frames)
        
        # 详细结果
        st.subheader("检测详情")
        for r in results:
            if r['class_id'] > 0:
                st.write(f"帧 {r['frame']}: {r['class']} ({r['confidence']:.2%})")
```

运行GUI:
```bash
streamlit run scripts/simple_gui.py
```

---

## ✅ 验证清单

完成以上步骤后，你应该能够：

- [ ] YOLOv8模型训练完成
- [ ] 特征提取器工作正常
- [ ] LSTM模型训练完成
- [ ] 完整管道能处理视频
- [ ] GUI界面可以使用

---

## 🎯 下一步

Phase 1完成后，继续：
1. 查看 `docs/TECHNICAL_RESEARCH.md` 了解Phase 2-4
2. 实现误报抑制机制
3. 添加烟雾分类功能
4. 部署到生产环境

---

## 💡 常见问题

**Q: YOLO训练很慢怎么办？**
A: 使用更小的batch size，或者使用预训练模型微调

**Q: LSTM需要多少训练数据？**
A: 建议至少1000个视频序列，每个30帧

**Q: 如何提高检测速度？**
A: 使用YOLOv8n（nano）版本，降低输入分辨率

**Q: 误报率太高怎么办？**
A: 进入Phase 2，实现目标追踪和误报抑制

---

**开始你的EmberGuard AI之旅吧！** 🚀
