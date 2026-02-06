# EmberGuard AI - 快速开始指南

## 🎯 当前项目状态

你的项目已经完成了 **Phase 1 的基础部分**：

✅ **已完成**:
- YOLOv8基础检测模型
- 完整的GUI界面（Tkinter）
- 图片/视频/摄像头/屏幕检测
- 训练、测试、验证脚本
- D-Fire数据集集成

🚧 **待实现** (根据技术研究报告):
- LSTM时序分析模块
- 16维特征提取器
- 目标追踪系统
- 误报抑制机制
- 炊烟vs火灾烟雾分类

---

## 🚀 快速开始

### 第一步：环境准备 (10分钟)

```bash
# 1. 克隆项目（如果还没有）
git clone https://github.com/create-meng/EmberGuard-AI-train.git
cd EmberGuard-AI-train

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "from ultralytics import YOLO; print('✅ YOLOv8 安装成功')"
```

### 第二步：使用现有功能 (立即可用)

#### 方式1: GUI界面（推荐）
```bash
python scripts/run_gui.py
```

功能：
- 📁 文件检测：拖拽或选择图片/视频
- 📹 摄像头检测：实时检测
- 🖥️ 屏幕检测：捕获屏幕内容
- ⚙️ 参数调整：置信度、IoU阈值
- 💾 结果保存：自动保存检测结果

#### 方式2: 命令行测试
```bash
# 测试图片
python scripts/test_model.py --source test_picture/1.png

# 测试视频
python scripts/test_model.py --source path/to/video.mp4

# 测试摄像头
python scripts/test_model.py --source 0

# 使用自定义模型和置信度
python scripts/test_model.py --source test_picture/1.png --model runs/detect/train2/weights/best.pt --conf 0.5
```

### 第三步：使用已训练的模型 ✅

**好消息！你已经训练好了YOLOv8模型！**

模型位置：`runs/detect/train2/weights/best.pt`

查看训练结果：
```bash
# 查看训练曲线
# 打开 runs/detect/train2/results.png

# 查看混淆矩阵
# 打开 runs/detect/train2/confusion_matrix.png
```

使用训练好的模型：
```bash
# GUI会自动使用最佳模型
python scripts/run_gui.py

# 命令行指定模型
python scripts/test_model.py --source test_picture/1.png --model runs/detect/train2/weights/best.pt
```

**如果需要重新训练或微调**：
```bash
# 使用现有脚本
python scripts/train_model.py

# 或者修改参数后训练
# 编辑 scripts/train_model.py，调整 epochs, batch 等参数
```

### 第四步：创建特征提取器 (1小时)

创建新文件 `emberguard/feature_extractor.py`：

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

### 第五步：构建LSTM模型 (2小时)

创建新文件 `emberguard/lstm_model.py`：

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

### 第六步：集成到现有GUI (1小时)

修改 `UI/detection_processor.py`，集成LSTM功能：

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

## ✅ 当前进度检查清单

### Phase 1 基础功能 ✅
- [x] YOLOv8模型训练完成 (`runs/detect/train2/weights/best.pt`)
- [x] GUI界面完整 (Tkinter)
- [x] 图片/视频/摄像头/屏幕检测
- [x] 训练、测试、验证脚本
- [x] D-Fire数据集集成

### Phase 1 LSTM扩展 🚧
- [ ] 特征提取器 (`emberguard/feature_extractor.py`)
- [ ] LSTM模型 (`emberguard/lstm_model.py`)
- [ ] 推理管道 (`emberguard/pipeline.py`)
- [ ] 集成到GUI
- [ ] 训练LSTM模型

### Phase 2 误报抑制 📋
- [ ] 扩展特征维度 (8维→16维)
- [ ] 目标追踪系统
- [ ] 误报抑制模块
- [ ] 性能优化

### Phase 3 烟雾分类 📋
- [ ] 烟雾特征提取
- [ ] 炊烟vs火灾分类器
- [ ] 集成到主管道

---

## 💡 立即行动建议

### 今天可以做的：
1. ✅ **测试现有模型** - 使用GUI测试各种场景
2. ✅ **查看训练结果** - 分析 `runs/detect/train2/` 中的图表
3. 🚀 **开始实现特征提取器** - 创建 `emberguard/` 目录和第一个模块

### 本周目标：
1. 实现8维特征提取器
2. 构建基础LSTM模型
3. 准备LSTM训练数据
4. 完成简单的时序分析

### 遇到问题？
- 查看 `docs/TECHNICAL_RESEARCH.md` 了解详细技术方案
- 参考 `docs/SUMMARY.md` 了解整体规划
- 检查 `PROJECT_STRUCTURE.md` 了解项目结构

---

## 🎊 恭喜！

你已经完成了EmberGuard AI的基础搭建：
- ✅ 完整的开发环境
- ✅ 训练好的YOLOv8模型
- ✅ 功能完善的GUI界面
- ✅ 完整的技术研究和实施方案

**现在可以开始实现LSTM时序分析，让系统更加智能！** 🚀
