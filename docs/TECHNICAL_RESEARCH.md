# EmberGuard AI - 技术研究报告

## 📋 文档信息

- **项目名称**: EmberGuard AI - 智能火灾检测系统
- **研究日期**: 2026年2月6日
- **研究目标**: 分析现有YOLO+LSTM火灾/烟雾检测方案，制定技术实现路线
- **研究方法**: GitHub开源项目分析、代码审查、架构对比

---

## 🎯 研究目标

基于产品说明书的要求，我们需要实现：

1. **YOLO-LSTM融合异常检测** - 结合目标检测与时序分析
2. **热红外特征融合** - 多模态数据融合
3. **时空上下文建模** - 区分炊烟与火灾烟雾
4. **误报率<2%** - 高精度检测
5. **边缘计算部署** - 轻量化模型

---

## 📊 研究项目概览

我们分析了4个高度相关的开源项目：

| 项目 | Stars | 技术栈 | 核心特点 | 适用性 |
|------|-------|--------|----------|--------|
| **yolo-lstm-fire** | 1 | YOLOv8 + LSTM | 直接火灾检测 | ⭐⭐⭐⭐⭐ |
| **Fire-Detection** | 76 | YOLOv5 + 时序追踪 | 时空模式分析 | ⭐⭐⭐⭐ |
| **STCNet** | 37 | 双流网络 | 时空交叉网络 | ⭐⭐⭐ |
| **YoloV8-LSTM-Violence** | 1 | YOLOv8 + LSTM | 暴力检测(可迁移) | ⭐⭐⭐⭐ |

---


## 🔬 项目一：YOLO-LSTM Fire Detection

### 基本信息
- **仓库**: sureshkumark23/yolo-lstm_fire_detection-in-cctv-videos
- **Stars**: 1
- **语言**: Python
- **最后更新**: 2025-10-11

### 技术架构

```
视频输入 → YOLOv8检测 → 特征提取 → LSTM分类 → 火灾判定
```

### 核心实现

#### 1. YOLOv8目标检测
```python
# 加载YOLOv8模型
yolo_model = YOLO("best.pt")

# 对每一帧进行检测
results = yolo_model(frame, verbose=False)

# 提取检测结果
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 边界框坐标
    conf = float(box.conf[0])               # 置信度
    cls = int(box.cls[0])                   # 类别
```

#### 2. 特征提取（8维特征向量）
```python
features = [
    cx,              # 中心点x坐标
    cy,              # 中心点y坐标
    w,               # 宽度
    h,               # 高度
    area,            # 面积
    aspect_ratio,    # 宽高比
    conf,            # 置信度
    cls              # 类别ID
]
```

#### 3. LSTM时序分析
```python
# 使用滑动窗口（30帧）
SEQ_LEN = 30
features_buffer = []

# 收集30帧特征
if len(features_buffer) >= SEQ_LEN:
    seq = np.array(features_buffer[-SEQ_LEN:])
    seq = np.expand_dims(seq, axis=0)  # (1, 30, 8)
    
    # LSTM预测
    pred = lstm_model.predict(seq)
    label = np.argmax(pred)  # 0: no_fire, 1: smoke, 2: fire
```

### 优点分析

✅ **简单直接** - 架构清晰，易于理解和实现
✅ **端到端** - 从视频输入到火灾判定的完整流程
✅ **实时性好** - 使用YOLOv8，推理速度快
✅ **特征工程** - 8维特征向量设计合理

### 缺点分析

❌ **特征单一** - 仅使用边界框几何特征，缺少颜色、纹理等
❌ **窗口固定** - 30帧固定窗口可能不适应所有场景
❌ **缺少后处理** - 没有时序平滑和误报抑制机制
❌ **数据集未公开** - 无法直接复现训练过程

### 适用性评估

**对EmberGuard AI的价值**: ⭐⭐⭐⭐⭐

这是最直接可用的方案，可以作为我们的**基础架构**：
- 直接使用YOLOv8 + LSTM的组合
- 特征提取方法可以直接借鉴
- 需要扩展特征维度（加入颜色、运动等）

---


## 🔬 项目二：Fire-Detection (时空模式分析)

### 基本信息
- **仓库**: pedbrgs/Fire-Detection
- **Stars**: 76 ⭐ (最受欢迎)
- **语言**: Python
- **最后更新**: 2026-01-06
- **论文支持**: 发表在Neural Computing and Applications

### 技术架构

```
视频输入 → YOLOv5检测 → 目标追踪 → 时序分析 → 火灾确认
                                    ↓
                            AVT(面积变化) / TPT(时序持续)
```

### 核心创新：两阶段混合系统

#### 阶段1：空间检测 (YOLOv5)
```python
# YOLOv5检测火焰/烟雾候选区域
pred = model(frame)
det = non_max_suppression(pred, conf_thres, iou_thres)

# 提取边界框
for *xyxy, conf, cls in det:
    xmin, ymin, xmax, ymax = xyxy
    coord_objs.append([xmin, ymin, xmax, ymax])
```

#### 阶段2：时序分析

**方法A: AVT (Area Variation Technique) - 面积变化技术**
- **适用场景**: 室外场景
- **原理**: 真实火灾的检测区域会持续扩大
- **实现**:
```python
class ObjectTracker:
    def bbox_suppression(self, log):
        for (id, areas) in log.areas.items():
            # 计算面积变化的变异系数
            var = variation(np.array(areas[-window_size:]))
            
            # 如果面积变化小于阈值，认为是误报
            if var < area_thresh:
                suppress_detection(id)
```

**方法B: TPT (Temporal Persistence Technique) - 时序持续技术**
- **适用场景**: 室内场景
- **原理**: 真实火灾会在多帧中持续出现
- **实现**:
```python
temporal_buffer = np.zeros((window_size))

# 记录检测结果
temporal_buffer[pos] = True if detected else False

# 计算持续性
persistence = np.sum(temporal_buffer) / window_size

# 如果持续性低于阈值，抑制检测
if persistence < persistence_thresh:
    suppress_detection()
```

### 目标追踪实现

```python
class ObjectTracker:
    def tracking(self, coord_objs):
        # 计算质心
        centroids = self.compute_centroids(coord_objs)
        
        # 计算面积
        areas = self.compute_areas(coord_objs)
        
        # 使用欧氏距离匹配
        D = dist.cdist(object_centroids, centroids)
        
        # 匈牙利算法匹配
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        # 更新追踪对象
        for (row, col) in zip(rows, cols):
            objectID = object_ids[row]
            self.centroids[objectID] = centroids[col]
            self.areas[objectID] = areas[col]
```

### 性能指标

| 方法 | 检测率 | 误报率 | 首帧检测时间 |
|------|--------|--------|--------------|
| YOLOv5 only | 高 | 高 | 快 |
| YOLOv5 + AVT | 中 | 低 | 中 |
| YOLOv5 + TPT | 中 | 极低 | 慢 |

### 优点分析

✅ **学术严谨** - 有论文支持，方法经过验证
✅ **误报控制** - 时序分析有效降低误报
✅ **场景适配** - AVT/TPT分别适配室内外场景
✅ **目标追踪** - 完整的追踪系统，可以跟踪火源扩散
✅ **代码完整** - 包含完整的训练、验证、测试流程

### 缺点分析

❌ **YOLOv5** - 使用较旧版本，可升级到YOLOv8
❌ **手工规则** - AVT/TPT是基于规则的，不够智能
❌ **计算开销** - 目标追踪增加计算复杂度
❌ **参数敏感** - area_thresh、window_size需要调优

### 适用性评估

**对EmberGuard AI的价值**: ⭐⭐⭐⭐

这个项目提供了**误报抑制**的最佳实践：
- AVT面积变化分析可以直接使用
- TPT时序持续性可以作为LSTM的补充
- 目标追踪系统可以用于火源扩散分析
- 需要将规则方法改为学习方法（LSTM）

---


## 🔬 项目三：STCNet (时空交叉网络)

### 基本信息
- **仓库**: Caoyichao/STCNet
- **Stars**: 37
- **语言**: Python
- **论文**: arXiv:2011.04863
- **应用**: 工业烟雾检测

### 技术架构：双流网络

```
视频输入
    ├─→ RGB帧 ────→ 空间分支 (MobileNetV2/SE-ResNeXt) ─┐
    │                                                  ├─→ 特征融合 → 分类
    └─→ 差分帧 ───→ 时序分支 (MobileNetV2/SE-ResNeXt) ─┘
```

### 核心创新

#### 1. 双流输入
```python
# RGB流 - 捕捉空间特征
rgb_frames = video[t:t+seq_len]

# 差分流 - 捕捉运动特征
diff_frames = []
for i in range(1, len(rgb_frames)):
    diff = rgb_frames[i] - rgb_frames[i-1]
    diff_frames.append(diff)
```

#### 2. 时空交叉注意力
```python
class STCNet(nn.Module):
    def __init__(self):
        # 空间分支
        self.spatial_branch = MobileNetV2()
        
        # 时序分支
        self.temporal_branch = MobileNetV2()
        
        # 交叉注意力模块
        self.cross_attention = CrossAttention()
    
    def forward(self, rgb, diff):
        # 提取空间特征
        spatial_feat = self.spatial_branch(rgb)
        
        # 提取时序特征
        temporal_feat = self.temporal_branch(diff)
        
        # 交叉注意力融合
        fused_feat = self.cross_attention(spatial_feat, temporal_feat)
        
        return self.classifier(fused_feat)
```

### 性能对比

| 模型 | 参数量 | FLOPs | 延迟 | 吞吐量 | F-Score |
|------|--------|-------|------|--------|---------|
| RGB-I3D | 12.3M | 62.7G | 30.56ms | 32.71 vid/s | 0.817 |
| STCNet-MobileNetV2 | **3.7M** | **2.4G** | **9.12ms** | **109.7 vid/s** | **0.868** |
| STCNet-SE-ResNeXt | 27.2M | 34.6G | 23.49ms | 42.57 vid/s | **0.885** |

### 优点分析

✅ **高效轻量** - MobileNetV2版本仅3.7M参数
✅ **性能优异** - F-Score达到0.868，超越I3D
✅ **实时性强** - 109.7 vid/s吞吐量，适合边缘部署
✅ **双流设计** - 同时捕捉空间和时序特征
✅ **可视化好** - 提供Grad-CAM可视化

### 缺点分析

❌ **非检测模型** - 是分类模型，不能定位火源
❌ **固定输入** - 需要固定长度的视频片段
❌ **工业场景** - 针对工业烟雾，需要迁移到火灾场景
❌ **无LSTM** - 使用CNN处理时序，不如LSTM灵活

### 适用性评估

**对EmberGuard AI的价值**: ⭐⭐⭐

这个项目提供了**轻量化部署**的思路：
- 双流网络架构可以借鉴
- 差分帧提取运动特征的方法很有价值
- MobileNetV2可以作为特征提取backbone
- 需要改造为检测模型（加入YOLO）

---


## 🔬 项目四：YoloV8-LSTM Violence Detection

### 基本信息
- **仓库**: harmeshgv/YoloV8-LSTM-video-Classification
- **Stars**: 1
- **语言**: Python
- **应用**: 暴力行为检测
- **特点**: 完整的工程化实现

### 技术架构

```
视频上传 → 帧提取 → YOLOv8检测 → 特征提取 → LSTM分类 → 报告生成
                                                        ↓
                                            FastAPI + Streamlit + React
```

### 核心实现

#### 1. 视频预处理
```python
class VideoDataExtractor:
    def extract_video_data(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLOv8检测
            results = self.yolo_model(frame)
            
            # 提取特征
            features = self.extract_features(results)
            frames_data.append(features)
        
        return pd.DataFrame(frames_data)
```

#### 2. 特征提取
```python
class FeatureExtractor:
    def extract_features(self, yolo_results):
        features = {
            'num_persons': 0,
            'avg_confidence': 0,
            'bbox_areas': [],
            'interactions': [],
            'scene_context': []
        }
        
        for detection in yolo_results:
            if detection.cls == 'person':
                features['num_persons'] += 1
                features['bbox_areas'].append(detection.area)
                features['avg_confidence'] += detection.conf
        
        return features
```

#### 3. LSTM分类器
```python
class ViolencePredictor:
    def __init__(self):
        self.lstm_model = self.build_lstm()
    
    def build_lstm(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_len, features)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(2, activation='softmax')  # violent / non-violent
        ])
        return model
    
    def predict(self, video_features):
        # 滑动窗口预测
        predictions = []
        for i in range(0, len(video_features) - seq_len):
            window = video_features[i:i+seq_len]
            pred = self.lstm_model.predict(window)
            predictions.append(pred)
        
        return self.aggregate_predictions(predictions)
```

#### 4. 完整的Web应用

**FastAPI后端**:
```python
@app.post("/analyze")
async def analyze_video(file: UploadFile):
    # 保存上传文件
    temp_path = save_upload(file)
    
    # 提取特征
    features = extractor.extract_video_data(temp_path)
    
    # LSTM预测
    prediction = predictor.predict(features)
    
    # 生成报告
    report = generate_report(prediction)
    
    return JSONResponse(report)
```

**Streamlit界面**:
```python
st.title("Violence Detection System")

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])

if uploaded_file:
    with st.spinner("Analyzing..."):
        result = analyze_video(uploaded_file)
    
    st.success(f"Analysis Complete!")
    st.json(result)
```

**React前端**:
```typescript
const AnalysisPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState(null);
  
  const handleAnalyze = async () => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post('/analyze', formData);
    setResult(response.data);
  };
  
  return (
    <div>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleAnalyze}>Analyze</button>
      {result && <ResultDisplay data={result} />}
    </div>
  );
};
```

### 优点分析

✅ **工程完整** - 包含前后端完整实现
✅ **用户友好** - 提供Web界面和API
✅ **模块化好** - 代码结构清晰，易于扩展
✅ **Docker支持** - 容器化部署
✅ **文档详细** - README完善，易于上手
✅ **实时反馈** - 提供进度条和实时结果

### 缺点分析

❌ **场景不同** - 暴力检测vs火灾检测
❌ **特征简单** - 主要基于人员检测
❌ **模型未开源** - 训练好的模型未提供

### 适用性评估

**对EmberGuard AI的价值**: ⭐⭐⭐⭐

这个项目提供了**完整的工程化方案**：
- FastAPI + Streamlit + React的架构可以直接复用
- 特征提取和LSTM预测的流程可以借鉴
- Docker部署方案可以直接使用
- 需要将人员检测改为火焰/烟雾检测

---


## 💡 综合分析与技术选型

### 各项目对比矩阵

| 维度 | YOLO-LSTM-Fire | Fire-Detection | STCNet | YoloV8-LSTM-Violence |
|------|----------------|----------------|--------|----------------------|
| **架构简洁性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **检测精度** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **实时性** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **误报控制** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **工程完整性** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可扩展性** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **边缘部署** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 关键技术提取

#### 1. 目标检测层
- **选择**: YOLOv8 (来自项目1和4)
- **理由**: 最新版本，速度快，精度高
- **改进**: 使用D-Fire数据集微调

#### 2. 特征提取层
**基础特征** (来自项目1):
- 边界框几何特征: cx, cy, w, h, area, aspect_ratio
- 检测置信度: conf
- 类别信息: cls

**扩展特征** (来自项目2和3):
- 面积变化率: area_change_rate
- 运动特征: optical_flow, diff_frames
- 颜色特征: rgb_histogram, hsv_features
- 纹理特征: lbp, gabor

**最终特征向量** (16维):
```python
features = [
    # 几何特征 (6维)
    cx, cy, w, h, area, aspect_ratio,
    
    # 检测特征 (2维)
    conf, cls,
    
    # 时序特征 (3维)
    area_change_rate, velocity_x, velocity_y,
    
    # 颜色特征 (3维)
    mean_r, mean_g, mean_b,
    
    # 纹理特征 (2维)
    texture_energy, texture_entropy
]
```

#### 3. 时序分析层
**LSTM架构** (来自项目1和4):
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 16)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # no_fire, smoke, fire
])
```

#### 4. 误报抑制层
**方法A: 面积变化分析** (来自项目2):
```python
def area_variation_check(areas, window=20, thresh=0.05):
    """真实火灾的面积会持续增长"""
    var = variation(areas[-window:])
    return var >= thresh
```

**方法B: 时序持续性检查** (来自项目2):
```python
def temporal_persistence_check(detections, window=20, thresh=0.5):
    """真实火灾会持续出现"""
    persistence = sum(detections[-window:]) / window
    return persistence >= thresh
```

**方法C: LSTM置信度平滑**:
```python
def confidence_smoothing(predictions, window=5):
    """使用移动平均平滑预测结果"""
    smoothed = np.convolve(predictions, np.ones(window)/window, mode='valid')
    return smoothed
```

#### 5. 目标追踪层 (来自项目2)
```python
class FireTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
    
    def update(self, detections):
        # 计算质心
        centroids = compute_centroids(detections)
        
        # 匹配现有追踪对象
        if len(self.trackers) > 0:
            distances = cdist(self.centroids, centroids)
            matches = hungarian_algorithm(distances)
            
            # 更新追踪对象
            for (tracker_id, det_id) in matches:
                self.trackers[tracker_id].update(detections[det_id])
        
        # 注册新对象
        for unmatched_det in unmatched_detections:
            self.register(unmatched_det)
```

---


## 🎯 EmberGuard AI 技术实现方案

基于以上研究，我们制定以下分阶段实现方案：

---

### Phase 1: 基础YOLO-LSTM系统 (2周)

#### 目标
实现基础的火灾检测系统，达到90%以上准确率

#### 技术栈
- YOLOv8 (目标检测)
- LSTM (时序分析)
- 8维基础特征

#### 实现步骤

**Step 1.1: YOLOv8微调** (3天)
```python
# 使用D-Fire数据集训练
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='configs/yolo_fire.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

**Step 1.2: 特征提取器** (2天)
```python
class FeatureExtractor:
    def extract(self, detection):
        x1, y1, x2, y2 = detection.xyxy[0]
        w, h = x2 - x1, y2 - y1
        
        return np.array([
            (x1 + x2) / 2,  # cx
            (y1 + y2) / 2,  # cy
            w,              # width
            h,              # height
            w * h,          # area
            w / h if h > 0 else 0,  # aspect_ratio
            detection.conf, # confidence
            detection.cls   # class
        ])
```

**Step 1.3: LSTM训练** (5天)
```python
# 数据准备
def prepare_sequences(video_features, seq_len=30):
    sequences = []
    labels = []
    
    for i in range(len(video_features) - seq_len):
        seq = video_features[i:i+seq_len]
        label = video_labels[i+seq_len]
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# LSTM模型
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 8)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Step 1.4: 推理管道** (2天)
```python
class FireDetectionPipeline:
    def __init__(self):
        self.yolo = YOLO('runs/detect/train2/weights/best.pt')
        self.lstm = load_model('models/lstm_fire_model.h5')
        self.feature_buffer = []
        self.seq_len = 30
    
    def process_frame(self, frame):
        # YOLO检测
        results = self.yolo(frame)
        
        # 提取特征
        features = []
        for det in results[0].boxes:
            feat = self.extract_features(det)
            features.append(feat)
        
        # 如果有检测结果，取置信度最高的
        if features:
            best_feat = max(features, key=lambda x: x[6])
            self.feature_buffer.append(best_feat)
        else:
            # 无检测，填充零向量
            self.feature_buffer.append(np.zeros(8))
        
        # LSTM预测
        if len(self.feature_buffer) >= self.seq_len:
            seq = np.array(self.feature_buffer[-self.seq_len:])
            seq = np.expand_dims(seq, axis=0)
            pred = self.lstm.predict(seq, verbose=0)
            return pred
        
        return None
```

#### 预期成果
- ✅ 基础火灾检测功能
- ✅ 实时推理能力 (~30 FPS)
- ✅ 准确率 > 90%
- ✅ 可视化检测结果

---

### Phase 2: 误报抑制与优化 (2周)

#### 目标
降低误报率至 < 5%，提升系统鲁棒性

#### 新增功能

**2.1: 扩展特征维度** (3天)
```python
class EnhancedFeatureExtractor:
    def extract(self, detection, frame, prev_frame):
        # 基础特征 (8维)
        basic_feat = self.extract_basic(detection)
        
        # 颜色特征 (3维)
        roi = frame[y1:y2, x1:x2]
        color_feat = np.mean(roi, axis=(0, 1))  # RGB均值
        
        # 运动特征 (2维)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion_feat = [
                np.mean(flow[..., 0]),  # x方向运动
                np.mean(flow[..., 1])   # y方向运动
            ]
        else:
            motion_feat = [0, 0]
        
        # 纹理特征 (3维)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_feat = [
            np.std(gray_roi),           # 标准差
            cv2.Laplacian(gray_roi, cv2.CV_64F).var(),  # 拉普拉斯方差
            np.mean(np.abs(np.diff(gray_roi, axis=0)))  # 梯度
        ]
        
        # 合并特征 (16维)
        return np.concatenate([
            basic_feat,    # 8维
            color_feat,    # 3维
            motion_feat,   # 2维
            texture_feat   # 3维
        ])
```

**2.2: 目标追踪系统** (4天)
```python
from scipy.spatial import distance as dist

class FireObjectTracker:
    def __init__(self, max_disappeared=30):
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.areas = OrderedDict()
        self.next_id = 0
        self.max_disappeared = max_disappeared
    
    def register(self, centroid, area):
        self.objects[self.next_id] = centroid
        self.areas[self.next_id] = [area]
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.areas[object_id]
    
    def update(self, detections):
        # 如果没有检测结果
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # 计算质心和面积
        centroids = []
        areas = []
        for det in detections:
            cx = (det[0] + det[2]) / 2
            cy = (det[1] + det[3]) / 2
            area = (det[2] - det[0]) * (det[3] - det[1])
            centroids.append((cx, cy))
            areas.append(area)
        
        # 如果没有追踪对象，注册所有检测
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i], areas[i])
        else:
            # 匹配现有对象
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # 计算距离矩阵
            D = dist.cdist(np.array(object_centroids), centroids)
            
            # 匈牙利算法匹配
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.areas[object_id].append(areas[col])
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # 处理未匹配的对象
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for col in unused_cols:
                self.register(centroids[col], areas[col])
        
        return self.objects
```

**2.3: 误报抑制模块** (3天)
```python
from scipy.stats import variation

class FalsePositiveSuppressor:
    def __init__(self, area_thresh=0.05, persistence_thresh=0.5, window_size=20):
        self.area_thresh = area_thresh
        self.persistence_thresh = persistence_thresh
        self.window_size = window_size
    
    def check_area_variation(self, tracker, object_id):
        """检查面积变化 - 真实火灾会扩大"""
        areas = tracker.areas[object_id]
        if len(areas) < self.window_size:
            return True  # 数据不足，暂不抑制
        
        recent_areas = areas[-self.window_size:]
        var = variation(recent_areas)
        
        return var >= self.area_thresh
    
    def check_temporal_persistence(self, detection_history):
        """检查时序持续性 - 真实火灾会持续出现"""
        if len(detection_history) < self.window_size:
            return True
        
        recent = detection_history[-self.window_size:]
        persistence = sum(recent) / len(recent)
        
        return persistence >= self.persistence_thresh
    
    def check_color_consistency(self, color_history):
        """检查颜色一致性 - 火焰颜色应该稳定"""
        if len(color_history) < 10:
            return True
        
        recent_colors = np.array(color_history[-10:])
        color_std = np.std(recent_colors, axis=0)
        
        # 火焰颜色应该在红-橙-黄范围内且相对稳定
        return np.mean(color_std) < 30  # 阈值可调
    
    def should_suppress(self, tracker, object_id, detection_history, color_history):
        """综合判断是否应该抑制检测"""
        checks = [
            self.check_area_variation(tracker, object_id),
            self.check_temporal_persistence(detection_history),
            self.check_color_consistency(color_history)
        ]
        
        # 至少通过2/3的检查才不抑制
        return sum(checks) < 2
```

**2.4: 集成管道** (4天)
```python
class EnhancedFireDetectionPipeline:
    def __init__(self):
        self.yolo = YOLO('runs/detect/train2/weights/best.pt')
        self.lstm = load_model('models/lstm_fire_model_v2.h5')
        self.feature_extractor = EnhancedFeatureExtractor()
        self.tracker = FireObjectTracker()
        self.suppressor = FalsePositiveSuppressor()
        
        self.feature_buffer = []
        self.detection_history = []
        self.color_history = defaultdict(list)
        self.prev_frame = None
        self.seq_len = 30
    
    def process_frame(self, frame):
        # YOLO检测
        results = self.yolo(frame)
        
        # 更新追踪器
        detections = []
        for det in results[0].boxes:
            bbox = det.xyxy[0].cpu().numpy()
            detections.append(bbox)
        
        tracked_objects = self.tracker.update(detections)
        
        # 提取特征
        features = []
        for det in results[0].boxes:
            feat = self.feature_extractor.extract(det, frame, self.prev_frame)
            features.append(feat)
            
            # 记录颜色历史
            object_id = self.find_object_id(det, tracked_objects)
            if object_id is not None:
                self.color_history[object_id].append(feat[8:11])
        
        # 误报抑制
        valid_features = []
        for i, feat in enumerate(features):
            object_id = self.find_object_id(results[0].boxes[i], tracked_objects)
            if object_id is not None:
                if not self.suppressor.should_suppress(
                    self.tracker, object_id, 
                    self.detection_history, 
                    self.color_history[object_id]
                ):
                    valid_features.append(feat)
        
        # 更新检测历史
        self.detection_history.append(len(valid_features) > 0)
        
        # LSTM预测
        if valid_features:
            best_feat = max(valid_features, key=lambda x: x[6])
            self.feature_buffer.append(best_feat)
        else:
            self.feature_buffer.append(np.zeros(16))
        
        if len(self.feature_buffer) >= self.seq_len:
            seq = np.array(self.feature_buffer[-self.seq_len:])
            seq = np.expand_dims(seq, axis=0)
            pred = self.lstm.predict(seq, verbose=0)
            
            self.prev_frame = frame.copy()
            return pred, tracked_objects
        
        self.prev_frame = frame.copy()
        return None, tracked_objects
```

#### 预期成果
- ✅ 误报率 < 5%
- ✅ 准确率 > 95%
- ✅ 目标追踪功能
- ✅ 火源扩散分析

---


### Phase 3: 炊烟vs火灾烟雾区分 (2周)

#### 目标
实现说明书中的"区分炊烟与火灾烟雾"功能，误报率降至 < 2%

#### 核心挑战
炊烟和火灾烟雾的视觉特征相似，需要从以下维度区分：
1. **运动模式**: 炊烟上升平稳，火灾烟雾扩散快速
2. **颜色变化**: 炊烟颜色单一，火灾烟雾颜色多变
3. **持续时间**: 炊烟短暂，火灾烟雾持续
4. **伴随特征**: 火灾烟雾常伴随火焰

#### 实现方案

**3.1: 烟雾特征提取器** (4天)
```python
class SmokeFeatureExtractor:
    def __init__(self):
        self.optical_flow = cv2.FarnebackOpticalFlow_create()
    
    def extract_smoke_features(self, smoke_roi, frame_sequence):
        """提取烟雾专用特征"""
        features = {}
        
        # 1. 运动特征
        features['motion'] = self.analyze_motion_pattern(frame_sequence)
        
        # 2. 扩散速度
        features['expansion_rate'] = self.calculate_expansion_rate(frame_sequence)
        
        # 3. 颜色时序变化
        features['color_variance'] = self.analyze_color_variance(frame_sequence)
        
        # 4. 纹理复杂度
        features['texture'] = self.analyze_texture(smoke_roi)
        
        # 5. 形状变化
        features['shape_change'] = self.analyze_shape_change(frame_sequence)
        
        return features
    
    def analyze_motion_pattern(self, frame_sequence):
        """分析运动模式"""
        flows = []
        for i in range(1, len(frame_sequence)):
            prev = cv2.cvtColor(frame_sequence[i-1], cv2.COLOR_BGR2GRAY)
            curr = cv2.cvtColor(frame_sequence[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        
        # 计算运动方向的一致性
        flow_directions = [np.arctan2(f[..., 1], f[..., 0]) for f in flows]
        direction_std = np.std(flow_directions)
        
        # 计算运动速度
        flow_magnitudes = [np.sqrt(f[..., 0]**2 + f[..., 1]**2) for f in flows]
        avg_speed = np.mean(flow_magnitudes)
        
        return {
            'direction_consistency': 1 / (1 + direction_std),  # 炊烟方向一致
            'average_speed': avg_speed,
            'speed_variance': np.std(flow_magnitudes)  # 火灾烟雾速度变化大
        }
    
    def calculate_expansion_rate(self, frame_sequence):
        """计算扩散速度"""
        areas = []
        for frame in frame_sequence:
            # 假设已经分割出烟雾区域
            smoke_mask = self.segment_smoke(frame)
            area = np.sum(smoke_mask > 0)
            areas.append(area)
        
        # 计算面积增长率
        if len(areas) > 1:
            growth_rates = np.diff(areas) / areas[:-1]
            return {
                'avg_growth_rate': np.mean(growth_rates),
                'max_growth_rate': np.max(growth_rates),
                'growth_acceleration': np.std(growth_rates)  # 火灾加速扩散
            }
        return {'avg_growth_rate': 0, 'max_growth_rate': 0, 'growth_acceleration': 0}
    
    def analyze_color_variance(self, frame_sequence):
        """分析颜色时序变化"""
        colors = []
        for frame in frame_sequence:
            smoke_roi = self.extract_smoke_roi(frame)
            mean_color = np.mean(smoke_roi, axis=(0, 1))
            colors.append(mean_color)
        
        colors = np.array(colors)
        
        return {
            'color_std': np.mean(np.std(colors, axis=0)),  # 火灾烟雾颜色变化大
            'color_trend': self.calculate_color_trend(colors),  # 火灾烟雾颜色变深
            'color_range': np.ptp(colors, axis=0).mean()  # 颜色范围
        }
    
    def analyze_texture(self, smoke_roi):
        """分析纹理复杂度"""
        gray = cv2.cvtColor(smoke_roi, cv2.COLOR_BGR2GRAY)
        
        # LBP纹理
        lbp = self.calculate_lbp(gray)
        
        # Gabor滤波
        gabor_features = self.calculate_gabor(gray)
        
        # 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'lbp_variance': np.var(lbp),
            'gabor_energy': np.mean(gabor_features),
            'edge_density': edge_density  # 火灾烟雾边缘更复杂
        }
```

**3.2: 烟雾分类器** (3天)
```python
class SmokeCookingFireClassifier:
    """三分类器: 炊烟 / 火灾烟雾 / 无烟雾"""
    
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        """构建专门的烟雾分类模型"""
        model = Sequential([
            # 输入: 烟雾特征序列 (seq_len, feature_dim)
            LSTM(64, return_sequences=True, input_shape=(20, 32)),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # cooking_smoke, fire_smoke, no_smoke
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, smoke_features_sequence):
        """准备输入特征"""
        feature_vector = []
        
        for features in smoke_features_sequence:
            vec = np.concatenate([
                # 运动特征 (3维)
                [features['motion']['direction_consistency'],
                 features['motion']['average_speed'],
                 features['motion']['speed_variance']],
                
                # 扩散特征 (3维)
                [features['expansion_rate']['avg_growth_rate'],
                 features['expansion_rate']['max_growth_rate'],
                 features['expansion_rate']['growth_acceleration']],
                
                # 颜色特征 (3维)
                [features['color_variance']['color_std'],
                 features['color_variance']['color_trend'],
                 features['color_variance']['color_range']],
                
                # 纹理特征 (3维)
                [features['texture']['lbp_variance'],
                 features['texture']['gabor_energy'],
                 features['texture']['edge_density']],
                
                # ... 其他特征
            ])
            
            feature_vector.append(vec)
        
        return np.array(feature_vector)
    
    def classify(self, smoke_features_sequence):
        """分类烟雾类型"""
        features = self.prepare_features(smoke_features_sequence)
        features = np.expand_dims(features, axis=0)
        
        pred = self.model.predict(features)
        
        classes = ['cooking_smoke', 'fire_smoke', 'no_smoke']
        class_idx = np.argmax(pred)
        confidence = pred[0][class_idx]
        
        return {
            'class': classes[class_idx],
            'confidence': float(confidence),
            'probabilities': {
                'cooking_smoke': float(pred[0][0]),
                'fire_smoke': float(pred[0][1]),
                'no_smoke': float(pred[0][2])
            }
        }
```

**3.3: 集成到主管道** (3天)
```python
class FireDetectionWithSmokeClassification:
    def __init__(self):
        self.fire_detector = EnhancedFireDetectionPipeline()
        self.smoke_extractor = SmokeFeatureExtractor()
        self.smoke_classifier = SmokeCookingFireClassifier()
        
        self.smoke_feature_buffer = []
        self.smoke_seq_len = 20
    
    def process_frame(self, frame):
        # 1. 基础火灾检测
        fire_pred, tracked_objects = self.fire_detector.process_frame(frame)
        
        if fire_pred is None:
            return None
        
        # 2. 如果检测到烟雾类别
        fire_class = np.argmax(fire_pred)
        if fire_class == 1:  # smoke class
            # 提取烟雾特征
            smoke_features = self.smoke_extractor.extract_smoke_features(
                smoke_roi=self.extract_smoke_roi(frame, tracked_objects),
                frame_sequence=self.get_recent_frames()
            )
            
            self.smoke_feature_buffer.append(smoke_features)
            
            # 烟雾分类
            if len(self.smoke_feature_buffer) >= self.smoke_seq_len:
                smoke_classification = self.smoke_classifier.classify(
                    self.smoke_feature_buffer[-self.smoke_seq_len:]
                )
                
                # 如果是炊烟，抑制告警
                if smoke_classification['class'] == 'cooking_smoke':
                    if smoke_classification['confidence'] > 0.8:
                        return {
                            'alert': False,
                            'reason': 'cooking_smoke_detected',
                            'confidence': smoke_classification['confidence']
                        }
                
                # 如果是火灾烟雾，增强告警
                elif smoke_classification['class'] == 'fire_smoke':
                    return {
                        'alert': True,
                        'type': 'fire_smoke',
                        'confidence': smoke_classification['confidence'],
                        'fire_prediction': fire_pred,
                        'tracked_objects': tracked_objects
                    }
        
        # 3. 如果检测到火焰
        elif fire_class == 2:  # fire class
            return {
                'alert': True,
                'type': 'fire',
                'confidence': float(fire_pred[0][fire_class]),
                'tracked_objects': tracked_objects
            }
        
        return {
            'alert': False,
            'reason': 'no_fire_detected'
        }
```

**3.4: 数据增强与训练** (4天)
```python
# 数据增强策略
def augment_smoke_data(video_path, label):
    """针对烟雾数据的增强"""
    augmentations = [
        # 亮度变化
        A.RandomBrightnessContrast(p=0.5),
        
        # 模糊
        A.GaussianBlur(p=0.3),
        
        # 噪声
        A.GaussNoise(p=0.3),
        
        # 颜色抖动
        A.ColorJitter(p=0.5),
        
        # 时间扭曲 (改变播放速度)
        TimeWarp(rate_range=(0.8, 1.2)),
        
        # 帧丢失 (模拟网络不稳定)
        FrameDrop(drop_rate=0.1)
    ]
    
    return apply_augmentations(video_path, augmentations)

# 训练策略
def train_smoke_classifier():
    # 数据集划分
    # - 炊烟视频: 1000个
    # - 火灾烟雾视频: 1000个
    # - 无烟雾视频: 500个
    
    train_data, val_data, test_data = prepare_smoke_dataset()
    
    # 类别权重 (处理不平衡)
    class_weights = {
        0: 1.0,  # cooking_smoke
        1: 1.5,  # fire_smoke (更重要)
        2: 0.5   # no_smoke
    }
    
    # 训练
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=10),
            ReduceLROnPlateau(patience=5),
            ModelCheckpoint('best_smoke_classifier.h5')
        ]
    )
```

#### 预期成果
- ✅ 炊烟识别准确率 > 90%
- ✅ 火灾烟雾识别准确率 > 95%
- ✅ 总体误报率 < 2%
- ✅ 实现说明书核心功能

---


### Phase 4: 工程化与部署 (2周)

#### 目标
完整的Web应用、API接口、Docker部署

#### 技术栈
- **后端**: FastAPI
- **前端**: Streamlit + React (可选)
- **部署**: Docker + Docker Compose
- **监控**: Prometheus + Grafana (可选)

#### 实现步骤

**4.1: FastAPI后端** (3天)
```python
# main.py
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np

app = FastAPI(title="EmberGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器
detector = FireDetectionWithSmokeClassification()

@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """分析上传的视频文件"""
    # 保存临时文件
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 处理视频
    results = []
    cap = cv2.VideoCapture(temp_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        if result and result.get('alert'):
            results.append({
                'frame': frame_count,
                'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                'type': result['type'],
                'confidence': result['confidence']
            })
        
        frame_count += 1
    
    cap.release()
    
    return {
        'total_frames': frame_count,
        'alerts': results,
        'summary': generate_summary(results)
    }

@app.websocket("/ws/realtime")
async def realtime_detection(websocket: WebSocket):
    """实时检测WebSocket接口"""
    await websocket.accept()
    
    try:
        while True:
            # 接收帧数据
            data = await websocket.receive_bytes()
            
            # 解码图像
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 检测
            result = detector.process_frame(frame)
            
            # 发送结果
            if result:
                await websocket.send_json(result)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        'status': 'healthy',
        'model_loaded': detector is not None,
        'version': '1.0.0'
    }

@app.get("/api/stats")
async def get_statistics():
    """获取统计信息"""
    return {
        'total_detections': detector.get_total_detections(),
        'false_positives_suppressed': detector.get_suppressed_count(),
        'average_confidence': detector.get_average_confidence()
    }
```

**4.2: Streamlit界面** (2天)
```python
# streamlit_app.py
import streamlit as st
import requests
import cv2
from PIL import Image

st.set_page_config(
    page_title="EmberGuard AI",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 EmberGuard AI - 智能火灾检测系统")

# 侧边栏
with st.sidebar:
    st.header("设置")
    detection_mode = st.selectbox(
        "检测模式",
        ["视频文件", "实时摄像头", "RTSP流"]
    )
    
    confidence_threshold = st.slider(
        "置信度阈值",
        0.0, 1.0, 0.7, 0.05
    )
    
    enable_smoke_classification = st.checkbox(
        "启用炊烟识别",
        value=True
    )

# 主界面
if detection_mode == "视频文件":
    uploaded_file = st.file_uploader(
        "上传视频文件",
        type=['mp4', 'avi', 'mov']
    )
    
    if uploaded_file:
        if st.button("开始分析"):
            with st.spinner("正在分析视频..."):
                # 调用API
                files = {'file': uploaded_file}
                response = requests.post(
                    "http://localhost:8000/api/analyze/video",
                    files=files
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # 显示结果
                    st.success(f"分析完成！共检测到 {len(results['alerts'])} 个异常")
                    
                    # 时间线
                    st.subheader("检测时间线")
                    for alert in results['alerts']:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"⏱️ {alert['timestamp']:.2f}s")
                        with col2:
                            st.write(f"🔥 {alert['type']}")
                        with col3:
                            st.write(f"📊 {alert['confidence']:.2%}")
                    
                    # 统计图表
                    st.subheader("统计分析")
                    import plotly.express as px
                    import pandas as pd
                    
                    df = pd.DataFrame(results['alerts'])
                    fig = px.line(df, x='timestamp', y='confidence', 
                                  title='置信度变化曲线')
                    st.plotly_chart(fig)

elif detection_mode == "实时摄像头":
    st.subheader("实时检测")
    
    # 摄像头选择
    camera_id = st.number_input("摄像头ID", 0, 10, 0)
    
    if st.button("开始检测"):
        # 创建占位符
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            result = detector.process_frame(frame)
            
            # 显示帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            # 显示告警
            if result and result.get('alert'):
                alert_placeholder.error(
                    f"⚠️ 检测到{result['type']}！置信度: {result['confidence']:.2%}"
                )
            else:
                alert_placeholder.success("✅ 正常")
        
        cap.release()
```

**4.3: Docker部署** (3天)
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000 8501

# 启动命令
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  emberguard-api:
    build: .
    container_name: emberguard-api
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # 可选: Redis缓存
  redis:
    image: redis:alpine
    container_name: emberguard-redis
    ports:
      - "6379:6379"
    restart: unless-stopped

  # 可选: PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    container_name: emberguard-db
    environment:
      POSTGRES_DB: emberguard
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

**4.4: 性能优化** (3天)
```python
# 模型量化
def quantize_model(model_path, output_path):
    """量化模型以减小体积和提升速度"""
    import torch
    
    model = torch.load(model_path)
    model.eval()
    
    # 动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8
    )
    
    torch.save(quantized_model, output_path)
    
    # 对比大小
    original_size = os.path.getsize(model_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"原始模型: {original_size:.2f} MB")
    print(f"量化模型: {quantized_size:.2f} MB")
    print(f"压缩率: {(1 - quantized_size/original_size)*100:.1f}%")

# 批处理优化
class BatchProcessor:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.frame_buffer = []
    
    def add_frame(self, frame):
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) >= self.batch_size:
            return self.process_batch()
        
        return None
    
    def process_batch(self):
        """批量处理帧"""
        batch = np.array(self.frame_buffer)
        
        # YOLO批量推理
        results = self.yolo(batch)
        
        # 清空缓冲区
        self.frame_buffer = []
        
        return results

# 缓存机制
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_feature_extraction(frame_hash):
    """缓存特征提取结果"""
    # 特征提取逻辑
    pass
```

**4.5: 监控与日志** (2天)
```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger('emberguard')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        'logs/emberguard.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
detection_counter = Counter(
    'fire_detections_total',
    'Total number of fire detections'
)

false_positive_counter = Counter(
    'false_positives_suppressed_total',
    'Total number of false positives suppressed'
)

inference_time = Histogram(
    'inference_duration_seconds',
    'Time spent on inference'
)

active_alerts = Gauge(
    'active_fire_alerts',
    'Number of active fire alerts'
)

# 使用示例
def process_with_metrics(frame):
    with inference_time.time():
        result = detector.process_frame(frame)
    
    if result and result.get('alert'):
        detection_counter.inc()
        active_alerts.inc()
    
    return result
```

#### 预期成果
- ✅ 完整的Web应用
- ✅ RESTful API接口
- ✅ WebSocket实时检测
- ✅ Docker一键部署
- ✅ 性能监控系统
- ✅ 日志记录完善

---


## 📈 性能指标与评估

### 目标性能指标

| 指标 | 目标值 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|--------|---------|---------|---------|---------|
| **检测准确率** | >99% | 90% | 95% | 97% | 99% |
| **误报率** | <2% | 10% | 5% | 2% | <2% |
| **推理速度(GPU)** | >30 FPS | 45 FPS | 35 FPS | 30 FPS | 30 FPS |
| **推理速度(CPU)** | >10 FPS | 8 FPS | 6 FPS | 5 FPS | 10 FPS |
| **模型大小** | <50MB | 6MB | 15MB | 25MB | 20MB |
| **首帧检测时间** | <3s | 1s | 1.5s | 2s | 1s |

### 评估数据集

#### 训练集
- **D-Fire数据集**: 21,527张图像
- **自采集数据**: 5,000张图像
- **数据增强**: 3x扩充
- **总计**: ~80,000张图像

#### 验证集
- **D-Fire验证集**: 2,000张图像
- **自采集验证集**: 500张图像
- **总计**: 2,500张图像

#### 测试集
- **真实场景视频**: 100个
  - 室内火灾: 30个
  - 室外火灾: 30个
  - 炊烟场景: 20个
  - 正常场景: 20个

### 评估方法

```python
def evaluate_model(model, test_dataset):
    """全面评估模型性能"""
    
    metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'false_positive_rate': 0,
        'false_negative_rate': 0,
        'inference_time': [],
        'confusion_matrix': None
    }
    
    y_true = []
    y_pred = []
    
    for video_path, label in test_dataset:
        start_time = time.time()
        
        # 处理视频
        predictions = process_video(model, video_path)
        
        # 记录推理时间
        inference_time = time.time() - start_time
        metrics['inference_time'].append(inference_time)
        
        # 聚合预测结果
        final_pred = aggregate_predictions(predictions)
        
        y_true.append(label)
        y_pred.append(final_pred)
    
    # 计算指标
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # 计算误报率和漏报率
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    metrics['false_positive_rate'] = fp / (fp + tn)
    metrics['false_negative_rate'] = fn / (fn + tp)
    
    # 平均推理时间
    metrics['avg_inference_time'] = np.mean(metrics['inference_time'])
    
    return metrics
```

---

## 🔧 开发工具与环境

### 开发环境
```bash
# Python环境
Python 3.11+
CUDA 11.8+
cuDNN 8.6+

# 核心依赖
ultralytics==8.3.0
torch==2.0.0
torchvision==0.15.0
opencv-python==4.8.0
tensorflow==2.13.0  # for LSTM
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
scipy==1.11.0

# Web框架
fastapi==0.104.0
streamlit==1.28.0
uvicorn==0.24.0

# 工具库
albumentations==1.3.0  # 数据增强
plotly==5.17.0  # 可视化
prometheus-client==0.18.0  # 监控
```

### 推荐硬件配置

#### 开发环境
- **CPU**: Intel i7/AMD Ryzen 7 或更高
- **GPU**: NVIDIA RTX 3060 (12GB) 或更高
- **内存**: 32GB RAM
- **存储**: 500GB SSD

#### 生产环境
- **CPU**: Intel Xeon/AMD EPYC
- **GPU**: NVIDIA T4/A10 或更高
- **内存**: 64GB RAM
- **存储**: 1TB NVMe SSD

#### 边缘设备
- **Jetson Nano**: 入门级边缘部署
- **Jetson Xavier NX**: 推荐配置
- **Jetson AGX Orin**: 高性能配置

---

## 📚 参考资料

### 学术论文
1. **YOLOv8**: Ultralytics YOLOv8 Documentation
2. **Fire Detection**: "A hybrid method for fire detection based on spatial and temporal patterns" (Neural Computing and Applications, 2023)
3. **STCNet**: "STCNet: Spatio-Temporal Cross Network for Industrial Smoke Detection" (arXiv:2011.04863)
4. **LSTM**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)

### 开源项目
1. [sureshkumark23/yolo-lstm_fire_detection](https://github.com/sureshkumark23/yolo-lstm_fire_detection-in-cctv-videos)
2. [pedbrgs/Fire-Detection](https://github.com/pedbrgs/Fire-Detection)
3. [Caoyichao/STCNet](https://github.com/Caoyichao/STCNet)
4. [harmeshgv/YoloV8-LSTM-video-Classification](https://github.com/harmeshgv/YoloV8-LSTM-video-Classification)

### 数据集
1. **D-Fire Dataset**: [GitHub](https://github.com/gaiasd/DFireDataset)
2. **FireNet Dataset**: [Google Drive](https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq)
3. **Foggia's Dataset**: [MIVIA](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)

### 技术文档
1. [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
2. [PyTorch Documentation](https://pytorch.org/docs/)
3. [TensorFlow/Keras LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
4. [FastAPI Documentation](https://fastapi.tiangolo.com/)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🎯 总结与建议

### 核心技术路线

```
YOLOv8检测 → 特征提取(16维) → LSTM时序分析 → 误报抑制 → 烟雾分类
    ↓            ↓                  ↓              ↓            ↓
  D-Fire      几何+颜色+运动      滑动窗口      追踪+AVT/TPT   炊烟vs火灾
```

### 关键成功因素

1. **数据质量** ⭐⭐⭐⭐⭐
   - 高质量标注数据
   - 多样化场景覆盖
   - 充分的数据增强

2. **特征工程** ⭐⭐⭐⭐
   - 16维综合特征
   - 时序特征提取
   - 多模态融合

3. **误报控制** ⭐⭐⭐⭐⭐
   - 目标追踪
   - 面积变化分析
   - 时序持续性检查
   - 烟雾分类

4. **工程实现** ⭐⭐⭐⭐
   - 模块化设计
   - 完善的API
   - Docker部署
   - 性能监控

### 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 数据不足 | 高 | 数据增强、迁移学习 |
| 误报率高 | 高 | 多层误报抑制机制 |
| 实时性差 | 中 | 模型量化、批处理优化 |
| 炊烟误报 | 高 | 专门的烟雾分类器 |
| 边缘部署难 | 中 | 轻量化模型、TensorRT加速 |

### 下一步行动

1. **立即开始** (本周)
   - ✅ 搭建开发环境
   - ✅ 准备D-Fire数据集
   - ✅ 训练基础YOLOv8模型

2. **Phase 1实施** (第1-2周)
   - 实现YOLO-LSTM基础架构
   - 完成特征提取器
   - 训练LSTM模型
   - 构建推理管道

3. **Phase 2实施** (第3-4周)
   - 扩展特征维度
   - 实现目标追踪
   - 添加误报抑制
   - 性能优化

4. **Phase 3实施** (第5-6周)
   - 烟雾特征提取
   - 训练烟雾分类器
   - 集成到主管道
   - 测试炊烟场景

5. **Phase 4实施** (第7-8周)
   - 开发Web应用
   - 实现API接口
   - Docker部署
   - 性能测试

---

## 📞 联系与支持

如有技术问题或需要进一步讨论，请联系：

- **项目负责人**: EmberGuard Team
- **技术支持**: [GitHub Issues](https://github.com/create-meng/EmberGuard-AI-train/issues)
- **文档更新**: 2026年2月6日

---

**文档结束**

*本技术研究报告基于4个开源项目的深入分析，提供了完整的技术实现路线图。建议按照Phase 1-4的顺序逐步实施，每个阶段都有明确的目标和可交付成果。*
