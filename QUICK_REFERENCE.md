# EmberGuard AI - 快速参考

## 🚀 快速开始

### 1. YOLO训练（已完成✅）
```bash
python scripts/1_train_yolo.py
python scripts/2_validate_yolo.py
```

### 2. LSTM训练（待完成）
```bash
# 下载数据集
kaggle datasets download -d ritupande/fire-detection-from-cctv

# 准备数据
python scripts/3_prepare_lstm_data.py

# 训练模型
python scripts/4_train_lstm.py --data_dir datasets/lstm_data --epochs 50
```

### 3. 运行GUI
```bash
python scripts/5_run_gui.py
```

---

## 📂 项目结构

```
EmberGuard-AI-train/
├── configs/              # 配置文件
├── datasets/             # 数据集
│   ├── D-Fire/          # YOLO训练数据（已有）
│   └── lstm_data/       # LSTM训练数据（待准备）
├── docs/                 # 文档
├── emberguard/          # LSTM模块
│   ├── feature_extractor.py
│   ├── lstm_model.py
│   └── pipeline.py
├── models/              # 预训练模型
├── runs/                # 训练结果
│   └── detect/train2/weights/best.pt  # YOLOv8模型
├── scripts/             # 脚本（按顺序编号）
│   ├── 1_train_yolo.py
│   ├── 2_validate_yolo.py
│   ├── 3_prepare_lstm_data.py
│   ├── 4_train_lstm.py
│   └── 5_run_gui.py
└── UI/                  # GUI界面
```

---

## 📊 推荐数据集

| 数据集 | 规模 | 链接 | 推荐度 |
|--------|------|------|--------|
| Fire Detection from CCTV | 1000视频 | [Kaggle](https://www.kaggle.com/datasets/ritupande/fire-detection-from-cctv) | ⭐⭐⭐⭐⭐ |
| Fire and Smoke Dataset | 900视频 | [Mendeley](https://data.mendeley.com/datasets/gjxz5w7xp7/1) | ⭐⭐⭐⭐⭐ |
| MIVIA Fire Detection | 54视频 | [MIVIA](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) | ⭐⭐⭐⭐ |

**推荐**: 下载Kaggle数据集（1000视频）作为主要训练数据

---

## 🎯 当前状态

### ✅ 已完成
- [x] YOLOv8模型训练（D-Fire数据集）
- [x] GUI界面开发
- [x] LSTM模块代码（特征提取器、模型、管道）
- [x] 训练脚本（数据准备、训练）
- [x] 项目文档

### 🔄 进行中
- [ ] 下载视频数据集
- [ ] 标注视频数据
- [ ] 训练LSTM模型

### 📋 待开始
- [ ] GUI集成LSTM
- [ ] 性能测试
- [ ] 部署优化

---

## 💡 使用示例

### Python API
```python
from emberguard import FireDetectionPipeline

# 创建管道
pipeline = FireDetectionPipeline(
    yolo_model_path='runs/detect/train2/weights/best.pt',
    lstm_model_path='models/lstm/best.pt'  # 可选
)

# 检测单帧
result = pipeline.detect_frame(frame)
print(result['has_detection'])
print(result['lstm_class_name'])  # 无火/烟雾/火焰

# 处理视频
results = pipeline.process_video('input.mp4', 'output.mp4')
```

### 命令行
```bash
# YOLO训练
python scripts/1_train_yolo.py

# LSTM训练
python scripts/4_train_lstm.py \
    --data_dir datasets/lstm_data \
    --epochs 50 \
    --batch_size 32

# GUI
python scripts/5_run_gui.py
```

---

## 📈 性能目标

| 指标 | 当前 | 目标 |
|------|------|------|
| YOLO准确率 | ~95% | >95% ✅ |
| LSTM准确率 | - | >99% |
| 误报率 | ~5% | <2% |
| 推理速度 | ~30ms | <50ms |

---

## 🔗 重要文档

- **开发日志**: `DEVELOPMENT_LOG.md` - 完整开发记录
- **数据集搜索**: `DATASET_SEARCH.md` - 数据集推荐
- **LSTM模块**: `emberguard/README.md` - 模块使用指南
- **脚本说明**: `scripts/README.md` - 脚本详细说明
- **技术研究**: `docs/TECHNICAL_RESEARCH.md` - 技术方案

---

## 📞 下一步行动

1. **下载数据集** (Kaggle Fire Detection)
2. **标注数据** (无火/烟雾/火焰)
3. **运行脚本3** (准备LSTM数据)
4. **运行脚本4** (训练LSTM模型)
5. **测试效果** (GUI界面)

---

**最后更新**: 2026年2月6日  
**项目状态**: Phase 2 进行中（80%）
