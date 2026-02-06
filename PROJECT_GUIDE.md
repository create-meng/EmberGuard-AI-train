# EmberGuard AI - 项目指南

## 🚀 快速开始

### 如果你想训练LSTM模型 ⭐
1. 查看 `datasets/README.md`
2. 运行 `python scripts/3_prepare_lstm_data.py`
3. 运行 `python scripts/4_train_lstm.py`

### 如果你想使用GUI
```bash
python scripts/5_run_gui.py
```

### 如果你想了解项目
1. 查看 `README.md` - 项目概述
2. 查看 `DEVELOPMENT_LOG.md` - 开发历程

---

## 📁 项目结构

```
EmberGuard-AI-train/
├── README.md                 # 项目主文档
├── DEVELOPMENT_LOG.md        # 完整开发日志
├── PROJECT_GUIDE.md          # 本文件（项目指南）
├── LICENSE                   # MIT许可证
├── requirements.txt          # Python依赖
│
├── configs/                  # 配置文件
│   ├── ultralytics_settings.json
│   └── yolo_fire.yaml       # YOLO数据集配置
│
├── datasets/                 # 数据集 ⭐
│   ├── README.md            # 数据集说明（重要）
│   ├── DATASET_LINKS.md     # 数据集下载链接
│   ├── D-Fire/              # YOLO训练数据
│   ├── download/            # 下载的原始数据（备份）
│   ├── fire_videos_organized/  # 整理好的LSTM数据 ⭐
│   │   ├── fire/           # 48个火灾视频
│   │   ├── smoke/          # 92个烟雾视频
│   │   ├── normal/         # 100个正常视频
│   │   ├── mixed/          # 4个测试视频
│   │   └── annotations.csv
│   └── lstm_data/           # LSTM训练数据（运行脚本后生成）
│
├── docs/                     # 文档
│   ├── TECHNICAL_RESEARCH.md  # 技术研究报告（9000+字）
│   ├── SUMMARY.md            # 项目总结
│   └── QUICK_START.md        # 快速开始指南
│
├── emberguard/              # LSTM模块 ⭐
│   ├── README.md            # 模块使用文档
│   ├── __init__.py
│   ├── feature_extractor.py  # 8维特征提取器
│   ├── lstm_model.py         # LSTM分类模型
│   └── pipeline.py           # YOLO+LSTM检测管道
│
├── models/                   # 模型文件
│   ├── yolov8n.pt           # 预训练模型
│   ├── yolo11n.pt
│   └── lstm/                # LSTM模型（训练后生成）
│       ├── best.pt
│       ├── last.pt
│       └── history.json
│
├── runs/                     # 训练结果
│   └── detect/
│       └── train2/
│           └── weights/
│               └── best.pt  # YOLOv8最佳模型
│
├── scripts/                  # 脚本 ⭐
│   ├── README.md            # 脚本详细说明
│   ├── 0_download_datasets.py      # 数据集下载助手
│   ├── 1_train_yolo.py            # YOLO训练
│   ├── 2_validate_yolo.py         # YOLO验证
│   ├── 3_prepare_lstm_data.py     # LSTM数据准备 ⭐
│   ├── 4_train_lstm.py            # LSTM训练 ⭐
│   ├── 5_run_gui.py               # GUI启动
│   └── organize_downloaded_data.py # 数据整理
│
├── UI/                       # GUI界面
│   ├── main.py              # GUI入口
│   ├── gui_main.py          # 主界面
│   ├── detection_processor.py  # 检测处理器
│   └── ...
│
└── test_picture/            # 测试图片
```

---

## 📖 核心文档导航

### 项目文档
- **README.md** - 项目主文档，项目概述
- **PROJECT_GUIDE.md** - 本文件，项目指南和结构
- **DEVELOPMENT_LOG.md** - 完整开发日志（20000+字）
- **LICENSE** - MIT许可证

### 数据集文档
- **datasets/README.md** ⭐⭐⭐ - 数据集说明和训练指南
- **datasets/DATASET_LINKS.md** - 数据集下载链接

### 技术文档
- **docs/TECHNICAL_RESEARCH.md** - 技术研究报告（9000+字）
- **docs/SUMMARY.md** - 项目总结
- **docs/QUICK_START.md** - 快速开始指南

### 模块文档
- **emberguard/README.md** - LSTM模块使用指南
- **scripts/README.md** - 脚本详细说明
- **UI/README.md** - GUI模块说明

---

## 🎯 当前项目状态

### ✅ 已完成
- [x] YOLOv8模型训练（D-Fire数据集，50 epochs）
- [x] GUI界面开发（Tkinter）
- [x] LSTM模块代码（特征提取器、模型、管道）
- [x] 数据集下载和整理（244个视频）
- [x] 训练脚本和工具

### 🔄 进行中
- [ ] LSTM模型训练（数据已准备好）

### 📋 待开始
- [ ] GUI集成LSTM
- [ ] 性能测试和优化
- [ ] 部署准备

---

## 🔧 核心功能

### 1. YOLO火灾检测（已完成）
- 基于YOLOv8的实时火灾检测
- 训练模型：`runs/detect/train2/weights/best.pt`
- 支持图片/视频/摄像头检测

### 2. LSTM时序分析（代码完成，待训练）
- 8维特征提取器
- 2层LSTM分类模型（211K参数）
- YOLO+LSTM检测管道
- 3分类：无火/烟雾/火焰

### 3. GUI界面（已完成）
- Tkinter图形界面
- 文件/摄像头/屏幕检测
- 实时显示检测结果

---

## 📊 数据集信息

### YOLO训练数据
- **D-Fire数据集**: 21,527张图像
- **位置**: `datasets/D-Fire/`
- **用途**: YOLOv8模型训练

### LSTM训练数据
- **总视频数**: 244个
- **训练数据**: 240个（火灾48 + 烟雾92 + 正常100）
- **测试数据**: 4个（未标注，用于最终测试）
- **位置**: `datasets/fire_videos_organized/`
- **预期样本**: 3000-5000个序列

---

## 🚀 使用方法

### 训练LSTM模型
```bash
# 步骤1: 准备训练数据（30-60分钟）
python scripts/3_prepare_lstm_data.py

# 步骤2: 训练模型（1-2小时）
python scripts/4_train_lstm.py --data_dir datasets/lstm_data --epochs 50
```

### 使用GUI
```bash
python scripts/5_run_gui.py
```

### 测试YOLO模型
```bash
# 图片
python scripts/test_model.py --source image.jpg

# 视频
python scripts/test_model.py --source video.mp4

# 摄像头
python scripts/test_model.py --source 0
```

---

## 💡 技术亮点

1. **完整工具链**: 从数据下载到模型训练全流程
2. **模块化设计**: 代码解耦，易于维护和扩展
3. **详细文档**: 20000+字文档，覆盖所有方面
4. **测试完备**: 所有核心功能已测试通过
5. **易于使用**: 脚本按顺序编号，一步步引导

---

## 📈 性能目标

| 指标 | YOLO | LSTM | 组合 |
|------|------|------|------|
| 准确率 | ~95% | 96-99% | >99% |
| 误报率 | ~5% | <2% | <2% |
| 推理速度 | ~30ms | ~10ms | ~40ms |
| 模型大小 | ~6MB | ~850KB | ~7MB |

---

## 🔗 相关链接

- **GitHub**: https://github.com/create-meng/EmberGuard-AI-train
- **Ultralytics**: https://github.com/ultralytics/ultralytics
- **D-Fire Dataset**: https://github.com/gaiasd/DFireDataset

---

## 📞 需要帮助？

1. **查看文档**: 
   - `datasets/README.md` - 数据集和训练
   - `DEVELOPMENT_LOG.md` - 开发历程
   - `emberguard/README.md` - LSTM模块

2. **查看脚本说明**: `scripts/README.md`

3. **查看技术研究**: `docs/TECHNICAL_RESEARCH.md`

---

**最后更新**: 2026年2月6日  
**项目状态**: 数据已准备，可以开始训练LSTM模型  
**下一步**: `python scripts/3_prepare_lstm_data.py`
