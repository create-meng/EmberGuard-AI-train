# EmberGuard AI - 智能火灾检测训练系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**基于YOLOv8的智能火灾检测与训练系统**

[功能特性](#功能特性) • [快速开始](#快速开始) • [使用说明](#使用说明) • [项目结构](#项目结构) • [开发路线](#开发路线)

</div>

---

## 📖 项目简介

EmberGuard AI 是一款基于YOLOv8的智能火灾检测训练系统，专注于火灾、烟雾等异常情况的实时识别。本项目为智能视频监控安全增强系统的AI训练模块，提供模型训练、验证、测试和可视化界面。

### 核心目标
- 🔥 高精度火灾/烟雾检测
- ⚡ 实时推理能力
- 🎯 低误报率（目标<2%）
- 🖥️ 友好的GUI操作界面
- 📊 完整的训练验证流程

---

## ✨ 功能特性

### 当前已实现
- ✅ 基于YOLOv8的火灾检测模型训练
- ✅ D-Fire数据集支持（21,527张图像）
- ✅ 图形化操作界面（GUI）
- ✅ 图片/视频/摄像头实时检测
- ✅ 模型验证与性能评估
- ✅ 检测结果可视化保存
- ✅ **LSTM时序分析模块**（新增）
- ✅ **8维特征提取器**（新增）
- ✅ **YOLO+LSTM检测管道**（新增）

### 开发中
- 🚧 LSTM模型训练（数据准备中）
- 🚧 GUI界面LSTM集成
- 🚧 热红外特征融合
- 🚧 多传感器数据融合
- 🚧 边缘设备部署优化

---

## 🚀 快速开始

### 环境要求
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.0 (GPU训练推荐)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/create-meng/EmberGuard-AI-train.git
cd EmberGuard-AI-train
```

2. **安装依赖**
```bash
pip install ultralytics opencv-python pillow
```

3. **准备数据集**
- 下载D-Fire数据集或使用自定义数据集
- 将数据集放置在 `datasets/` 目录下
- 配置 `configs/yolo_fire.yaml`

4. **运行GUI界面**
```bash
python scripts/run_gui.py
```

---

## 📚 使用说明

### 1. 训练YOLOv8模型
```bash
python scripts/train_model.py
```

### 2. 验证模型
```bash
python scripts/validate_model.py
```

### 3. 测试检测

**图片检测**
```bash
python scripts/test_model.py --source image.jpg
```

**视频检测**
```bash
python scripts/test_model.py --source video.mp4
```

**摄像头实时检测**
```bash
python scripts/test_model.py --source 0
```

### 4. LSTM时序分析（新增）

**准备训练数据**
```bash
python scripts/prepare_lstm_data.py
```

**训练LSTM模型**
```bash
python scripts/train_lstm.py --data_dir datasets/lstm_data --epochs 50
```

**测试YOLO+LSTM管道**
```bash
python scripts/test_pipeline.py
```

详细使用说明请参考：`emberguard/README.md`

### 4.1 数字孪生Demo对比测试（digital-twin-web，新增）

项目包含一个轻量的Web演示端，用于在同一套前端UI下快速对比不同推理/降噪/融合策略。

**启动后端**
```bash
python digital-twin-web/backend/app.py
```

浏览器访问：

- **http://127.0.0.1:5000/**

**实验档位切换（用于对比试验）**

在 `digital-twin-web/backend/app.py` 修改：

```py
EXPERIMENT_PROFILE = 'yolo_lstm_denoise_fusion'
```

可选值：

- **`yolo`**：纯YOLO（最终告警=YOLO是否检出 fire/smoke）
- **`yolo_lstm`**：YOLO + LSTM（最终告警=LSTM；LSTM未就绪/前30帧用YOLO兜底）
- **`yolo_lstm_denoise`**：YOLO + LSTM + 降噪（最终告警=LSTM；LSTM未就绪用YOLO兜底；特征级+帧级降噪均开启）
- **`yolo_lstm_fusion`**：YOLO + LSTM + 告警融合（最终告警=时间窗投票/占比 + 迟滞；YOLO强证据可快速触发 fire；LSTM未就绪时用YOLO投票）
- **`yolo_lstm_denoise_fusion`**：YOLO + LSTM + 降噪 + 告警融合（最终告警=时间窗投票/占比 + 迟滞；同时开启两层降噪）

融合策略（方案A）参数含义：

- `windowSize`：滑动时间窗长度（帧数）
- `onFireRatio/offFireRatio`：触发/解除 fire 的占比阈值（迟滞）
- `onSmokeRatio/offSmokeRatio`：触发/解除 smoke 的占比阈值（迟滞）

改完后需要：

- 重启后端（Ctrl+C 后重新运行）
- 浏览器强刷（Ctrl+F5）


### 5. GUI界面使用
运行GUI后可以：
- 选择检测源（图片/视频/摄像头）
- 实时查看检测结果
- 调整检测参数
- 保存检测结果

---

## 📁 项目结构

```
EmberGuard-AI-train/
├── configs/                    # 配置文件
│   ├── ultralytics_settings.json
│   └── yolo_fire.yaml         # 数据集配置
├── datasets/                   # 数据集目录
│   └── D-Fire/                # 火灾检测数据集
├── docs/                       # 文档目录
│   ├── TECHNICAL_RESEARCH.md  # 技术研究报告
│   ├── SUMMARY.md             # 项目总结
│   └── QUICK_START.md         # 快速开始指南
├── emberguard/                 # LSTM时序分析模块 ⭐新增
│   ├── feature_extractor.py   # 特征提取器
│   ├── lstm_model.py          # LSTM模型
│   ├── pipeline.py            # 检测管道
│   └── README.md              # 模块文档
├── models/                     # 预训练模型
│   ├── yolov8n.pt
│   └── yolo11n.pt
├── runs/                       # 训练结果
│   └── detect/train2/weights/
│       └── best.pt            # 最佳YOLOv8模型
├── scripts/                    # 脚本文件
│   ├── run_gui.py             # GUI启动脚本
│   ├── train_model.py         # YOLO训练脚本
│   ├── validate_model.py      # 验证脚本
│   ├── test_model.py          # 测试脚本
│   ├── prepare_lstm_data.py   # LSTM数据准备 ⭐新增
│   ├── train_lstm.py          # LSTM训练脚本 ⭐新增
│   └── test_pipeline.py       # 管道测试 ⭐新增
├── UI/                         # GUI界面模块
│   ├── gui_main.py            # 主界面
│   ├── detection_processor.py # 检测处理器
│   └── ...
├── DEVELOPMENT_LOG.md          # 开发日志 ⭐新增
└── PROJECT_STRUCTURE.md        # 详细结构说明
```

---

## 🛣️ 开发路线

### Phase 1: 基础检测 ✅
- [x] YOLOv8模型集成
- [x] D-Fire数据集训练（50 epochs）
- [x] GUI界面开发
- [x] 基础检测功能

### Phase 2: LSTM时序分析 🚧 (80%完成)
- [x] 8维特征提取器
- [x] LSTM模型架构（2层，211K参数）
- [x] YOLO+LSTM检测管道
- [x] 数据准备工具
- [x] 训练脚本
- [ ] 视频数据收集与标注
- [ ] LSTM模型训练
- [ ] GUI集成
- [ ] 炊烟vs火灾烟雾区分测试

### Phase 3: 多模态融合 📋
- [ ] 温度传感器接口
- [ ] 烟感传感器接口
- [ ] 多传感器数据融合
- [ ] 联动告警机制

### Phase 4: 边缘部署 📋
- [ ] 模型量化优化
- [ ] ONNX/TensorRT导出
- [ ] 树莓派/Jetson适配
- [ ] 性能基准测试

---

## 📊 性能指标

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 检测准确率 | ~95% | >99% |
| 误报率 | ~5% | <2% |
| 推理速度(GPU) | ~30ms | <20ms |
| 模型大小 | ~6MB | <10MB |

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 核心检测框架
- [D-Fire Dataset](https://github.com/gaiasd/DFireDataset) - 火灾检测数据集

---

## 📧 联系方式

- 项目主页: [https://github.com/create-meng/EmberGuard-AI-train](https://github.com/create-meng/EmberGuard-AI-train)
- 问题反馈: [Issues](https://github.com/create-meng/EmberGuard-AI-train/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**

Made with ❤️ by EmberGuard Team

</div>
