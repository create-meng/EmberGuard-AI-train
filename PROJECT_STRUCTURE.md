# EmberGuard AI - 项目结构说明

## 📁 目录结构

```
EmberGuard-AI-train/
├── configs/                    # 配置文件
│   ├── ultralytics_settings.json  # Ultralytics本地配置
│   └── yolo_fire.yaml             # 火灾检测数据集配置
├── datasets/                   # 数据集目录
│   └── D-Fire/                    # 火灾检测数据集
│       ├── train/                 # 训练集
│       ├── val/                   # 验证集
│       └── test/                  # 测试集
├── detection_saves/            # 检测结果保存目录
├── models/                     # 预训练模型
│   ├── yolov8n.pt                 # YOLOv8 nano预训练模型
│   └── yolo11n.pt                 # YOLO11 nano预训练模型
├── runs/                       # 训练和检测运行结果
│   └── detect/
│       └── train2/
│           ├── weights/
│           │   └── best.pt        # 训练的最佳模型
│           ├── results.png        # 训练结果图表
│           └── confusion_matrix.png
├── scripts/                    # 脚本文件
│   ├── run_gui.py                 # 启动GUI界面
│   ├── train_model.py             # 训练模型
│   ├── validate_model.py          # 验证模型
│   ├── test_model.py              # 测试模型
│   └── README.md                  # 脚本使用说明
├── UI/                         # GUI界面模块
│   ├── __init__.py                # 包初始化
│   ├── config.py                  # GUI配置
│   ├── detection_processor.py     # 检测处理器
│   ├── detection_ui.py            # 检测界面组件
│   ├── file_handler.py            # 文件处理
│   ├── gui_main.py                # 主界面
│   ├── gui_utils.py               # 工具函数
│   ├── main.py                    # UI入口
│   └── README.md                  # UI模块说明
├── .gitignore                  # Git忽略文件配置
├── LICENSE                     # MIT许可证
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖包列表
└── PROJECT_STRUCTURE.md        # 本文件

```

## 📝 核心文件说明

### 配置文件
- **configs/yolo_fire.yaml**: 数据集路径、类别定义、训练参数配置
- **configs/ultralytics_settings.json**: Ultralytics框架的本地设置

### 脚本文件
- **scripts/run_gui.py**: 启动图形界面的入口脚本
- **scripts/train_model.py**: 模型训练脚本，支持自定义参数
- **scripts/validate_model.py**: 模型验证脚本，评估性能指标
- **scripts/test_model.py**: 模型测试脚本，支持图片/视频/摄像头

### GUI模块
- **UI/main.py**: GUI应用程序入口
- **UI/gui_main.py**: 主窗口界面实现
- **UI/detection_processor.py**: 检测逻辑处理
- **UI/detection_ui.py**: 检测界面UI组件
- **UI/file_handler.py**: 文件选择和保存处理

## 🔧 依赖说明

本项目使用 `pip install ultralytics` 安装YOLO框架，不再包含ultralytics源码。

所有依赖包列在 `requirements.txt` 中。

```

## 🚀 使用方法

### 1. 运行GUI应用（推荐）
```bash
python scripts/run_gui.py
```

### 2. 测试模型
```bash
# 测试图片
python scripts/test_model.py --source image.jpg

# 测试视频
python scripts/test_model.py --source video.mp4

# 测试摄像头
python scripts/test_model.py --source 0
```

### 3. 训练模型
```bash
python scripts/train_model.py
```

### 4. 验证模型
```bash
python scripts/validate_model.py
```