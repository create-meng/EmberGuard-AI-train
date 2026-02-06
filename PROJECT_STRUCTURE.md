# 项目结构说明

## 📁 目录结构

```
ultralytics-main/
├── configs/                    # 配置文件
│   ├── ultralytics_settings.json  # Ultralytics本地配置
│   └── yolo_fire.yaml             # 火灾检测数据集配置
├── datasets/                   # 数据集
│   └── D-Fire/                    # 火灾检测数据集
├── detection_saves/            # 检测结果保存目录
├── models/                     # 模型文件
│   ├── yolov8n.pt                 # YOLOv8 nano预训练模型
│   └── yolo11n.pt                 # YOLO11 nano预训练模型
├── runs/                       # 训练和检测运行结果
│   └── detect/
│       └── train2/
│           └── weights/
│               └── best.pt        # 采用D-Fire数据集训练的最佳模型
├── scripts/                    # 脚本文件
│   ├── run_gui.py                 # 运行GUI界面
│   ├── train_model.py             # 训练模型
│   ├── validate_model.py          # 验证模型
│   ├── test_model.py              # 测试模型
│   └── README.md                  # 脚本使用说明
├── UI/                         # GUI界面模块
│   ├── __init__.py                # 包初始化文件
│   ├── config.py                  # GUI配置
│   ├── detection_processor.py     # 检测处理器
│   ├── detection_ui.py            # 检测界面组件
│   ├── file_handler.py            # 文件处理
│   ├── gui_main.py                # 主界面
│   ├── gui_utils.py               # 工具函数
│   ├── main.py                    # UI入口
│   └── README.md                  # UI模块说明
├── ultralytics/                # Ultralytics核心库
└── PROJECT_STRUCTURE.md        # 项目结构说明文档

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
