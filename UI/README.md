# YOLO目标检测GUI应用程序

## 文件结构说明

### 核心文件

- **`main.py`** - 程序入口，创建主窗口并启动应用
- **`gui_main.py`** - 主窗口类，包含模式选择界面和主逻辑
- **`config.py`** - 配置文件和常量定义

### UI组件

- **`detection_ui.py`** - 检测界面UI组件创建
- **`gui_utils.py`** - 线程安全的GUI更新工具类

### 功能模块

- **`detection_processor.py`** - 检测处理逻辑（屏幕、摄像头、文件检测）
- **`file_handler.py`** - 文件处理和保存功能

## 使用方法

### 方式一：使用运行脚本（推荐）

在项目根目录运行：

```bash
python run_gui.py
```

### 方式二：直接运行模块

```bash
python -m hys.main
```

### 方式三：在代码中导入

```python
import tkinter as tk

from hys import YOLODetectionGUI

root = tk.Tk()
app = YOLODetectionGUI(root)
root.mainloop()
```

## 模块说明

### config.py

包含所有配置常量：

- `MODEL_PATH` - YOLO模型路径
- `WINDOW_TITLE` - 窗口标题
- `WINDOW_SIZE` - 窗口大小
- `DND_AVAILABLE` - 拖拽功能是否可用

### gui_main.py

主窗口类 `YOLODetectionGUI`，负责：

- 模式选择界面
- 检测界面管理
- 文件选择和处理
- 结果保存

### detection_processor.py

检测处理器 `DetectionProcessor`，负责：

- 屏幕检测
- 摄像头检测
- 文件检测（图片/视频）

### file_handler.py

文件处理类 `FileHandler`，提供：

- 文件选择
- 文件类型判断
- 图片预览加载
- 检测结果保存

### gui_utils.py

线程安全的GUI更新器 `ThreadSafeGUIUpdater`，确保：

- 所有GUI更新在主线程执行
- 避免线程安全问题
- 安全的窗口关闭处理

### detection_ui.py

检测界面UI组件创建类 `DetectionUI`，提供：

- 统一的检测界面创建
- 文件检测特殊UI组件

## 依赖项

- `ultralytics` - YOLO模型
- `tkinter` - GUI框架
- `opencv-python` - 图像处理
- `PIL` (Pillow) - 图像显示
- `tkinterdnd2` (可选) - 文件拖拽支持

## 注意事项

1. 确保 `yolov8n.pt` 模型文件在项目根目录
2. 安装可选依赖 `tkinterdnd2` 以启用文件拖拽功能：
   ```bash
   pip install tkinterdnd2
   ```
