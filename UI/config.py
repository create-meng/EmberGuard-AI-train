"""
配置和常量定义
"""
import tkinter as tk

# 尝试导入拖拽支持库（可选）
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    # 创建一个兼容类
    class TkinterDnD:
        class Tk(tk.Tk):
            pass
    # 定义 DND_FILES 占位符（即使不使用）
    DND_FILES = None

# 模型路径
MODEL_PATH = "runs/detect/train2/weights/best.pt"

# GUI配置
WINDOW_TITLE = "YOLO 目标检测界面"
WINDOW_SIZE = "1200x800"

# 视频显示配置
MAX_DISPLAY_WIDTH = 1000
MAX_DISPLAY_HEIGHT = 600

# 信息显示配置
MAX_INFO_LINES = 50

# GUI更新队列处理间隔（毫秒）
GUI_QUEUE_INTERVAL = 100

# 默认检测参数
DEFAULT_CONF = 0.25  # 默认置信度阈值 (0-1)
DEFAULT_IOU = 0.45   # 默认IoU阈值 (0-1)

# 默认保存文件夹
DEFAULT_SAVE_DIR = "detection_saves"  # 默认检测结果保存文件夹

