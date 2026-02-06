"""程序入口."""

import os
import sys
import tkinter as tk

# 添加父目录到路径，以便导入hys包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 尝试相对导入（当作为包的一部分运行时）
    from .config import DND_AVAILABLE, TkinterDnD
    from .gui_main import YOLODetectionGUI
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行时）
    from UI.config import DND_AVAILABLE, TkinterDnD
    from UI.gui_main import YOLODetectionGUI


def main():
    """主函数."""
    # 使用TkinterDnD来支持拖拽功能（如果可用）
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        print("提示: 安装 tkinterdnd2 库可启用文件拖拽功能: pip install tkinterdnd2")

    YOLODetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
