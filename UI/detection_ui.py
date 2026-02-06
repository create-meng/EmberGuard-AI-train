"""
检测界面UI组件创建
"""
import tkinter as tk
from tkinter import ttk
import sys
import os

# 处理相对导入和绝对导入
try:
    # 尝试相对导入（当作为包的一部分运行时）
    from .config import DND_AVAILABLE, DND_FILES, DEFAULT_CONF, DEFAULT_IOU
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行时）
    # 添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from hys.config import DND_AVAILABLE, DND_FILES, DEFAULT_CONF, DEFAULT_IOU
    except ImportError:
        # 如果都失败，使用默认值
        DND_AVAILABLE = False
        DND_FILES = None
        DEFAULT_CONF = 0.25
        DEFAULT_IOU = 0.45


class DetectionUI:
    """检测界面UI组件"""
    
    @staticmethod
    def create_detection_ui(root, mode_name, control_frame_callback=None):
        """创建检测界面"""
        # 清除现有组件
        for widget in root.winfo_children():
            widget.destroy()
        
        # 主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题栏
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
        
        title_label = ttk.Label(header_frame, text=f"YOLO {mode_name}", 
                                 font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # 返回按钮
        back_button = ttk.Button(header_frame, text="← 返回模式选择", 
                                 command=control_frame_callback)
        back_button.pack(side=tk.RIGHT)
        
        # 视频显示区域
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # 视频标签（居中显示，使用grid布局防止窗口收缩）
        video_label = ttk.Label(video_frame, text="准备中...", 
                                background="black", foreground="white",
                                font=("Arial", 12), anchor="center")
        video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 控制按钮框架
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, pady=10)
        
        # 参数设置框架
        params_frame = ttk.LabelFrame(control_frame, text="检测参数", padding="5")
        params_frame.pack(side=tk.LEFT, padx=10)
        
        # 使用已导入的默认值（在文件顶部已导入）
        
        # Conf参数输入
        conf_frame = ttk.Frame(params_frame)
        conf_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(conf_frame, text="Conf:", font=("Arial", 9)).pack(side=tk.LEFT)
        conf_var = tk.StringVar(value=str(DEFAULT_CONF))
        conf_entry = ttk.Entry(conf_frame, textvariable=conf_var, width=6, font=("Arial", 9))
        conf_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(conf_frame, text="(0-1)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
        
        # IOU参数输入
        iou_frame = ttk.Frame(params_frame)
        iou_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(iou_frame, text="IOU:", font=("Arial", 9)).pack(side=tk.LEFT)
        iou_var = tk.StringVar(value=str(DEFAULT_IOU))
        iou_entry = ttk.Entry(iou_frame, textvariable=iou_var, width=6, font=("Arial", 9))
        iou_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(iou_frame, text="(0-1)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
        
        # 参数说明提示（换行显示更清晰）
        help_text = "Conf:置信度阈值(0-1,越高越严格) | IOU:重叠度阈值(0-1,越高越宽松)"
        help_label = ttk.Label(params_frame, 
                              text=help_text,
                              font=("Arial", 8), foreground="gray")
        help_label.pack(side=tk.LEFT, padx=10)
        
        # 启动/停止按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=10)
        
        start_button = ttk.Button(button_frame, text="开始检测", 
                                  command=lambda: None)  # 将在外部设置
        start_button.pack(side=tk.LEFT, padx=5)
        
        stop_button = ttk.Button(button_frame, text="停止检测", 
                                 state=tk.DISABLED)
        stop_button.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        status_label = ttk.Label(control_frame, text="状态: 未启动", 
                                 font=("Arial", 10))
        status_label.pack(side=tk.LEFT, padx=20)
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        info_frame.columnconfigure(0, weight=1)
        
        info_text = tk.Text(info_frame, height=5, width=80, wrap=tk.WORD)
        info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        info_text.configure(yscrollcommand=scrollbar.set)
        
        return {
            'main_frame': main_frame,
            'video_label': video_label,
            'control_frame': control_frame,
            'start_button': start_button,
            'stop_button': stop_button,
            'status_label': status_label,
            'info_text': info_text,
            'conf_var': conf_var,
            'iou_var': iou_var,
            'conf_entry': conf_entry,
            'iou_entry': iou_entry
        }
    
    @staticmethod
    def setup_file_detection_ui(control_frame, video_label, select_file_callback, 
                                 on_file_drop_callback, save_results_callback):
        """设置文件检测界面的额外组件"""
        # 添加文件选择区域
        file_select_frame = ttk.Frame(control_frame)
        file_select_frame.pack(pady=10)
        
        select_button = ttk.Button(file_select_frame, text="选择文件", 
                                   command=select_file_callback)
        select_button.pack(side=tk.LEFT, padx=5)
        
        # 启用拖拽功能（如果可用）
        if DND_AVAILABLE:
            video_label.drop_target_register(DND_FILES)
            video_label.dnd_bind('<<Drop>>', on_file_drop_callback)
        
        # 提示文字
        hint_label = ttk.Label(file_select_frame, 
                              text="或拖拽文件到显示区域",
                              font=("Arial", 9), foreground="gray")
        hint_label.pack(side=tk.LEFT, padx=10)
        
        # 保存结果按钮
        save_button = ttk.Button(control_frame, text="保存检测结果", 
                                command=save_results_callback, 
                                state=tk.DISABLED)
        save_button.pack(side=tk.LEFT, padx=5)
        
        return save_button

