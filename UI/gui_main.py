"""
主窗口和模式选择界面
"""
import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue
import os
import cv2
import sys

# 处理相对导入和绝对导入
try:
    # 尝试相对导入（当作为包的一部分运行时）
    from .config import MODEL_PATH, WINDOW_TITLE, WINDOW_SIZE, DND_AVAILABLE, DND_FILES
    from .detection_ui import DetectionUI
    from .gui_utils import ThreadSafeGUIUpdater
    from .detection_processor import DetectionProcessor
    from .file_handler import FileHandler
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行时）
    # 添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hys.config import MODEL_PATH, WINDOW_TITLE, WINDOW_SIZE, DND_AVAILABLE, DND_FILES
    from hys.detection_ui import DetectionUI
    from hys.gui_utils import ThreadSafeGUIUpdater
    from hys.detection_processor import DetectionProcessor
    from hys.file_handler import FileHandler

from ultralytics import YOLO


class YOLODetectionGUI:
    """YOLO目标检测GUI主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        
        # 模型路径（初始使用默认路径）
        self.model_path = MODEL_PATH
        self.yolo = None
        self.model_loaded = False
        
        # LSTM设置
        self.use_lstm = False  # 默认不使用LSTM
        self.lstm_model_path = 'models/lstm/train/best.pt'  # 默认LSTM模型路径
        
        # 加载YOLO模型
        self.load_model(self.model_path)
        
        # 检测模式：'screen', 'camera', 'file', None
        self.detection_mode = None
        self.is_running = False
        
        # 文件检测设置
        self.selected_file_path = None
        self.detection_results = None
        self.detection_results_info = None
        self.detection_file_type = None
        self.detection_has_results = False
        
        # 参数输入控件引用（初始化为None）
        self.conf_var = None
        self.iou_var = None
        
        # 保存文件夹设置
        try:
            from .config import DEFAULT_SAVE_DIR
        except ImportError:
            from hys.config import DEFAULT_SAVE_DIR
        self.save_dir = DEFAULT_SAVE_DIR
        # 确保默认文件夹存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 当前显示的图像
        self.current_frame = None
        
        # 线程安全的GUI更新队列
        self.gui_queue = Queue()
        self.gui_updater = ThreadSafeGUIUpdater(self.root, self.gui_queue)
        self.root.after(100, self.gui_updater.process_gui_queue)
        
        # UI组件引用
        self.ui_components = {}
        self.buttons = {}
        self.detection_processor = None
        
        # 显示模式选择界面
        self.show_mode_selection()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self, model_path):
        """加载YOLO模型"""
        try:
            self.yolo = YOLO(model_path, task="detect")
            self.model_path = model_path
            self.model_loaded = True
            return True
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("错误", f"无法加载YOLO模型: {str(e)}")
    
    def toggle_lstm(self):
        """切换LSTM开关"""
        self.use_lstm = self.lstm_var.get()
        
        # 更新状态标签
        if hasattr(self, 'lstm_status_label'):
            if self.use_lstm:
                self.lstm_status_label.config(text="✓ LSTM已启用", foreground="green")
            else:
                self.lstm_status_label.config(text="○ LSTM未启用（仅使用YOLO）", foreground="gray")
            return False
    
    def select_model(self):
        """选择模型文件"""
        # 如果正在检测，先停止
        if self.is_running:
            result = messagebox.askyesno("提示", "当前正在检测中，更换模型将停止检测。\n\n是否继续？")
            if not result:
                return
            self.stop_detection()
        
        from tkinter import filedialog
        model_path = filedialog.askopenfilename(
            title="选择YOLO模型文件",
            filetypes=[
                ("PyTorch模型", "*.pt"),
                ("所有文件", "*.*")
            ],
            initialdir="."
        )
        
        if model_path:
            # 尝试加载模型
            if self.load_model(model_path):
                # 更新界面显示
                self.show_mode_selection()
                messagebox.showinfo("成功", f"模型加载成功：\n{os.path.basename(model_path)}")
            else:
                # 如果加载失败，尝试恢复默认模型
                if os.path.exists(MODEL_PATH):
                    self.load_model(MODEL_PATH)
                    self.show_mode_selection()
    
    def show_mode_selection(self):
        """显示模式选择界面"""
        # 保存当前窗口大小
        # 清除现有组件
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # 标题
        title_label = ttk.Label(main_frame, text="YOLO 目标检测系统", 
                               font=("Arial", 20, "bold"))
        title_label.pack(pady=20)
        
        # 副标题
        subtitle_label = ttk.Label(main_frame, text="请选择检测模式", 
                                    font=("Arial", 12))
        subtitle_label.pack(pady=10)
        
        # 模型选择区域
        model_frame = ttk.LabelFrame(main_frame, text="模型设置", padding="10")
        model_frame.pack(pady=15, padx=20, fill=tk.X)
        
        # 当前模型显示
        model_info_frame = ttk.Frame(model_frame)
        model_info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_info_frame, text="当前模型：", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # 显示模型文件名（如果路径太长，显示文件名；否则显示完整路径）
        if self.model_path:
            model_name = os.path.basename(self.model_path)
            # 如果文件名太长，截断
            if len(model_name) > 30:
                model_name = model_name[:27] + "..."
        else:
            model_name = "未加载"
        
        model_status_label = ttk.Label(model_info_frame, text=model_name, 
                                       font=("Arial", 10, "bold"),
                                       foreground="blue" if self.model_loaded else "red")
        model_status_label.pack(side=tk.LEFT, padx=5)
        
        # 添加完整路径提示（鼠标悬停时显示）
        if self.model_path and os.path.exists(self.model_path):
            full_path = os.path.abspath(self.model_path)
            model_status_label.bind("<Enter>", lambda e: self._show_tooltip(e, full_path))
            model_status_label.bind("<Leave>", lambda e: self._hide_tooltip())
        
        # 选择模型按钮
        select_model_button = ttk.Button(model_frame, text="📦 选择模型", 
                                        command=self.select_model,
                                        width=20)
        select_model_button.pack(pady=5)
        
        # 模型状态提示
        if not self.model_loaded:
            status_text = "⚠️ 模型未加载，请先选择模型"
            status_color = "red"
        else:
            status_text = "✓ 模型已加载"
            status_color = "green"
        
        status_hint = ttk.Label(model_frame, text=status_text, 
                               font=("Arial", 9),
                               foreground=status_color)
        status_hint.pack(pady=2)
        
        # LSTM设置区域
        lstm_frame = ttk.LabelFrame(main_frame, text="LSTM时序分析（可选）", padding="10")
        lstm_frame.pack(pady=15, padx=20, fill=tk.X)
        
        # LSTM开关
        lstm_switch_frame = ttk.Frame(lstm_frame)
        lstm_switch_frame.pack(fill=tk.X, pady=5)
        
        self.lstm_var = tk.BooleanVar(value=self.use_lstm)
        lstm_checkbox = ttk.Checkbutton(lstm_switch_frame, 
                                        text="启用LSTM时序分析（提高准确率，降低误报）",
                                        variable=self.lstm_var,
                                        command=self.toggle_lstm)
        lstm_checkbox.pack(side=tk.LEFT, padx=5)
        
        # LSTM状态提示
        lstm_status_text = "✓ LSTM已启用" if self.use_lstm else "○ LSTM未启用（仅使用YOLO）"
        lstm_status_color = "green" if self.use_lstm else "gray"
        
        self.lstm_status_label = ttk.Label(lstm_frame, text=lstm_status_text,
                                          font=("Arial", 9),
                                          foreground=lstm_status_color)
        self.lstm_status_label.pack(pady=2)
        
        # LSTM说明
        lstm_hint = ttk.Label(lstm_frame,
                             text="LSTM通过分析30帧序列来判断火灾，准确率更高但需要更多时间",
                             font=("Arial", 8),
                             foreground="gray")
        lstm_hint.pack(pady=2)
        
        # 保存文件夹设置区域
        save_frame = ttk.LabelFrame(main_frame, text="保存设置", padding="10")
        save_frame.pack(pady=15, padx=20, fill=tk.X)
        
        # 当前保存文件夹显示
        save_info_frame = ttk.Frame(save_frame)
        save_info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(save_info_frame, text="保存文件夹：", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # 显示保存文件夹路径（如果路径太长，截断）
        save_dir_display = self.save_dir
        if len(save_dir_display) > 50:
            save_dir_display = "..." + save_dir_display[-47:]
        
        save_dir_label = ttk.Label(save_info_frame, text=save_dir_display, 
                                   font=("Arial", 9),
                                   foreground="blue")
        save_dir_label.pack(side=tk.LEFT, padx=5)
        
        # 添加完整路径提示
        full_save_path = os.path.abspath(self.save_dir)
        save_dir_label.bind("<Enter>", lambda e: self._show_tooltip(e, full_save_path))
        save_dir_label.bind("<Leave>", lambda e: self._hide_tooltip())
        
        # 选择保存文件夹按钮
        select_save_button = ttk.Button(save_frame, text="📁 选择保存文件夹", 
                                       command=self.select_save_folder,
                                       width=20)
        select_save_button.pack(pady=5)
        
        # 保存文件夹说明
        save_hint = ttk.Label(save_frame, 
                             text="检测到目标时，会自动保存帧到此文件夹",
                             font=("Arial", 8),
                             foreground="gray")
        save_hint.pack(pady=2)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=30)
        
        # 屏幕检测按钮
        screen_button = ttk.Button(button_frame, text="🖥️ 屏幕检测", 
                                    command=lambda: self.select_mode('screen'),
                                    width=25)
        screen_button.pack(pady=15, padx=10)
        
        # 摄像头检测按钮
        camera_button = ttk.Button(button_frame, text="📷 摄像头检测", 
                                    command=lambda: self.select_mode('camera'),
                                    width=25)
        camera_button.pack(pady=15, padx=10)
        
        # 文件检测按钮
        file_button = ttk.Button(button_frame, text="📁 文件检测（图片/视频）", 
                                  command=lambda: self.select_mode('file'),
                                  width=25)
        file_button.pack(pady=15, padx=10)
        
        # 说明文字
        info_label = ttk.Label(main_frame, 
                               text="提示：文件检测模式下，您可以拖拽文件到窗口或点击按钮选择文件",
                               font=("Arial", 9), foreground="gray")
        info_label.pack(pady=20)
    
    def select_mode(self, mode):
        """选择检测模式"""
        if not self.model_loaded:
            messagebox.showerror("错误", "YOLO模型未加载，无法进行检测\n\n请先点击'选择模型'按钮加载模型")
            return
        
        self.detection_mode = mode
        
        if mode == 'screen':
            self.setup_screen_detection()
        elif mode == 'camera':
            self.setup_camera_detection()
        elif mode == 'file':
            self.setup_file_detection()
    
    def setup_screen_detection(self):
        """设置屏幕检测界面"""
        self.create_detection_ui("屏幕检测")
    
    def setup_camera_detection(self):
        """设置摄像头检测界面"""
        self.create_detection_ui("摄像头检测")
    
    def setup_file_detection(self):
        """设置文件检测界面"""
        self.create_detection_ui("文件检测")
        
        # 清空之前选择的文件和结果
        self.selected_file_path = None
        self.detection_results = None
        self.detection_results_info = None
        
        # 设置文件检测的额外UI
        save_button = DetectionUI.setup_file_detection_ui(
            self.ui_components['control_frame'],
            self.ui_components['video_label'],
            self.select_file,
            self.on_file_drop,
            self.save_detection_results
        )
        self.buttons['save'] = save_button
        
        # 更新状态提示
        self.gui_updater.update_status(self.ui_components['status_label'], "请选择文件")
    
    def create_detection_ui(self, mode_name):
        """创建检测界面"""
        ui_dict = DetectionUI.create_detection_ui(
            self.root, 
            mode_name, 
            self.show_mode_selection
        )
        
        self.ui_components = ui_dict
        
        # 设置按钮命令
        ui_dict['start_button'].config(command=self.toggle_detection)
        ui_dict['stop_button'].config(command=self.stop_detection)
        
        # 保存按钮引用
        self.buttons = {
            'start': ui_dict['start_button'],
            'stop': ui_dict['stop_button']
        }
        
        # 获取LSTM设置（如果有）
        use_lstm = getattr(self, 'use_lstm', False)
        lstm_model_path = getattr(self, 'lstm_model_path', None)
        
        # 创建检测处理器
        self.detection_processor = DetectionProcessor(
            self.yolo,
            self.gui_updater,
            self.buttons,
            ui_dict['status_label'],
            ui_dict['info_text'],
            ui_dict['video_label'],
            use_lstm=use_lstm,
            lstm_model_path=lstm_model_path
        )
        
        # 设置保存文件夹（仅在屏幕和摄像头检测模式下）
        if self.detection_mode in ['screen', 'camera']:
            self.detection_processor.set_save_dir(self.save_dir)
            # 重置保存帧计数
            self.detection_processor.save_frame_count = 0
        
        # 保存参数输入控件的引用
        if 'conf_var' in ui_dict and 'iou_var' in ui_dict:
            self.conf_var = ui_dict['conf_var']
            self.iou_var = ui_dict['iou_var']
        else:
            # 如果没有参数控件，创建默认值
            self.conf_var = None
            self.iou_var = None
    
    def _validate_and_get_params(self):
        """验证并获取检测参数"""
        from .config import DEFAULT_CONF, DEFAULT_IOU
        
        try:
            # 获取conf参数
            if self.conf_var:
                conf_str = self.conf_var.get().strip()
                conf = float(conf_str) if conf_str else DEFAULT_CONF
            else:
                conf = DEFAULT_CONF
            
            # 验证conf范围
            if conf < 0 or conf > 1:
                messagebox.showerror("参数错误", f"Conf参数必须在0-1之间，当前值: {conf}")
                return None, None
            conf = max(0.0, min(1.0, conf))  # 确保在范围内
            
            # 获取iou参数
            if self.iou_var:
                iou_str = self.iou_var.get().strip()
                iou = float(iou_str) if iou_str else DEFAULT_IOU
            else:
                iou = DEFAULT_IOU
            
            # 验证iou范围
            if iou < 0 or iou > 1:
                messagebox.showerror("参数错误", f"IOU参数必须在0-1之间，当前值: {iou}")
                return None, None
            iou = max(0.0, min(1.0, iou))  # 确保在范围内
            
            return conf, iou
            
        except ValueError:
            messagebox.showerror("参数错误", "请输入有效的数字（0-1之间）")
            return None, None
        except Exception as e:
            messagebox.showerror("参数错误", f"读取参数时发生错误: {str(e)}")
            return None, None
    
    def toggle_detection(self):
        """切换检测状态"""
        if self.detection_mode == 'screen':
            if not self.is_running:
                # 获取并验证参数
                conf, iou = self._validate_and_get_params()
                if conf is None or iou is None:
                    return
                
                # 设置参数
                self.detection_processor.set_params(conf, iou)
                self.detection_processor.start_screen_detection()
                self.is_running = True
            else:
                self.stop_detection()
        elif self.detection_mode == 'camera':
            if not self.is_running:
                # 获取并验证参数
                conf, iou = self._validate_and_get_params()
                if conf is None or iou is None:
                    return
                
                # 设置参数
                self.detection_processor.set_params(conf, iou)
                self.detection_processor.start_camera_detection()
                self.is_running = True
            else:
                self.stop_detection()
        elif self.detection_mode == 'file':
            if not self.is_running:
                if self.selected_file_path:
                    # 获取并验证参数
                    conf, iou = self._validate_and_get_params()
                    if conf is None or iou is None:
                        return
                    
                    # 设置参数
                    self.detection_processor.set_params(conf, iou)
                    self.detection_processor.start_file_detection(
                        self.selected_file_path,
                        self._set_detection_file_type,
                        self._set_detection_results,
                        self._set_detection_has_results
                    )
                    self.is_running = True
                else:
                    messagebox.showinfo("提示", "请先选择要检测的文件")
            else:
                self.stop_detection()
    
    def _set_detection_file_type(self, file_type):
        """设置检测文件类型"""
        self.detection_file_type = file_type
    
    def _set_detection_results(self, results, info=None):
        """设置检测结果"""
        self.detection_results = results
        if info is not None:
            self.detection_results_info = info
    
    def _set_detection_has_results(self, has_results):
        """设置是否有检测结果"""
        self.detection_has_results = has_results
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        
        if self.detection_processor:
            self.detection_processor.stop()
        
        # 更新控件状态
        try:
            if 'start' in self.buttons and self.buttons['start'].winfo_exists():
                self.buttons['start'].config(state=tk.NORMAL)
            if 'stop' in self.buttons and self.buttons['stop'].winfo_exists():
                self.buttons['stop'].config(state=tk.DISABLED)
            
            status_label = self.ui_components.get('status_label')
            if status_label and status_label.winfo_exists():
                if self.detection_mode == 'file' and self.selected_file_path:
                    self.gui_updater.update_status(status_label, "已停止，可重新开始检测")
                else:
                    self.gui_updater.update_status(status_label, "已停止")
            
            info_text = self.ui_components.get('info_text')
            if info_text and info_text.winfo_exists():
                self.gui_updater.add_info(info_text, "检测已停止。")
            
            # 清空显示（文件检测模式不清空，保持文件预览）
            video_label = self.ui_components.get('video_label')
            if video_label and video_label.winfo_exists() and self.detection_mode != 'file':
                video_label.config(image='', text="检测已停止")
            
            # 禁用保存按钮（如果存在）
            if 'save' in self.buttons and self.buttons['save'].winfo_exists():
                self.buttons['save'].config(state=tk.DISABLED)
        except:
            pass  # 窗口可能已关闭，忽略错误
    
    def select_file(self):
        """选择文件"""
        file_path = FileHandler.select_file()
        
        if file_path:
            self.selected_file_path = file_path
            info_text = self.ui_components.get('info_text')
            if info_text:
                self.gui_updater.add_info(info_text, f"已选择文件: {os.path.basename(file_path)}")
            
            status_label = self.ui_components.get('status_label')
            if status_label:
                self.gui_updater.update_status(status_label, "已选择文件，点击'开始检测'开始检测")
            
            # 显示文件预览（如果是图片）
            if not FileHandler.is_video_file(file_path):
                img = FileHandler.load_image_preview(file_path)
                if img is not None:
                    video_label = self.ui_components.get('video_label')
                    if video_label:
                        self.gui_updater.update_frame(video_label, img)
    
    def on_file_drop(self, event):
        """处理文件拖拽"""
        file_path = event.data.strip()
        # 移除可能的花括号
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            self.selected_file_path = file_path
            info_text = self.ui_components.get('info_text')
            if info_text:
                self.gui_updater.add_info(info_text, f"已选择文件: {os.path.basename(file_path)}")
            
            status_label = self.ui_components.get('status_label')
            if status_label:
                self.gui_updater.update_status(status_label, "已选择文件，点击'开始检测'开始检测")
            
            # 显示文件预览（如果是图片）
            if not FileHandler.is_video_file(file_path):
                img = FileHandler.load_image_preview(file_path)
                if img is not None:
                    video_label = self.ui_components.get('video_label')
                    if video_label:
                        self.gui_updater.update_frame(video_label, img)
        else:
            messagebox.showerror("错误", "文件不存在")
    
    def save_detection_results(self):
        """保存检测结果"""
        def add_info(msg):
            info_text = self.ui_components.get('info_text')
            if info_text:
                self.gui_updater.add_info(info_text, msg)
        
        def show_message(title, msg):
            messagebox.showinfo(title, msg)
        
        FileHandler.save_detection_results(
            self.detection_results,
            self.detection_file_type,
            self.selected_file_path,
            add_info,
            show_message
        )
    
    def select_save_folder(self):
        """选择保存文件夹"""
        from tkinter import filedialog
        folder = filedialog.askdirectory(
            title="选择保存文件夹",
            initialdir=self.save_dir if os.path.exists(self.save_dir) else "."
        )
        
        if folder:
            self.save_dir = folder
            # 确保文件夹存在
            os.makedirs(self.save_dir, exist_ok=True)
            # 刷新界面显示
            self.show_mode_selection()
            messagebox.showinfo("成功", f"保存文件夹已设置为：\n{os.path.abspath(folder)}")
    
    def _show_tooltip(self, event, text):
        """显示工具提示"""
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        label = tk.Label(tooltip, text=text, background="yellow", 
                        relief="solid", borderwidth=1, font=("Arial", 9))
        label.pack()
        self._tooltip_window = tooltip
    
    def _hide_tooltip(self):
        """隐藏工具提示"""
        if hasattr(self, '_tooltip_window'):
            try:
                self._tooltip_window.destroy()
            except:
                pass
            if hasattr(self, '_tooltip_window'):
                delattr(self, '_tooltip_window')
    
    def on_closing(self):
        """窗口关闭时的处理"""
        # 只有在检测界面时才需要停止检测
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

